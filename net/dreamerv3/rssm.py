"""DreamerV3 循环状态空间模型 RSSM (net/dreamerv3/rssm.py)。

对外接口:
    RSSM — 确定性 GRU 状态 + 离散/连续随机隐变量的递归世界状态;
           observe(后验序列)/ imagine(先验 rollout)/ kl_loss(动力学/表征 KL)。

从 blocks 组装:递归用 blocks.GRUCell(LayerNorm + 凸更新,I5/I7 安全),
离散隐变量分布用 blocks.OneHotDist(unimix + 直通梯度),序列展开用 blocks.static_scan。
状态以 dict {logit/mean/std, stoch, deter} 表示;特征 = cat(stoch_flat, deter)。
设计与算法见 [knowledge/dreamer.md];结构对齐 Hafner 等 arXiv:2301.04104。
"""
import torch
import torch.nn as nn
from torch import distributions as torchd

from blocks import GRUCell, OneHotDist, static_scan


class RSSM(nn.Module):
    """循环状态空间模型(确定性 deter + 随机 stoch)。

    Args:
        stoch: 随机隐变量组数。
        deter: 确定性 GRU 隐状态维。
        hidden: 内部投影 MLP 宽度。
        discrete: 每组类别数(0 = 连续高斯,>0 = 离散 one-hot)。
        num_actions: 动作维(拼接到 img_step 输入)。
        embed_dim: 观测嵌入维(拼接到 obs_step 输入)。
        rec_depth: img_step 内 GRU 递推次数。
        unimix_ratio: 离散分布均匀混合比。
        min_std: 连续高斯隐变量的标准差下界(I1)。

    状态张量约定(B 为批,离散时 stoch 维含两轴):
        deter:  [B, deter]
        stoch:  [B, stoch, discrete](离散)| [B, stoch](连续)
        logit:  [B, stoch, discrete](离散)
        mean/std: [B, stoch](连续)
    """

    def __init__(self, stoch=32, deter=512, hidden=512, discrete=32,
                 num_actions=17, embed_dim=4096, rec_depth=1,
                 unimix_ratio=0.01, min_std=0.1):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._rec_depth = rec_depth
        self._unimix_ratio = unimix_ratio
        self._min_std = min_std
        stoch_flat = stoch * discrete if discrete else stoch

        def block(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden, bias=False),
                nn.LayerNorm(hidden, eps=1e-3),
                nn.SiLU(),
            )

        # 先验链:[stoch_flat + action] → hidden → GRUCell → deter → hidden → 统计层
        self._img_in = block(stoch_flat + num_actions)
        self._cell = GRUCell(hidden, deter, norm=True)
        self._img_out = block(deter)
        # 后验链:[deter + embed] → hidden → 统计层
        self._obs_out = block(deter + embed_dim)

        stat_out = stoch * discrete if discrete else 2 * stoch
        self._img_stat = nn.Linear(hidden, stat_out)
        self._obs_stat = nn.Linear(hidden, stat_out)
        # 学习型初始 deter(经 tanh 有界)
        self.W = nn.Parameter(torch.zeros(1, deter))

    # ── 状态工具 ──────────────────────────────────────────────────────────────
    def initial(self, batch_size, device):
        """零观测起点状态 dict。deter 由学习参数 tanh 后广播。"""
        deter = torch.tanh(self.W).repeat(batch_size, 1)
        if self._discrete:
            state = dict(
                logit=torch.zeros(batch_size, self._stoch, self._discrete, device=device),
                stoch=torch.zeros(batch_size, self._stoch, self._discrete, device=device),
                deter=deter.to(device),
            )
        else:
            state = dict(
                mean=torch.zeros(batch_size, self._stoch, device=device),
                std=torch.zeros(batch_size, self._stoch, device=device),
                stoch=torch.zeros(batch_size, self._stoch, device=device),
                deter=deter.to(device),
            )
        return state

    def get_feat(self, state):
        """特征 = cat(stoch_flat, deter),[B, stoch_flat + deter]。"""
        stoch = state["stoch"]
        if self._discrete:
            stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, state["deter"]], dim=-1)

    def get_dist(self, state):
        """随机隐变量分布(离散 OneHot+unimix / 连续对角高斯),含独立事件维。"""
        if self._discrete:
            logit = state["logit"].float()
            return torchd.Independent(
                OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)
        mean, std = state["mean"].float(), state["std"].float()
        return torchd.Independent(torchd.Normal(mean, std), 1)

    def _suff_stats(self, x, stat_layer):
        x = stat_layer(x)
        if self._discrete:
            logit = x.reshape(*x.shape[:-1], self._stoch, self._discrete)
            return {"logit": logit}
        mean, std = torch.chunk(x, 2, dim=-1)
        std = 2.0 * torch.sigmoid(std / 2.0) + self._min_std   # I3 有界,I1 下界
        return {"mean": mean, "std": std}

    # ── 单步 ──────────────────────────────────────────────────────────────────
    def img_step(self, prev_state, prev_action, sample=True):
        """先验一步:由 (prev_stoch, action) 经 GRU 推 deter 并预测先验 stoch。"""
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            prev_stoch = prev_stoch.reshape(
                *prev_stoch.shape[:-2], self._stoch * self._discrete)
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self._img_in(x)
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, deter_list = self._cell(x, [deter])
            deter = deter_list[0]
        x = self._img_out(x)
        stats = self._suff_stats(x, self._img_stat)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        return {"stoch": stoch.to(prev_action.dtype), "deter": deter, **stats}

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """后验一步:先走先验得 deter,再融合观测 embed 预测后验 stoch。

        is_first: [B] float(1=轨迹起点),为 1 的样本把 prev_state/action 清零重置。
        """
        if prev_state is None or torch.all(is_first.bool()):
            prev_state = self.initial(is_first.shape[0], is_first.device)
            prev_action = torch.zeros(
                (*is_first.shape, prev_action.shape[-1]),
                device=is_first.device, dtype=embed.dtype)
        elif torch.any(is_first.bool()):
            mask = (1.0 - is_first)[..., None]
            prev_action = prev_action * mask
            init = self.initial(is_first.shape[0], is_first.device)
            for k, v in prev_state.items():
                m = mask.reshape(mask.shape[0], *([1] * (v.dim() - 1)))
                prev_state[k] = v * m + init[k] * (1.0 - m)

        prior = self.img_step(prev_state, prev_action, sample=sample)
        x = torch.cat([prior["deter"], embed], dim=-1)
        x = self._obs_out(x)
        stats = self._suff_stats(x, self._obs_stat)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": stoch.to(embed.dtype), "deter": prior["deter"], **stats}
        return post, prior

    # ── 序列 ──────────────────────────────────────────────────────────────────
    def observe(self, embed, action, is_first, state=None):
        """沿时间展开后验/先验序列。

        Args:
            embed:    [B, T, embed_dim]。
            action:   [B, T, num_actions] one-hot/连续。
            is_first: [B, T] float。
            state:    起点状态 dict,None 则用 initial。

        Returns:
            post, prior — 各为 dict,字段张量首维为 B,次维为 T。
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, x.dim())))
        embed_t, action_t, isfirst_t = swap(embed), swap(action), swap(is_first)
        if state is None:
            state = self.initial(action.shape[0], action.device)

        def step(prev, action_s, embed_s, isfirst_s):
            post, prior = self.obs_step(prev[0], action_s, embed_s, isfirst_s)
            return post, prior

        post, prior = static_scan(
            step, (action_t, embed_t, isfirst_t), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        """从给定状态沿动作序列做先验 rollout。

        Args:
            action: [B, T, num_actions]。
            state:  起点状态 dict(字段首维 B)。

        Returns:
            prior — dict,字段首维 B、次维 T。
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, x.dim())))
        action_t = swap(action)
        prior = static_scan(
            lambda prev, act: self.img_step(prev, act), (action_t,), state)[0]
        return {k: swap(v) for k, v in prior.items()}

    # ── 损失 ──────────────────────────────────────────────────────────────────
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """动力学/表征 KL(free-bits 截断)。

        dyn: 把先验拉向 stop-grad 后验(学动力学);rep: 把后验拉向 stop-grad 先验(正则表征)。

        Returns:
            loss:  加权标量 β_dyn·dyn + β_rep·rep。
            dyn, rep: 各项 free-bits 后的标量(监控用)。
        """
        kld = torchd.kl.kl_divergence
        sg = lambda d: {k: v.detach() for k, v in d.items()}
        dyn = kld(self.get_dist(sg(post)), self.get_dist(prior))
        rep = kld(self.get_dist(post), self.get_dist(sg(prior)))
        dyn = torch.clip(dyn, min=free).mean()
        rep = torch.clip(rep, min=free).mean()
        loss = dyn_scale * dyn + rep_scale * rep
        return loss, dyn, rep
