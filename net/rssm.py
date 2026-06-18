"""帧级 RSSM + 后继特征(Successor Features)世界模型核(纯 net/,无 IO/domain/mock)。

设计与数学见 knowledge/rssm_sf_design.md。要点:
  - 状态 s_t=(h_t 确定递归态, z_t 随机态);h_t=GRU([z_{t-1},a_{t-1}], h_{t-1}) 把全历史积分。
  - 随机态因子化 z=(z_rev 高斯连续, z_inv 离散组),先验 p(z|h)(开环可想象)/后验 q(z|h,e)。
  - decoder-free grounding:feat→ê 预测**冻结**骨干嵌入 e(固定目标 ⇒ 无坍缩平凡解)。
  - 后继特征头 ψ(feat) 由 TD(λ) 拟合(训练侧),数学上是 γ-压缩不动点,承载长程后果。

数值不变量(AGENTS §6):KL/采样/归一化全程 fp32(I4);分母 clamp(min=1e-4)(I1);
高斯 std 有界(I3);递归输入 LayerNorm(I7)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.config import RSSMConfig

EPS = 1e-4


class _MLP(nn.Module):
    """LayerNorm → Linear → SiLU → Linear 小前馈(Shape: [..,in]→[..,out])。"""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


class RSSM(nn.Module):
    """递归状态空间核 + grounding/后继特征头。

    构造形状契约(B=批,T=序列长,E=embed_dim,A=act_dim):
      observe(embeds[B,T,E], actions[B,T-1,A]) → feats[B,T,feat_dim], post_stats, prior_stats, states
      imagine(state, actions[B,H,A])          → feats[B,H,feat_dim]   (先验开环 rollout)
      grounding_head(feat)                      → ê[..,E]
      sf_head(feat)                             → ψ[..,sf_dim]
    """

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.deter = cfg.deter
        self.d_rev = cfg.d_rev
        self.groups = cfg.inv_groups
        self.classes = cfg.inv_classes
        self.stoch_dim = cfg.stoch_dim
        self.feat_dim = cfg.feat_dim
        self.min_std = cfg.min_std
        self.unimix = cfg.unimix

        inv_dim = self.groups * self.classes
        stat_dim = 2 * self.d_rev + inv_dim                 # rev(mean,std) + inv logits
        # 递归输入:[z_{t-1}, a_{t-1}] → hidden → GRUCell → h_t
        self.recur_in = nn.Sequential(
            nn.Linear(self.stoch_dim + cfg.act_dim, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU())
        self.gru = nn.GRUCell(cfg.hidden, self.deter)
        # 先验 p(z|h) / 后验 q(z|h,e)
        self.prior_net = _MLP(self.deter, cfg.hidden, stat_dim)
        self.post_net = _MLP(self.deter + cfg.embed_dim, cfg.hidden, stat_dim)
        # decoder-free grounding 头 + 后继特征头
        self.grounding_head = _MLP(self.feat_dim, cfg.hidden, cfg.embed_dim)
        self.sf_head = _MLP(self.feat_dim, cfg.sf_hidden, cfg.sf_dim)

    # ---- 状态 ----
    def initial(self, B, device):
        """零初始状态 {h[B,deter], z[B,stoch]}。"""
        return {"h": torch.zeros(B, self.deter, device=device),
                "z": torch.zeros(B, self.stoch_dim, device=device)}

    def feat(self, state):
        """state{h,z} → feat[..,feat_dim] = concat(h, z)。"""
        return torch.cat([state["h"], state["z"]], dim=-1)

    # ---- 充分统计量 ----
    def _split_stats(self, raw):
        """raw[..,stat_dim] → dict(rev_mean[..,d_rev], rev_std[..,d_rev], inv_logits[..,G,C])。"""
        rev_mean = raw[..., :self.d_rev].float()
        rev_std = F.softplus(raw[..., self.d_rev:2 * self.d_rev].float()) + self.min_std   # I3 有界正
        inv_logits = raw[..., 2 * self.d_rev:].float().reshape(*raw.shape[:-1], self.groups, self.classes)
        return {"rev_mean": rev_mean, "rev_std": rev_std, "inv_logits": inv_logits}

    def _prior_stats(self, h):
        return self._split_stats(self.prior_net(h))

    def _post_stats(self, h, embed):
        return self._split_stats(self.post_net(torch.cat([h, embed], dim=-1)))

    def _unimix_probs(self, logits):
        """categorical 概率 + 均匀混合(防死类)。[..,G,C] → [..,G,C] fp32。"""
        p = F.softmax(logits.float(), dim=-1)
        return (1.0 - self.unimix) * p + self.unimix / self.classes

    def _sample(self, stats):
        """由统计量采样随机态 z[..,stoch_dim](rev reparam;inv 直通 one-hot)。fp32 采样。"""
        eps = torch.randn_like(stats["rev_std"])
        z_rev = stats["rev_mean"] + stats["rev_std"] * eps                  # reparam
        probs = self._unimix_probs(stats["inv_logits"])                    # [..,G,C]
        idx = torch.multinomial(probs.reshape(-1, self.classes), 1).reshape(probs.shape[:-1])
        onehot = F.one_hot(idx, self.classes).float()
        z_inv = onehot + probs - probs.detach()                            # straight-through
        z_inv = z_inv.reshape(*z_inv.shape[:-2], self.groups * self.classes)
        return torch.cat([z_rev, z_inv], dim=-1).to(stats["rev_mean"].dtype)

    # ---- 单步 ----
    def _recur(self, h, z, action):
        x = self.recur_in(torch.cat([z, action.to(z.dtype)], dim=-1))
        return self.gru(x, h)

    def obs_step(self, prev, prev_action, embed):
        """后验一步:返回 (post_state{h,z}, post_stats, prior_stats)。"""
        h = self._recur(prev["h"], prev["z"], prev_action)
        prior = self._prior_stats(h)
        post = self._post_stats(h, embed)
        z = self._sample(post)
        return {"h": h, "z": z}, post, prior

    def img_step(self, prev, prev_action):
        """先验一步(开环,不看观测):返回 (prior_state{h,z}, prior_stats)。"""
        h = self._recur(prev["h"], prev["z"], prev_action)
        prior = self._prior_stats(h)
        z = self._sample(prior)
        return {"h": h, "z": z}, prior

    # ---- 序列 ----
    def observe(self, embeds, actions):
        """后验滚动整段。

        embeds[B,T,E]、actions[B,T-1,A] → (feats[B,T,feat_dim], post_stats, prior_stats, states)。
        t=0 用零动作起步;states 含 h[B,T,deter]、z[B,T,stoch] 供 imagine 从任意帧续推。
        """
        B, T = embeds.shape[0], embeds.shape[1]
        device = embeds.device
        zero_a = torch.zeros(B, actions.shape[-1], device=device)
        state = self.initial(B, device)
        hs, zs, posts, priors = [], [], [], []
        for t in range(T):
            a_prev = actions[:, t - 1] if t > 0 else zero_a
            state, post, prior = self.obs_step(state, a_prev, embeds[:, t])
            hs.append(state["h"]); zs.append(state["z"])
            posts.append(post); priors.append(prior)
        states = {"h": torch.stack(hs, 1), "z": torch.stack(zs, 1)}
        feats = torch.cat([states["h"], states["z"]], dim=-1)
        return feats, _stack_stats(posts), _stack_stats(priors), states

    def imagine(self, state, actions):
        """先验开环滚动。state{h,z}(单帧),actions[B,H,A] → feats[B,H,feat_dim]。"""
        feats = []
        for k in range(actions.shape[1]):
            state, _ = self.img_step(state, actions[:, k])
            feats.append(self.feat(state))
        return torch.stack(feats, 1)

    # ---- KL(free-bits balanced,数学见设计文档 §3.1)----
    def kl_loss(self, post, prior):
        """free-bits + balancing KL。返回 (kl_loss 标量, kl_value 监控标量)。

        dyn=KL(sg(post)‖prior) 训先验、rep=KL(post‖sg(prior)) 训后验,各按 free_nats 设地板。
        """
        free, ds, rs = self.cfg.free_nats, self.cfg.dyn_scale, self.cfg.rep_scale
        dyn = _kl_total(_sg_stats(post), prior, self).clamp(min=free).mean()
        rep = _kl_total(post, _sg_stats(prior), self).clamp(min=free).mean()
        kl_value = _kl_total(post, prior, self).mean().detach()
        return ds * dyn + rs * rep, kl_value


# ---- 模块级 helper(无状态)----
def _stack_stats(stats_list):
    """list[dict] → dict[stacked over time dim 1]。"""
    keys = stats_list[0].keys()
    return {k: torch.stack([s[k] for s in stats_list], dim=1) for k in keys}


def _sg_stats(stats):
    """stop-grad 一份统计量(供 KL balancing 的非对称项)。"""
    return {k: v.detach() for k, v in stats.items()}


def _gauss_kl(qm, qs, pm, ps):
    """对角高斯 KL(q‖p),按维求和。各项 [..,d] → [..]。fp32(I4),分母 clamp(I1)。"""
    qs2, ps2 = qs.pow(2), ps.pow(2)
    kl = (torch.log(ps.clamp(min=EPS)) - torch.log(qs.clamp(min=EPS))
          + (qs2 + (qm - pm).pow(2)) / (2.0 * ps2.clamp(min=EPS)) - 0.5)
    return kl.sum(dim=-1)


def _cat_kl(q_logits, p_logits, rssm):
    """分组 categorical KL(q‖p),按组与类求和。logits[..,G,C] → [..]。fp32,log 域(I2)。"""
    qp = rssm._unimix_probs(q_logits)
    pp = rssm._unimix_probs(p_logits)
    kl = qp * (torch.log(qp.clamp(min=EPS)) - torch.log(pp.clamp(min=EPS)))
    return kl.sum(dim=(-1, -2))


def _kl_total(q, p, rssm):
    """rev 高斯 KL + inv categorical KL → [..](逐序列步)。"""
    return (_gauss_kl(q["rev_mean"], q["rev_std"], p["rev_mean"], p["rev_std"])
            + _cat_kl(q["inv_logits"], p["inv_logits"], rssm))
