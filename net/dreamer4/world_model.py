"""Dreamer4 世界模型 (net/dreamer4/world_model.py)。

对外接口:
    WorldModel — Tokenizer + 时空 Transformer 动力学 + shortcut-forcing 速度头 +
                 reward/cont 头;forward() 给一次形状自洽的前向(编码→上下文→少步流生成→解码);
                 loss() 给世界模型训练损失(重建 + shortcut forcing 流匹配 + 自一致 + 可选 reward/cont,
                 与 net/dreamerv3.WorldModel.loss 同置层先例),训练循环在 train/(离线 VPT:
                 train/minecraft/train_dreamer4;在线 CraftGround:train/craftground/train_dreamer4)。

从 blocks 组装(tokenizer/dynamics 见同包子模块,标量头用 blocks.MLP/DiscDist/Bernoulli)。
设计见 Dreamer 4(2025):连续潜 token + 因果时空 Transformer + shortcut forcing 少步生成。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import MLP, DiscDist, Bernoulli
from net.dreamer4.config import Dreamer4Config
from net.dreamer4.tokenizer import Tokenizer
from net.dreamer4.dynamics import SpaceTimeTransformer, ShortcutHead


class WorldModel(nn.Module):
    """Dreamer4 世界模型。

    Args:
        cfg: Dreamer4Config。
    """

    def __init__(self, cfg: Dreamer4Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = Tokenizer(cfg)
        self.num_tokens = self.tokenizer.num_tokens
        self.dynamics = SpaceTimeTransformer(cfg, self.num_tokens)
        self.shortcut = ShortcutHead(cfg)

        feat_dim = self.num_tokens * cfg.token_dim     # 每帧池化前的展平上下文维
        self.reward = MLP(feat_dim, cfg.reward_bins, hidden=cfg.units, layers=cfg.mlp_layers)
        self.cont = MLP(feat_dim, 1, hidden=cfg.units, layers=cfg.mlp_layers)
        # 变化难度头 D(ctx):预测该转移的归一化流匹配误差(detach 目标,不回传动力学)。
        # 动作信息增益 IG = D(无动作 ctx) − D(带动作 ctx),见 eval_action_ig。
        self.difficulty = MLP(cfg.token_dim, 1, hidden=cfg.units, layers=2)

    def reward_dist(self, feat):
        """池化上下文 → two-hot symexp 奖励分布。"""
        logits = self.reward(feat)
        return DiscDist(logits, device=logits.device)

    def cont_dist(self, feat):
        """池化上下文 → 终止延续伯努利分布。"""
        logits = self.cont(feat)
        return Bernoulli(torch.distributions.independent.Independent(
            torch.distributions.bernoulli.Bernoulli(logits=logits), 1))

    def generate_next(self, context, steps=4):
        """从噪声出发用 shortcut 速度头做 `steps` 步 Euler 流积分,生成下一帧 token。

        Args:
            context: [B, T, S, D] 动力学上下文。
            steps:   Euler 步数(shortcut forcing 支持少步,默认 4)。

        Returns:
            tokens: [B, T, S, D] 生成的 token(τ=1 处)。
        """
        b, t, s, d = context.shape
        x = torch.randn(b, t, s, d, device=context.device)        # τ=0 噪声
        dt = 1.0 / steps
        d_emb = torch.full((b, t, 1), dt, device=context.device)
        for i in range(steps):
            tau = torch.full((b, t, 1), i * dt, device=context.device)
            v = self.shortcut(context, x, tau, d_emb)
            x = x + dt * v
        return x

    def loss(self, image, actions, reward=None, cont=None, mask=None,
             d_min=0.125, sc_weight=1.0, delta_weight=False, hard_weight=0.0):
        """世界模型训练损失(shortcut forcing,Dreamer 4 §世界模型预训练的单循环简化)。

        组成:
          - recon: tokenizer 重建 MSE(tokenizer 只由此项训练;动力学侧 token detach,
            近似论文的两阶段 tokenizer→动力学训练)。
          - flow: 基础流匹配。x_τ=(1-τ)ε+τz₁,速度目标 v*=z₁-ε,在最小步长 d_min 处监督。
          - sc:   自一致(shortcut)。随机较大步长 d,目标 = 两个 d/2 半步的平均速度
            (stop-grad),使少步 Euler 生成可用(I8:长链梯度经 stop-grad 截断)。
          - reward/cont: 可选(在线数据才有);two-hot symexp NLL / 伯努利 NLL,
            与 context[t](已见 o_{≤t}, a_{≤t})对齐 reward[t]/cont[t]。

        Args:
            image:   [B, T, C, H, W] float ∈ [0, 1]。
            actions: [B, T, A] float(one-hot 或连续动作向量)。
            reward:  [B, T-1] float 或 None(转移 t→t+1 的奖励,对齐 context[:, :T-1])。
            cont:    [B, T-1] float ∈ {0,1} 或 None(1=未终止)。
            mask:    [B, T-1] float 或 None;0 处的转移不计 flow/sc 损失(episode 边界)。
            d_min:   基础流匹配的最小步长。
            sc_weight: 自一致项权重。
            hard_weight: >0 时按模型**自身流误差**(detach)对转移做温度化重加权
                (OHEM 式硬样本挖掘,权重 = (err/mean)^hard_weight ∈ [0.25,4])。
                与 delta_weight 的区别:delta_weight 看数据(|Δz|² 变化大小),
                hard_weight 看模型(哪里还没学会)——后者随训练进程自动转移注意,
                是 knowledge/design_learned_attention.md v0 的损失侧实现。

        Returns:
            (total, metrics dict)。
        """
        b, t = image.shape[:2]
        tokens, vq_loss = self.tokenizer.encode(image)
        recon = self.tokenizer.decode(tokens)
        recon_loss = F.mse_loss(recon, self.tokenizer.preprocess_image(image))

        z = tokens.detach()
        # 因果约定:actions[t] 是**离开**帧 t 的动作(转移 t→t+1),ctx[t] 因此
        # 同时看到 z_{≤t} 与 a_{≤t}(含正要执行的 a_t),才可能预测 z[t+1]。
        ctx = self.dynamics(z[:, :-1], actions[:, :-1])
        x1 = z[:, 1:]                                          # 目标:下一帧 token
        t1 = t - 1
        m = mask if mask is not None else torch.ones(b, t1, device=z.device)

        # 逐转移权重:|Δz|² 加权(delta_weight,有界 [0.25,4] 承 I5 精神)× 边界掩码
        if delta_weight:
            dz = ((x1 - z[:, :-1]) ** 2).mean(dim=(-2, -1))            # [B,T1]
            w = (dz / dz.mean().clamp(min=1e-8)).clamp(0.25, 4.0) * m
        else:
            w = m

        def wmean(err_bt):
            return (err_bt * w).sum() / w.sum().clamp(min=1.0)

        # 基础流匹配(d = d_min)
        eps = torch.randn_like(x1)
        tau = torch.rand(b, t1, 1, device=z.device)
        x_tau = (1.0 - tau[..., None]) * eps + tau[..., None] * x1
        d_base = torch.full((b, t1, 1), d_min, device=z.device)
        flow_err = ((self.shortcut(ctx, x_tau, tau, d_base)
                     - (x1 - eps)) ** 2).mean(dim=(-2, -1))            # [B,T1]
        # OHEM 硬样本重加权(hard_weight>0):模型自身误差(detach)做温度化权重,
        # 有界 [0.25,4] 防噪声样本主导;同一 w 作用于 flow 与 sc(同转移同难度)
        if hard_weight > 0:
            e_ = flow_err.detach()
            w = w * (e_ / e_.mean().clamp(min=1e-8)).pow(hard_weight).clamp(0.25, 4.0)
        flow_loss = wmean(flow_err)

        # 自一致:d ∈ {2·d_min..1},目标 = 两个 d/2 半步的平均速度(stop-grad)
        n_lv = max(1, int(torch.log2(torch.tensor(1.0 / d_min)).round().item()))
        lv = torch.randint(0, n_lv, (b, t1, 1), device=z.device).float()
        d_sc = (2.0 ** (-lv)).clamp(min=2 * d_min)             # ≥ 2·d_min
        # τ 取 d 的整数倍且 τ + d ≤ 1(Euler 网格上的自一致)
        n_slot = (1.0 / d_sc).round()
        tau_sc = (torch.rand(b, t1, 1, device=z.device) * n_slot).floor() * d_sc
        tau_sc = tau_sc.clamp(max=1.0 - d_sc)
        x_sc = (1.0 - tau_sc[..., None]) * eps + tau_sc[..., None] * x1
        with torch.no_grad():
            half = d_sc / 2.0
            v1 = self.shortcut(ctx, x_sc, tau_sc, half)
            x_mid = x_sc + half[..., None] * v1
            v2 = self.shortcut(ctx, x_mid, tau_sc + half, half)
            v_sc_target = 0.5 * (v1 + v2)
        sc_err = ((self.shortcut(ctx, x_sc, tau_sc, d_sc)
                   - v_sc_target) ** 2).mean(dim=(-2, -1))
        sc_loss = wmean(sc_err)

        # 变化难度头:回归批内归一化的流误差(目标平稳化;detach 双向隔离——
        # 难度头不改动力学,动力学也不为"显得好预测"作弊)
        d_pred = self.difficulty(ctx.detach().mean(dim=2)).squeeze(-1)  # [B,T1]
        d_tgt = (flow_err / flow_err.mean().clamp(min=1e-8)).detach()
        dif_loss = 0.1 * ((d_pred - d_tgt) ** 2 * m).sum() / m.sum().clamp(min=1.0)

        total = recon_loss + vq_loss + flow_loss + sc_weight * sc_loss + dif_loss
        metrics = {"recon": recon_loss.item(), "flow": flow_loss.item(),
                   "sc": sc_loss.item(), "dif": dif_loss.item()}

        if reward is not None:
            feat = ctx.reshape(b, t1, -1)
            reward_loss = -self.reward_dist(feat).log_prob(
                reward.unsqueeze(-1)).mean()
            cont_loss = -self.cont_dist(feat).log_prob(cont.unsqueeze(-1)).mean()
            total = total + reward_loss + cont_loss
            metrics["reward"] = reward_loss.item()
            metrics["cont"] = cont_loss.item()

        return total, metrics

    @torch.no_grad()
    def eval_next_frame(self, image, actions, gen_steps=4, n_samples=4):
        """holdout 评估:少步流生成下一帧并解码,报告 PSNR(dB,[0,1] 值域)与 Δz 解释方差。

        Returns:
            dict:
              psnr_gen     单样本生成 vs 真值(与历史口径兼容)。
              psnr_genmean n_samples 个流样本的 token 均值解码 vs 真值(扣采样方差)。
              psnr_recon   tokenizer 重建上限。
              psnr_persist 持续性基线(上一帧当预测)。
              ev           Δz 解释方差 = 1 − ‖ẑ−z₁‖²/‖z₀−z₁‖²(token 空间;
                           persistence 恒为 0,>0 即预测了真实变化,与 PSNR 口径互补)。
        """
        def psnr(pred, target):
            mse = F.mse_loss(pred.clamp(0.0, 1.0), target)
            return float(-10.0 * torch.log10(mse.clamp(min=1e-10)))

        tokens, _ = self.tokenizer.encode(image)
        recon = self.tokenizer.decode(tokens) + 0.5
        ctx = self.dynamics(tokens[:, :-1], actions[:, :-1])
        gens = [self.generate_next(ctx, steps=gen_steps) for _ in range(n_samples)]
        gen_img = self.tokenizer.decode(gens[0]) + 0.5
        gen_mean = torch.stack(gens).mean(0)
        x1, x0 = tokens[:, 1:], tokens[:, :-1]
        ev = 1.0 - float(F.mse_loss(gen_mean, x1) / F.mse_loss(x0, x1).clamp(min=1e-10))
        return {
            "psnr_gen": psnr(gen_img, image[:, 1:]),
            "psnr_genmean": psnr(self.tokenizer.decode(gen_mean) + 0.5, image[:, 1:]),
            "psnr_recon": psnr(recon, image),
            "psnr_persist": psnr(image[:, :-1], image[:, 1:]),
            "ev": ev,
        }

    @torch.no_grad()
    def eval_action_ig(self, image, actions):
        """动作信息增益 IG = D(无动作 ctx) − D(带动作 ctx)(难度头单位,批内归一)。

        >0 表示"知道动作让未来更好预测"——动作语义被动力学利用的直接证据;
        persistence 与任何不看动作的模型在此口径下恒为 0。
        """
        tokens, _ = self.tokenizer.encode(image)
        ctx_a = self.dynamics(tokens[:, :-1], actions[:, :-1])
        ctx_0 = self.dynamics(tokens[:, :-1], torch.zeros_like(actions[:, :-1]))
        d_a = self.difficulty(ctx_a.mean(dim=2)).squeeze(-1)
        d_0 = self.difficulty(ctx_0.mean(dim=2)).squeeze(-1)
        return {"ig": float((d_0 - d_a).mean()), "d_act": float(d_a.mean())}

    @torch.no_grad()
    def eval_rollout(self, image, actions, context_len=8, horizon=8, gen_steps=4):
        """K 步开环 rollout:自回归生成 horizon 帧(生成 token 回喂上下文),
        对照 persistence(最后一帧真值复读)。模型优势随步数复利,persistence 线性衰减。

        Returns:
            (psnr_roll [horizon], psnr_persist [horizon])(逐步 PSNR 列表)。
        """
        def psnr(pred, target):
            mse = F.mse_loss(pred.clamp(0.0, 1.0), target)
            return float(-10.0 * torch.log10(mse.clamp(min=1e-10)))

        tokens, _ = self.tokenizer.encode(image)
        toks = tokens[:, :context_len]
        last_real = image[:, context_len - 1]
        roll, persist = [], []
        for k in range(horizon):
            ctx = self.dynamics(toks, actions[:, :context_len + k])
            nxt = self.generate_next(ctx[:, -1:], steps=gen_steps)
            toks = torch.cat([toks, nxt], dim=1)
            tgt = image[:, context_len + k]
            roll.append(psnr(self.tokenizer.decode(nxt)[:, 0] + 0.5, tgt))
            persist.append(psnr(last_real, tgt))
        return roll, persist

    def forward(self, image, actions, gen_steps=4):
        """一次完整前向(形状契约自检用,非训练)。

        Args:
            image:   [B, T, C, H, W] float ∈ [0, 1]。
            actions: [B, T, A] one-hot float。
            gen_steps: shortcut 生成 Euler 步数。

        Returns:
            dict:
                tokens:   [B, T, S, D] 编码 token。
                context:  [B, T, S, D] 动力学上下文。
                next_tokens: [B, T, S, D] shortcut 生成的下一帧 token。
                recon:    [B, T, C, H, W] 重建图像(值域 [-0.5, 0.5])。
                reward:   [B, T, 1] 预测奖励(mode)。
                cont:     [B, T, 1] 预测延续概率。
        """
        tokens, vq_loss = self.tokenizer.encode(image)
        context = self.dynamics(tokens, actions)
        next_tokens = self.generate_next(context, steps=gen_steps)
        recon = self.tokenizer.decode(tokens)
        b, t, s, d = context.shape
        feat = context.reshape(b, t, s * d)
        return {
            "tokens": tokens,
            "context": context,
            "next_tokens": next_tokens,
            "recon": recon,
            "reward": self.reward_dist(feat).mode(),
            "cont": self.cont_dist(feat).mean,
            "vq_loss": vq_loss,
        }
