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
             d_min=0.125, sc_weight=1.0):
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

        Returns:
            (total, metrics dict)。
        """
        b, t = image.shape[:2]
        tokens, vq_loss = self.tokenizer.encode(image)
        recon = self.tokenizer.decode(tokens)
        recon_loss = F.mse_loss(recon, self.tokenizer.preprocess_image(image))

        z = tokens.detach()
        ctx = self.dynamics(z[:, :-1], actions[:, :-1])       # context[t] ← z/a_{≤t}
        x1 = z[:, 1:]                                          # 目标:下一帧 token
        t1 = t - 1
        m = mask[..., None, None] if mask is not None else torch.ones(
            b, t1, 1, 1, device=z.device)

        def masked_mse(a_, b_):
            return ((a_ - b_) ** 2 * m).sum() / (m.sum() * a_.shape[-2] * a_.shape[-1]).clamp(min=1.0)

        # 基础流匹配(d = d_min)
        eps = torch.randn_like(x1)
        tau = torch.rand(b, t1, 1, device=z.device)
        x_tau = (1.0 - tau[..., None]) * eps + tau[..., None] * x1
        d_base = torch.full((b, t1, 1), d_min, device=z.device)
        flow_loss = masked_mse(self.shortcut(ctx, x_tau, tau, d_base), x1 - eps)

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
        sc_loss = masked_mse(self.shortcut(ctx, x_sc, tau_sc, d_sc), v_sc_target)

        total = recon_loss + vq_loss + flow_loss + sc_weight * sc_loss
        metrics = {"recon": recon_loss.item(), "flow": flow_loss.item(),
                   "sc": sc_loss.item()}

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
    def eval_next_frame(self, image, actions, gen_steps=4):
        """holdout 评估:少步流生成下一帧并解码,报告 PSNR(dB,[0,1] 值域)。

        Returns:
            dict: psnr_gen(生成的下一帧 vs 真值)、psnr_recon(tokenizer 重建)、
                  psnr_persist(基线:上一帧当预测)。生成质量必须同时对照重建上限
                  与持续性基线才有意义。
        """
        def psnr(pred, target):
            mse = F.mse_loss(pred.clamp(0.0, 1.0), target)
            return float(-10.0 * torch.log10(mse.clamp(min=1e-10)))

        tokens, _ = self.tokenizer.encode(image)
        recon = self.tokenizer.decode(tokens) + 0.5
        ctx = self.dynamics(tokens[:, :-1], actions[:, :-1])
        gen = self.generate_next(ctx, steps=gen_steps)
        gen_img = self.tokenizer.decode(gen) + 0.5
        return {
            "psnr_gen": psnr(gen_img, image[:, 1:]),
            "psnr_recon": psnr(recon, image),
            "psnr_persist": psnr(image[:, :-1], image[:, 1:]),
        }

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
