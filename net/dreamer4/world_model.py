"""Dreamer4 世界模型 (net/dreamer4/world_model.py)。

对外接口:
    WorldModel — Tokenizer + 时空 Transformer 动力学 + shortcut-forcing 速度头 +
                 reward/cont 头;forward() 给一次形状自洽的前向(编码→上下文→少步流生成→解码)。

从 blocks 组装(tokenizer/dynamics 见同包子模块,标量头用 blocks.MLP/DiscDist/Bernoulli)。
**仅构建,不训练**:本仓当前不提供 Dreamer4 训练循环(流匹配/想象 actor-critic 待补);
forward 只验证各部件 shape 契约自洽。设计见 Dreamer 4(2025)。
"""
import torch
import torch.nn as nn

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
