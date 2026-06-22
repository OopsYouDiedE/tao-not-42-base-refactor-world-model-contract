"""Dreamer4 时空 Transformer 动力学与 shortcut-forcing 速度头 (net/dreamer4/dynamics.py)。

对外接口:
    SpaceTimeTransformer — 因果时空 Transformer:每块 = 帧内空间自注意 + 跨帧因果时间自注意,
                           动作经 AdaLN 式调制注入。给出每 token 的预测上下文。
    ShortcutHead         — shortcut forcing 的流匹配速度头:给定上下文 + 噪声 token + 流时间 τ
                           + 步长 d,预测速度 v(单步/少步生成),训练目标在 train/ 实现。

从 blocks 组装:序列混合用 blocks.MHABlock(空间非因果 / 时间因果可切换),速度头用 blocks.MLP。
设计对齐 Dreamer 4(2025)的可扩展 Transformer 世界模型 + shortcut forcing 少步采样。
"""
import math

import torch
import torch.nn as nn

from blocks import MHABlock, MLP
from net.dreamer4.config import Dreamer4Config


class SpaceTimeTransformer(nn.Module):
    """因果时空 Transformer 动力学骨干。

    每块顺序:动作 AdaLN 调制 → 帧内空间自注意(非因果)→ 跨帧时间自注意(因果)。
    动作以 one-hot 线性嵌入后逐 (B, T) 调制全部空间 token(末层零初始 ⇒ 冷启动恒等)。

    Args:
        cfg: Dreamer4Config。
        num_tokens: 每帧空间 token 数 S(由 tokenizer 决定)。

    Forward:
        tokens:  [B, T, S, D]。
        actions: [B, T, A] one-hot。
        → context: [B, T, S, D](每 token 的预测上下文)。
    """

    def __init__(self, cfg: Dreamer4Config, num_tokens: int):
        super().__init__()
        d = cfg.token_dim
        self.action_proj = nn.Linear(cfg.num_actions, d)
        self.pos_spatial = nn.Parameter(torch.zeros(1, 1, num_tokens, d))
        self.spatial = nn.ModuleList(
            [MHABlock(d, heads=cfg.dyn_heads, causal=False) for _ in range(cfg.dyn_layers)])
        self.temporal = nn.ModuleList(
            [MHABlock(d, heads=cfg.dyn_heads, causal=True) for _ in range(cfg.dyn_layers)])
        self.modulation = nn.ModuleList(
            [nn.Linear(d, 2 * d) for _ in range(cfg.dyn_layers)])
        for m in self.modulation:                      # 零初始 ⇒ 冷启动恒等(AdaLN)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tokens, actions):
        b, t, s, d = tokens.shape
        x = tokens + self.pos_spatial
        a = self.action_proj(actions)                  # [B, T, D]
        for spatial, temporal, mod in zip(self.spatial, self.temporal, self.modulation):
            gamma, beta = mod(a).chunk(2, dim=-1)       # [B, T, D]
            x = x * (1 + gamma[:, :, None]) + beta[:, :, None]
            x = spatial(x.reshape(b * t, s, d)).reshape(b, t, s, d)
            xt = x.permute(0, 2, 1, 3).reshape(b * s, t, d)
            xt = temporal(xt)
            x = xt.reshape(b, s, t, d).permute(0, 2, 1, 3)
        return x


class ShortcutHead(nn.Module):
    """shortcut-forcing 流匹配速度头。

    给定动力学上下文 context、当前流位置的噪声 token x_τ、流时间 τ∈[0,1] 与步长 d∈(0,1],
    预测把 x_τ 推向数据分布的速度 v。单步(d→0)即标准流匹配;较大 d 用 self-consistency
    训练以支持少步生成。流匹配/一致性损失本身落在 train/(本块只给前向速度)。

    Args:
        cfg: Dreamer4Config。

    Forward:
        context: [B, T, S, D]。
        x_tau:   [B, T, S, D](τ 时刻的噪声 token)。
        tau, d:  [B, T, 1](逐帧流时间与步长)。
        → velocity: [B, T, S, D]。
    """

    def __init__(self, cfg: Dreamer4Config):
        super().__init__()
        d = cfg.token_dim
        self.cond = nn.Linear(2, d)                    # (τ, log d) → D
        self.net = MLP(2 * d, d, hidden=cfg.shortcut_hidden,
                       layers=cfg.shortcut_layers)

    def forward(self, context, x_tau, tau, d):
        cond = self.cond(torch.cat([tau, torch.log(d.clamp(min=1e-4))], dim=-1))  # [B,T,D] I1
        h = torch.cat([context, x_tau + cond[:, :, None]], dim=-1)                # [B,T,S,2D]
        return self.net(h)
