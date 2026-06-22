"""Dreamer4 智能体装配 (net/dreamer4/agent.py)。

对外接口:
    Dreamer4       — 持有 WorldModel + 想象 actor-critic 头(策略/价值);**仅构建,不训练**。
    build_dreamer4 — 由 Dreamer4Config(或字段覆盖)一行构造智能体。

net/ 只装配结构、不含训练循环/优化器/数据加载。Dreamer4 的流匹配世界模型训练与想象 actor-critic
训练循环本仓暂未提供(待补),本智能体用于结构构建与 shape 自检。设计见 Dreamer 4(2025)。
"""
import torch
import torch.nn as nn

from blocks import MLP, OneHotDist, DiscDist
from net.dreamer4.config import Dreamer4Config
from net.dreamer4.world_model import WorldModel


class Dreamer4(nn.Module):
    """Dreamer4 智能体(可扩展 Transformer 世界模型 + 想象 actor-critic 头)。

    Args:
        cfg: Dreamer4Config。
    """

    def __init__(self, cfg: Dreamer4Config):
        super().__init__()
        self.cfg = cfg
        self.world_model = WorldModel(cfg)
        feat_dim = self.world_model.num_tokens * cfg.token_dim
        self.actor = MLP(feat_dim, cfg.num_actions, hidden=cfg.units, layers=cfg.mlp_layers)
        self.value = MLP(feat_dim, cfg.reward_bins, hidden=cfg.units, layers=cfg.mlp_layers)

    def actor_dist(self, feat):
        """池化上下文 → 离散动作 OneHot 分布(直通梯度)。"""
        return OneHotDist(self.actor(feat).float(), unimix_ratio=0.01)

    def value_dist(self, feat):
        """池化上下文 → two-hot symexp 价值分布。"""
        logits = self.value(feat)
        return DiscDist(logits, device=logits.device)


def build_dreamer4(device="cuda", **overrides) -> Dreamer4:
    """构造 Dreamer4 智能体。

    Args:
        device: 目标设备。
        **overrides: 覆盖 Dreamer4Config 任意字段。

    Returns:
        已移到 device 的 Dreamer4。
    """
    cfg = Dreamer4Config(**overrides)
    return Dreamer4(cfg).to(device)
