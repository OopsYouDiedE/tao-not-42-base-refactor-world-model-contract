"""DreamerV3 智能体装配 (net/dreamerv3/agent.py)。

对外接口:
    DreamerV3      — 持有 WorldModel + ImagBehavior,提供环境交互用的递归策略 policy()。
    build_dreamerv3 — 由 DreamerV3Config(或字段覆盖)一行构造智能体。

net/ 只装配结构、不含训练循环/优化器/数据加载(那些在 train/crafter/)。设计见 [knowledge/dreamer.md]。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.dreamerv3.config import DreamerV3Config
from net.dreamerv3.world_model import WorldModel
from net.dreamerv3.behavior import ImagBehavior


class DreamerV3(nn.Module):
    """DreamerV3 智能体(世界模型 + 想象 actor-critic)。

    Args:
        cfg: DreamerV3Config。
    """

    def __init__(self, cfg: DreamerV3Config):
        super().__init__()
        self.cfg = cfg
        self.world_model = WorldModel(cfg)
        self.behavior = ImagBehavior(cfg, self.world_model.feat_dim)

    @torch.no_grad()
    def encode_latent(self, obs, state, is_first):
        """观测 → 当前后验 latent(策略与规划器共用的 obs_step 部分)。

        Args:
            obs:      [B, C, H, W] float ∈ [0, 1]。
            state:    (latent dict, prev_action [B, A]) 或 None(轨迹起点)。
            is_first: [B] float(1 = 该 env 刚 reset)。

        Returns:
            latent: 后验状态 dict(字段首维 B)。
        """
        wm = self.world_model
        B = obs.shape[0]
        device = obs.device
        if state is None:
            latent = wm.dynamics.initial(B, device)
            prev_action = torch.zeros(B, self.cfg.num_actions, device=device)
        else:
            latent, prev_action = state
        embed = wm.encoder(wm.preprocess_image(obs))
        latent, _ = wm.dynamics.obs_step(latent, prev_action, embed, is_first)
        return latent

    @torch.no_grad()
    def policy(self, obs, state, is_first, training=True, goal=None):
        """单步递归策略(环境交互用)。

        Args:
            obs:      [B, C, H, W] float ∈ [0, 1]。
            state:    (latent dict, prev_action [B, A]) 或 None(轨迹起点)。
            is_first: [B] float(1 = 该 env 刚 reset)。
            training: True 采样,False 取众数(贪心评估)。
            goal:     use_goal 时为 [B, goal_text_dim] 文本嵌入;否则 None。

        Returns:
            action_idx:   [B] long(供环境 step)。
            action_onehot:[B, A] float。
            state:        更新后的 (latent, action_onehot),喂回下一步。
        """
        latent = self.encode_latent(obs, state, is_first)
        feat = self.world_model.dynamics.get_feat(latent)
        dist = self.behavior.actor_dist(feat, goal)
        action_onehot = dist.sample() if training else dist.mode()
        action_idx = action_onehot.argmax(dim=-1)
        return action_idx, action_onehot, (latent, action_onehot)


def build_dreamerv3(device="cuda", **overrides) -> DreamerV3:
    """构造 DreamerV3 智能体。

    Args:
        device: 目标设备。
        **overrides: 覆盖 DreamerV3Config 任意字段(如 dyn_deter、obs_shape、num_actions)。

    Returns:
        已移到 device 的 DreamerV3。
    """
    cfg = DreamerV3Config(**overrides)
    return DreamerV3(cfg).to(device)
