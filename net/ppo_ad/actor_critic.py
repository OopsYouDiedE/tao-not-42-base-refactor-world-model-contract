"""Crafter 用 PPO Actor-Critic 网络 (net/ppo_ad/actor_critic.py)。

对外接口:
    ActorCritic — ConvEncoder 骨干 + 单隐层 MLP + actor/critic 双头。

Crafter 固定观测: (64, 64, 3) uint8 → 归一化后 (3, 64, 64) float32 [0,1]。
动作空间: Discrete(17)。
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from blocks import ConvEncoder

N_ACTIONS = 17   # Crafter 动作数(固定,见 crafter.Env().action_space)


class ActorCritic(nn.Module):
    """卷积 Actor-Critic,适配 Crafter 64×64 RGB 观测。

    网络结构:
        ConvEncoder → LayerNorm FC → actor Linear | critic Linear

    Args:
        encoder_depths: ConvEncoder 各级通道数,长度 = 下采样级数。
        encoder_kernel: 卷积核大小。
        encoder_stride: 步长(每级空间 ÷ stride)。
        hidden_dim:     FC 隐藏维度。

    Forward:
        obs: (B, 3, H, W), float32, [0, 1]
        → logits: (B, N_ACTIONS), float32
        → value:  (B,), float32
    """

    def __init__(
        self,
        encoder_depths=(16, 32, 48, 64),
        encoder_kernel=3,
        encoder_stride=2,
        hidden_dim=256,
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            in_channels=3,
            depths=encoder_depths,
            kernel=encoder_kernel,
            stride=encoder_stride,
            flatten=True,
        )
        enc_dim = self.encoder.feature_dim((64, 64))
        self.fc = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # I7: 递归/rollout 路径用 LayerNorm
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, N_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

        # 正交初始化:actor 输出层小方差,critic 输出层接近零
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        """obs: (B, 3, H, W) float [0,1] → logits (B, A), value (B,)"""
        feat = self.fc(self.encoder(obs))
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        """采样动作并计算 log_prob、entropy、value。

        Args:
            obs:    (B, 3, H, W) float32.
            action: (B,) long,可选;None 时从分布采样。

        Returns:
            action:   (B,) long
            log_prob: (B,) float32
            entropy:  (B,) float32
            value:    (B,) float32
        """
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
