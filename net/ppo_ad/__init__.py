"""PPO + Achievement Distillation 网络模块 (net/ppo_ad/)。

对外接口:
    ActorCritic — 卷积 Actor-Critic 双头网络。
    PPOADConfig — 超参数配置 dataclass。
"""
from net.ppo_ad.actor_critic import ActorCritic, N_ACTIONS
from net.ppo_ad.config import PPOADConfig

__all__ = ["ActorCritic", "N_ACTIONS", "PPOADConfig"]
