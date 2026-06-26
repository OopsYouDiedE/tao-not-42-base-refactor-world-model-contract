"""PPO + Achievement Distillation 网络模块 (net/ppo_ad/)。

对外接口:
    PPOADModel  — IMPALA 编码器 + memory + FiLM + 成就表示的 PPO+AD 模型。
    PPOADConfig — 超参数配置 dataclass。
"""
from net.ppo_ad.model import PPOADModel
from net.ppo_ad.config import PPOADConfig

__all__ = ["PPOADModel", "PPOADConfig"]
