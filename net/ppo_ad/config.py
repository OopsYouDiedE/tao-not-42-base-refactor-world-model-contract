"""PPO + Achievement Distillation 超参数配置 schema (net/ppo_ad/config.py)。

对外接口:
    PPOADConfig — 纯 dataclass,涵盖网络结构、PPO、AD 与训练超参。
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PPOADConfig:
    """PPO + Achievement Distillation 的全量超参数配置。

    Attributes:
        encoder_depths: ConvEncoder 各级输出通道数，长度决定下采样级数。
        encoder_kernel: 卷积核大小。
        encoder_stride: 每级步长(同时也是 ConvDecoder 上采样倍率)。
        hidden_dim:     Actor/Critic MLP 隐藏层宽度。
        n_envs:         并行环境数。
        n_steps:        每次 rollout 每个 env 收集的步数。
        n_epochs:       每次 rollout 数据的 PPO 更新轮次。
        minibatch_size: PPO minibatch 大小(= n_envs × n_steps 的因子)。
        gamma:          折扣因子。
        gae_lambda:     GAE λ。
        clip_coef:      PPO clip 系数 ε。
        ent_coef:       熵奖励系数。
        vf_coef:        价值函数损失系数。
        lr:             Adam 学习率。
        max_grad_norm:  梯度裁剪范数上界。
        demo_len:       成就解锁前保留多少步作为 AD 示范。
        ad_buffer_cap:  每个成就最多存储的(obs, action)步数。
        ad_batch_size:  每次 AD BC 损失采样的步数。
        ad_coef:        AD BC 损失在总损失中的权重。
        total_timesteps: 总环境交互步数。
        log_interval:   每隔多少次 update 打印一次日志。
        save_interval:  每隔多少次 update 保存一次 checkpoint。
    """
    # 网络结构
    encoder_depths: Tuple[int, ...] = (16, 32, 48, 64)
    encoder_kernel: int = 3
    encoder_stride: int = 2
    hidden_dim: int = 256

    # PPO 收集
    n_envs: int = 4
    n_steps: int = 512

    # PPO 更新
    n_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    # Achievement Distillation
    demo_len: int = 64
    ad_buffer_cap: int = 100
    ad_batch_size: int = 32
    ad_coef: float = 1.0

    # 训练流程
    total_timesteps: int = 1_000_000
    log_interval: int = 10
    save_interval: int = 100
