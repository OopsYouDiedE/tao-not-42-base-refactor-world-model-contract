"""Craftground PPO+AD 训练配置 (train/craftground/training_config.py)。

关键参数：
  1. 帧率 (Frame Rate): 20 Hz（Minecraft 原生 tick rate）
  2. 动作序列长度 (Action Sequence): 多少步后更新
  3. 成就检测频率
  4. Rollout 缓冲大小
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CraftgroundTrainingConfig:
    """Craftground PPO+AD 训练超参。"""

    # ─── 环境参数 ──────────────────────────────────────────────────
    frame_rate: int = 20  # Minecraft 原生 tick rate (Hz)
    frame_skip: int = 1  # Frame skip（每 N 个 tick 执行一次动作）
    actual_frame_rate: int = frame_rate // frame_skip  # 实际控制频率

    # ─── 数据收集 ──────────────────────────────────────────────────
    n_envs: int = 16  # 并行环境数
    n_rollout_steps: int = 512  # 每次 rollout 收集的步数
                                  # （在此期间收集成就信息）
    max_episode_steps: int = 1000  # 单个 episode 的最大步数

    # ─── PPO 超参 ───────────────────────────────────────────────────
    ppo_epochs: int = 4  # PPO 优化 epoch（单次 rollout 后）
    ppo_batch_size: int = 256  # PPO mini-batch 大小
    ppo_clip: float = 0.2  # PPO clip ratio
    ppo_value_coeff: float = 0.5  # 值函数损失系数
    ppo_entropy_coeff: float = 0.01  # 熵正则系数
    gae_lambda: float = 0.95  # GAE λ

    # ─── Achievement Distillation (AD) ──────────────────────────────
    ad_enabled: bool = True  # 启用 AD
    ad_scale: float = 1.0  # AD 损失权重
    ad_reward_per_achievement: float = 1.0  # 每个新成就的奖励

    # ─── 学习率 ──────────────────────────────────────────────────────
    lr: float = 3e-4  # 初始学习率
    lr_decay: str = "cosine"  # 学习率衰减方式：linear / cosine / none

    # ─── 世界模型 (RSSM) ─────────────────────────────────────────
    world_model_enabled: bool = True  # 启用世界模型
    world_model_loss_scale: float = 0.5  # 世界模型损失权重

    # ─── 种子与重现性 ──────────────────────────────────────────────
    seed: int = 42  # 初始种子（用于验证）
    use_fixed_seed: bool = True  # 是否固定种子

    # ─── 地形检测 ───────────────────────────────────────────────────
    use_terrain_check: bool = True  # 启用地形检测（避免不利地形）
    favorable_biomes: Tuple[str, ...] = ("forest", "plains", "taiga")
    unfavorable_biomes: Tuple[str, ...] = ("ocean", "deep_ocean", "stone_shore")

    # ─── 日志与保存 ──────────────────────────────────────────────────
    log_interval: int = 10  # 日志输出间隔（更新次数）
    checkpoint_interval: int = 100  # 检查点保存间隔
    save_dir: str = "runs/craftground_ppo_ad_v1"

    # ─── 总训练步数 ──────────────────────────────────────────────────
    total_timesteps: int = 1_000_000  # 总训练步数（环境 steps）

    def __post_init__(self):
        """验证配置的一致性。"""
        self.actual_frame_rate = self.frame_rate // self.frame_skip

        # 计算每个 PPO 更新的总数据量
        self.total_ppo_steps = (self.total_timesteps + self.n_rollout_steps - 1) // self.n_rollout_steps

        print(f"🔧 训练配置验证:")
        print(f"   - 实际控制频率: {self.actual_frame_rate} Hz ({self.frame_skip} frame skip)")
        print(f"   - 单次 rollout: {self.n_rollout_steps} 步 × {self.n_envs} 环境 = {self.n_rollout_steps * self.n_envs} 样本")
        print(f"   - 总 PPO 更新: ~{self.total_ppo_steps} 次")
        print(f"   - 地形检测: {'启用' if self.use_terrain_check else '禁用'}")


# 默认配置
DEFAULT_CONFIG = CraftgroundTrainingConfig()

# 快速验证配置
QUICK_TEST_CONFIG = CraftgroundTrainingConfig(
    n_envs=4,
    n_rollout_steps=128,
    total_timesteps=10_000,
    use_fixed_seed=True,
)
