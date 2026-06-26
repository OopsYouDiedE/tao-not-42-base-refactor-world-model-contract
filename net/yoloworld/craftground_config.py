"""YOLO-World-Dreamer for Craftground 配置 (net/yoloworld/craftground_config.py)。

基于 YoloWorldConfig，针对 Minecraft 1.21 Craftground 调整：
  - 观测: 640x360 (原生 Craftground 分辨率)
  - 动作: 27 维离散动作
  - 成就: Minecraft 官方成就（数量动态获取）
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CraftgroundConfig:
    """YOLO-World-Dreamer for Craftground 超参配置。

    基于 Crafter 配置调整，适配 Minecraft 1.21。
    """

    # ─── 观测与动作 ──────────────────────────────────────────────
    obs_shape: Tuple[int, int, int] = (3, 384, 640)  # 填充后的分辨率（384 = ceil(360/32)*32）
    obs_shape_native: Tuple[int, int, int] = (3, 360, 640)  # Craftground 原生输出
    num_actions: int = 27  # Minecraft Java 标准离散动作
    n_achievements: int = 100  # 占位符，实际从 train/craftground_minecraft_ml_env/achievements.py 动态获取

    # ─── RSSM 世界模型 ──────────────────────────────────────────
    dyn_deter: int = 512  # 确定性状态维
    dyn_stoch: int = 32  # 随机隐变量组数
    dyn_discrete: int = 32  # 每组类别数
    dyn_hidden: int = 512  # RSSM 内部投影宽度
    dyn_rec_depth: int = 1  # GRU 递推次数
    unimix_ratio: float = 0.01  # 离散分布均匀混合比

    # ─── 编码器/解码器 ──────────────────────────────────────────
    units: int = 512  # MLP 宽度
    mlp_layers: int = 2  # MLP 隐藏层数
    enc_depths: Tuple[int, ...] = (32, 64, 128, 256)  # ConvEncoder 通道数（针对 640x360）
    dec_depths: Tuple[int, ...] = (256, 128, 64, 32)  # ConvDecoder 通道数（倒序）
    conv_kernel: int = 4  # 卷积核大小
    conv_stride: int = 2  # 步长
    dec_min_res: int = 4  # 解码器起始分辨率

    # ─── 损失权重 ──────────────────────────────────────────────
    kl_free: float = 0.0  # KL free-bits
    kl_dyn_scale: float = 0.5  # 动力学 KL 权重
    kl_rep_scale: float = 0.1  # 表征 KL 权重
    reward_bins: int = 255  # Two-hot 桶数
    ach_scale: float = 1.0  # 成就头 BCE 权重

    # ─── 候选规划（行为线）──────────────────────────────────────
    n_candidates: int = 256  # 候选动作序列数
    plan_horizon: int = 15  # 计划长度
    query_dim: int = 128  # Query 向量维
    head_hidden: int = 256  # 小头解码器宽度

    # ─── Rollout 老师 ──────────────────────────────────────────
    n_rollout: int = 64  # Top-M rollout 实际数
    n_explore: int = 16  # 额外随机候选数
    n_start: int = 0  # 起点状态数（0 = 全部）
    teacher_temp: float = 1.0  # 老师温度
    select_beta: float = 1.0  # 选择点乘系数

    # ─── 强化学习超参 ──────────────────────────────────────────
    discount: float = 0.99  # 折扣率
    disc_lambda: float = 0.95  # λ-return 的 λ
    value_decay: float = 0.99  # 慢靶 EMA 混合率
    actor_entropy: float = 0.001  # 计划熵正则

    # ─── PPO+AD 超参 ───────────────────────────────────────────
    ppo_epochs: int = 4  # PPO 优化 epoch
    ppo_clip: float = 0.2  # PPO clip 范围
    ppo_gae_lambda: float = 0.95  # GAE λ
    ppo_ent_coeff: float = 0.01  # 熵系数
    ad_scale: float = 0.5  # Achievement Distillation 权重

    # ─── 成就相关 ──────────────────────────────────────────────
    cls_scale: float = 1.0  # L_cls 权重
    plan_scale: float = 1.0  # L_plan 权重
    align_scale: float = 0.5  # L_align 权重
    div_scale: float = 0.1  # L_div 权重（反候选坍缩）
    load_scale: float = 0.1  # L_load 权重（反坍缩）


# 便捷预设
CRAFTGROUND_DEFAULT = CraftgroundConfig()
