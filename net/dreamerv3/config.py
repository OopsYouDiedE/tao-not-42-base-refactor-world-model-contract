"""DreamerV3 结构超参 schema(纯 dataclass,无 IO)(net/dreamerv3/config.py)。

对外接口:
    DreamerV3Config — RSSM / 编解码器 / 头 / actor-critic / 想象的全量结构超参。

数值默认对齐 Hafner 等《Mastering Diverse Domains through World Models》(arXiv:2301.04104)
与社区移植 NM512/dreamerv3-torch 的 `configs.yaml` defaults(离散 32×32 隐变量、deter=512、
units=512、horizon=15 等)。net/ 不读 yaml:训练端用本 dataclass 构造,可逐字段覆盖(小尺寸冒烟)。
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DreamerV3Config:
    """DreamerV3 全量结构超参。

    Attributes:
        obs_shape:        观测形状 (C, H, W)。Crafter 为 (3, 64, 64)。
        num_actions:      离散动作数。Crafter 为 17。
        dyn_deter:        RSSM 确定性状态(GRU 隐)维度。
        dyn_stoch:        随机隐变量组数(离散时为类别向量数)。
        dyn_discrete:     每组随机隐变量的类别数(0 = 连续高斯隐,>0 = 离散)。
        dyn_hidden:       RSSM 内部投影 MLP 宽度。
        dyn_rec_depth:    img_step 内 GRU 递推次数(>1 为深递归)。
        unimix_ratio:     离散分布的均匀混合比(防 0 概率,稳定 KL)。
        units:            头/actor/critic MLP 宽度。
        mlp_layers:       头/actor/critic MLP 隐藏层数。
        enc_depths:       ConvEncoder 各级通道数(长度 = 下采样级数)。
        dec_depths:       ConvDecoder 各级通道数(编码器倒序)。
        conv_kernel:      编/解码器卷积核大小。
        conv_stride:      编/解码器每级下/上采样步长。
        dec_min_res:      解码器 Linear 投影后的起始方形分辨率。
        kl_free:          KL free-bits 下界(每序列 nat)。
        kl_dyn_scale:     动力学 KL(stop-grad 后验)权重 β_dyn。
        kl_rep_scale:     表征 KL(stop-grad 先验)权重 β_rep。
        horizon:          想象 rollout 步数 H。
        discount:         折扣 γ(乘到 cont 概率上构成 pcont)。
        disc_lambda:      λ-return 的 λ。
        actor_entropy:    actor 熵正则系数 η。
        actor_grad:       想象策略梯度类型,离散用 "reinforce"。
        value_decay:      慢靶 critic 向在线 critic 的 EMA 混合率(每次更新)。
        reward_bins:      reward/value 头 two-hot 离散分布的桶数。
    """
    obs_shape: Tuple[int, int, int] = (3, 64, 64)
    num_actions: int = 17

    # RSSM
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_hidden: int = 512
    dyn_rec_depth: int = 1
    unimix_ratio: float = 0.01

    # 头 / actor / critic MLP
    units: int = 512
    mlp_layers: int = 2

    # 编/解码器
    enc_depths: Tuple[int, ...] = (32, 64, 128, 256)
    dec_depths: Tuple[int, ...] = (256, 128, 64, 32)
    conv_kernel: int = 4
    conv_stride: int = 2
    dec_min_res: int = 4

    # 世界模型损失
    kl_free: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1

    # 想象 actor-critic
    horizon: int = 15
    discount: float = 0.997
    disc_lambda: float = 0.95
    actor_entropy: float = 3e-4
    actor_grad: str = "reinforce"
    value_decay: float = 0.02
    reward_bins: int = 255

    # ── 文本目标条件化 + 稀疏/密集规划(默认全关 ⇒ vanilla DreamerV3 不变)──────
    # use_goal:        启用 goal 条件化 actor(文本点乘动作打分)。
    # goal_text_dim:   目标文本嵌入维(MiniLM = 384)。
    # goal_dim:        点乘空间维度;0 = 用 units。
    # plan_candidates: 稀疏规划器候选动作序列数 N(Phase 2)。
    # plan_horizon:    候选序列长度 L(Phase 2)。
    # goal_align_coef: 候选打分里 goal 对齐项权重 α(Phase 2)。
    # distill_coef:    规划器选中动作蒸回密集 actor 的损失权重(Phase 2)。
    use_goal: bool = False
    goal_text_dim: int = 384
    goal_dim: int = 0
    plan_candidates: int = 64
    plan_horizon: int = 8
    goal_align_coef: float = 1.0
    distill_coef: float = 0.0
