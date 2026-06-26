"""YOLO-World-Dreamer 结构超参 schema(纯 dataclass,无 IO)(net/yoloworld/config.py)。

对外接口:
    YoloWorldConfig — RSSM 世界模型 + 成就头 + 256 候选小头 + rollout 老师 + 目标条件 critic
                      的全量结构超参。

世界模型部分默认对齐 DreamerV3(arXiv:2301.04104):能学会 Crafter 的 `crafter` 预设为
deter=512 / 离散 32×32 / units=512。行为双头部分为本设计新增(见 knowledge/yoloworld.md)。
net/ 不读 yaml:训练端用本 dataclass 构造,可逐字段覆盖(小尺寸冒烟)。
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class YoloWorldConfig:
    """YOLO-World-Dreamer 全量结构超参。

    Attributes:
        obs_shape:        观测 (C, H, W),Crafter (3, 64, 64)。
        num_actions:      离散动作数 A,Crafter 17。
        n_achievements:   成就数 U,Crafter 22(成就头维度与 E 行数)。

        dyn_deter:        RSSM 确定性状态维 D。
        dyn_stoch:        随机隐变量组数 S。
        dyn_discrete:     每组类别数 C(0 = 连续高斯)。
        dyn_hidden:       RSSM 内部投影宽度。
        dyn_rec_depth:    img_step 内 GRU 递推次数。
        unimix_ratio:     离散分布均匀混合比。

        units:            头/critic MLP 宽度。
        mlp_layers:       头/critic MLP 隐藏层数。
        enc_depths:       ConvEncoder 各级通道。
        dec_depths:       ConvDecoder 各级通道(编码器倒序)。
        conv_kernel:      编/解码器卷积核。
        conv_stride:      编/解码器每级步长。
        dec_min_res:      解码器起始方形分辨率。

        kl_free:          KL free-bits 下界(nat)。
        kl_dyn_scale:     动力学 KL 权重 β_dyn。
        kl_rep_scale:     表征 KL 权重 β_rep。
        reward_bins:      reward/value two-hot 桶数。
        ach_scale:        成就头 BCE 权重(L_ach)。

        task_dim:         任务句向量维 d_g(MiniLM 384)。
        task_proj_dim:    任务投影维 d_g'(小头/critic 内部条件维)。
        shaping_tau:      w(g)=softmax(E·g/τ) 的温度 τ。

        n_candidates:     候选动作序列数 K。
        plan_horizon:     计划/想象步长 H。
        query_dim:        小头 query 向量维 d_q。
        head_hidden:      小头共享解码器宽度。

        n_rollout:        rollout 老师按 α 预排序实际滚动的候选数 M(top-M)。
        n_explore:        额外随机候选数(覆盖探索,计入 M 之外)。
        n_start:          行为线每次子采样的起点状态数(控 CPU 算力;0 = 用全部 B·T)。
        teacher_temp:     老师软信念 t=softmax(R/η) 的温度 η。
        select_beta:      选择/混合权重里点乘项系数 β:α=softmax(p+β·e·g)。

        discount:         折扣 γ。
        disc_lambda:      λ-return 的 λ。
        value_decay:      慢靶 critic EMA 混合率。
        actor_entropy:    计划熵正则系数 λ_H。
        cls_scale:        L_cls 权重。
        plan_scale:       L_plan 权重。
        align_scale:      L_align 权重。
        div_scale:        L_div 权重(slot 嵌入互斥,反候选坍缩 → 语义多样)。
        load_scale:       L_load 权重(batch 选择负熵,均衡 slot 使用 → 反坍缩)。
    """
    obs_shape: Tuple[int, int, int] = (3, 64, 64)
    num_actions: int = 17
    n_achievements: int = 22

    # RSSM
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_hidden: int = 512
    dyn_rec_depth: int = 1
    unimix_ratio: float = 0.01

    # 头 / critic MLP
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
    reward_bins: int = 255
    ach_scale: float = 1.0

    # 任务语言条件
    task_dim: int = 384
    task_proj_dim: int = 64
    shaping_tau: float = 0.1

    # 候选小头(YOLO 稀疏头)
    n_candidates: int = 256
    plan_horizon: int = 16
    query_dim: int = 64
    head_hidden: int = 256

    # rollout 老师 / 行为线
    n_rollout: int = 32
    n_explore: int = 8
    n_start: int = 0
    teacher_temp: float = 1.0
    select_beta: float = 1.0

    # actor-critic
    discount: float = 0.997
    disc_lambda: float = 0.95
    value_decay: float = 0.02
    actor_entropy: float = 3e-4
    cls_scale: float = 1.0
    plan_scale: float = 1.0
    align_scale: float = 1.0
    div_scale: float = 0.1
    load_scale: float = 0.01

    @property
    def feat_dim(self) -> int:
        """世界状态特征维 d_φ = S·C + D(离散)或 S + D(连续)。"""
        stoch_flat = (self.dyn_stoch * self.dyn_discrete if self.dyn_discrete
                      else self.dyn_stoch)
        return stoch_flat + self.dyn_deter
