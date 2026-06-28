"""PPO + Achievement Distillation 超参数配置 schema (net/ppo_ad/config.py)。

对外接口:
    PPOADConfig — 纯 dataclass,涵盖 IMPALA 模型结构、PPO 与 AD 辅助阶段、训练流程超参。

默认值 1:1 对齐 snu-mllab/Achievement-Distillation 的 configs/ppo_ad.yaml(NeurIPS 2023)。
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple


def _impala_kwargs() -> Dict:
    return {"chans": (64, 128, 128), "outsize": 256, "nblock": 2, "post_pool_groups": 1}


def _init_norm_kwargs() -> Dict:
    # 卷积层:GroupNorm(1),不用 BatchNorm(I7)
    return {"batch_norm": False, "group_norm_groups": 1}


def _dense_init_norm_kwargs() -> Dict:
    # dense/MLP 层:LayerNorm
    return {"layer_norm": True}


@dataclass
class PPOADConfig:
    """PPO + Achievement Distillation 全量超参数。

    Attributes:
        obs_shape:   观测形状 (C, H, W)。
        num_actions: 离散动作数(Crafter=17)。
        hidsize:     编码器投影 / memory state 维度。
        impala_kwargs / init_norm_kwargs / dense_init_norm_kwargs: 见 net.ppo_ad.model。
        temperature: InfoNCE 温度。
        use_memory:  是否启用成就 memory 拼接。
        nstep:       每次 rollout 每 env 步数。
        nproc:       并行环境数。
        nepoch:      训练轮数(总步数 = nstep×nproc×nepoch)。
        gamma / gae_lambda: 折扣与 GAE λ(Crafter 调参:0.95 / 0.65)。
        ppo_nepoch:  每轮 PPO 更新遍数。
        ppo_nbatch:  PPO minibatch 划分数(每 minibatch = nstep×nproc/ppo_nbatch)。
        clip_param / vf_loss_coef / ent_coef: PPO 损失系数。
        lr / max_grad_norm: 优化器与梯度裁剪。
        aux_freq:    每多少次 PPO 更新跑一次辅助蒸馏阶段。
        aux_nepoch:  辅助阶段遍数。
        pi_dist_coef / vf_dist_coef: 辅助阶段对 old-model 的 KL/MSE 蒸馏正则系数。
        save_freq:   每多少 epoch 存一次 checkpoint。
    """
    # 模型结构
    obs_shape: Tuple[int, ...] = (3, 64, 64)
    num_actions: int = 17
    hidsize: int = 1024
    impala_kwargs: Dict = field(default_factory=_impala_kwargs)
    init_norm_kwargs: Dict = field(default_factory=_init_norm_kwargs)
    dense_init_norm_kwargs: Dict = field(default_factory=_dense_init_norm_kwargs)
    temperature: float = 0.1
    use_memory: bool = True

    # rollout / 折扣
    nstep: int = 512
    nproc: int = 8
    nepoch: int = 250
    gamma: float = 0.95
    gae_lambda: float = 0.65

    # PPO 更新
    ppo_nepoch: int = 3
    ppo_nbatch: int = 8
    clip_param: float = 0.2
    vf_loss_coef: float = 0.5
    ent_coef: float = 0.05  # 从 0.01 提升到 0.05，延缓熵坍缩
    lr: float = 3.0e-4
    max_grad_norm: float = 0.5

    # 辅助蒸馏阶段
    aux_freq: int = 8
    aux_nepoch: int = 6
    pi_dist_coef: float = 1.0
    vf_dist_coef: float = 1.0

    # 训练流程
    save_freq: int = 50
