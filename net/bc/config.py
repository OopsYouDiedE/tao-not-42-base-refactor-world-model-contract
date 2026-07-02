"""VPT 行为克隆策略的结构配置(纯 dataclass,无 IO,不 import 数据域)。

领域常量(动作维数/相机分箱数)由 train/ 侧传入——net/ 只认整数,
与 train/minecraft/vpt_action.py 的 ACTION_DIM/CAMERA_BINS 对齐与否由调用方保证。
"""
from dataclasses import dataclass, field

from net.config import BackboneConfig


@dataclass
class BCConfig:
    """BCPolicy 结构超参。

    Attributes
    ----------
    backbone : BackboneConfig
        冻结视觉骨干选择(dinov3/dinov2;mock 经依赖注入,见 tests/)。
    d : int
        时序骨干宽度(骨干特征经线性投影到该维度)。
    heads : int
        自注意力头数(须整除 d)。
    layers : int
        因果 Transformer 块数(MHABlock + FFN 为一块)。
    dropout : float
        注意力与 FFN dropout。
    max_len : int
        学习式位置编码支持的最大序列长度(训练 seq_len 不得超过)。
    action_dim : int
        动作向量维数(相机 2 维 + 二值键;由 train 侧传入)。
    n_mouse : int
        动作向量头部连续相机维数。
    camera_bins : int
        相机 mu-law 分箱数(逆动力学监督目标;由 train 侧传入)。
    """
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    d: int = 384
    heads: int = 6
    layers: int = 2
    dropout: float = 0.1
    max_len: int = 128
    action_dim: int = 22
    n_mouse: int = 2
    camera_bins: int = 11
