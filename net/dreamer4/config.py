"""Dreamer4 结构超参 schema(纯 dataclass,无 IO)(net/dreamer4/config.py)。

对外接口:
    Dreamer4Config — tokenizer / 时空 Transformer 动力学 / shortcut-forcing 头 /
                     reward·cont·actor·critic 头的全量结构超参。

数值默认对齐 Hafner 等《Dreamer 4: Training Agents Inside of Scalable World Models》(2025)的
设计要点(连续潜 token tokenizer + 因果时空 Transformer 动力学 + shortcut forcing 少步生成),
为本机可构造规模做了缩放(非论文原尺度)。net/ 不读 yaml:训练端用本 dataclass 构造。
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Dreamer4Config:
    """Dreamer4 全量结构超参。

    Attributes:
        obs_shape:      观测形状 (C, H, W)。Crafter 为 (3, 64, 64)。
        num_actions:    离散动作数。
        token_dim:      每个潜 token 的维度 D。
        enc_depths:     tokenizer 编码器各级通道数(长度 = 下采样级数,决定空间 token 网格)。
        dec_depths:     tokenizer 解码器各级通道数(编码器倒序)。
        conv_kernel:    tokenizer 编/解码卷积核。
        conv_stride:    tokenizer 编/解码每级下/上采样步长。
        dec_min_res:    解码器起始方形分辨率。
        dyn_layers:     时空 Transformer 动力学块数 N(每块含空间注意 + 因果时间注意)。
        dyn_heads:      注意力头数。
        shortcut_hidden: shortcut-forcing 速度头 MLP 宽度。
        shortcut_layers: 速度头 MLP 隐藏层数。
        units:          reward/cont/actor/critic 头 MLP 宽度。
        mlp_layers:     头 MLP 隐藏层数。
        reward_bins:    reward/value 头 two-hot 离散分布桶数。
        use_vq:         True 则在 tokenizer 瓶颈处接 blocks.VectorQuantizer(离散码本)。
        vq_codes:       VQ 码本大小(use_vq 时生效)。
    """
    obs_shape: Tuple[int, int, int] = (3, 64, 64)
    num_actions: int = 17

    # tokenizer
    token_dim: int = 256
    enc_depths: Tuple[int, ...] = (32, 64, 128, 256)
    dec_depths: Tuple[int, ...] = (256, 128, 64, 32)
    conv_kernel: int = 4
    conv_stride: int = 2
    dec_min_res: int = 4

    # 时空 Transformer 动力学
    dyn_layers: int = 6
    dyn_heads: int = 8

    # shortcut forcing 速度头
    shortcut_hidden: int = 512
    shortcut_layers: int = 2

    # 头
    units: int = 512
    mlp_layers: int = 2
    reward_bins: int = 255

    # 可选离散码本 tokenizer
    use_vq: bool = False
    vq_codes: int = 1024
