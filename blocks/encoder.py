"""L1 primitive 积木库 - 卷积图像编码器 (blocks/encoder.py)。

对外接口:
    ConvEncoder — DreamerV3/DRAMA 式步长卷积编码器(逐级 ×stride 下采样、通道按 depths 递增)。

复用 blocks/conv.py 的 Conv2dSamePad(TF-"same" 动态填充)与 ImgChLayerNorm(通道维 LayerNorm)。
本块只产出连续视觉特征,**不含离散化**——categorical/VQ 量化由 blocks.regularization.StochLatent、
blocks.distributions.OneHotDist 或 blocks.quantization.VectorQuantizer 在外部组合(职责分离)。
所有尺寸/层数/卷积核/步长/激活/归一化经构造参数注入,**不写死**;默认值对齐 DRAMA 离散VAE 编码器
(depths=(16,32,48,64,64), kernel=5, SiLU, 开 BatchNorm→此处用通道 LayerNorm 替代以承袭仓内 I7 约定)。
"""
import math

import torch
import torch.nn as nn

from blocks.conv import Conv2dSamePad, ImgChLayerNorm


class ConvEncoder(nn.Module):
    """步长卷积图像编码器。

    每个 stage: Conv2dSamePad(stride 下采样) → [通道 LayerNorm] → 激活。
    输入接受任意前导维 [..., C, H, W](展平后过卷积再还原),故 [B,C,H,W] 与
    [B,T,C,H,W] 均可直接喂入,时间维不写死。

    Args:
        in_channels: 输入图像通道数。
        depths: 每级输出通道数的序列;其长度 = 下采样级数(分辨率 ÷ stride^len)。
        kernel: 卷积核大小(same-pad,不随 stride 改变语义)。
        stride: 每级下采样步长。
        act: 激活层**类**(可调用,无参构造),默认 nn.SiLU。
        norm: 是否在每级卷积后插入通道维 LayerNorm(I7)。
        flatten: True 则前向把空间维展平成向量 [..., C'·H'·W'];False 返回特征图 [..., C', H', W']。
    """

    def __init__(self, in_channels=3, depths=(16, 32, 48, 64, 64), kernel=5,
                 stride=2, act=nn.SiLU, norm=True, flatten=True):
        super().__init__()
        self.depths = tuple(depths)
        self.stride = stride
        self.flatten = flatten
        self.out_channels = self.depths[-1]
        layers = []
        c = in_channels
        for d in self.depths:
            layers.append(Conv2dSamePad(c, d, kernel, stride=stride, bias=not norm))
            if norm:
                layers.append(ImgChLayerNorm(d))
            layers.append(act())
            c = d
        self.layers = nn.Sequential(*layers)

    def feature_dim(self, in_hw):
        """给定输入 (H, W),返回展平后的特征维 C'·H'·W'(供下游 Linear/量化器对接,不写死)。"""
        h, w = in_hw
        for _ in self.depths:
            h = math.ceil(h / self.stride)
            w = math.ceil(w / self.stride)
        return self.out_channels * h * w

    def forward(self, x):
        *lead, c, h, w = x.shape
        x = self.layers(x.reshape(-1, c, h, w))            # [N, C', H', W']
        if self.flatten:
            x = x.reshape(*lead, -1)                        # [..., C'·H'·W']
        else:
            x = x.reshape(*lead, *x.shape[1:])             # [..., C', H', W']
        return x


def pool9(z, grid=9, out=3):
    """token 网格空间均池:[B, grid², C] → [B, out²·C]。

    默认 9×9→3×3(每 3×3 patch 块取均值),压维但保留空间粗布局。要求 grid 能被
    out 整除。与任务无关的纯空间降维算子,不含领域字眼。

    Shape: z [B, grid*grid, C] → [B, out*out*C];Dtype 原样透传。
    """
    B, N, C = z.shape
    assert N == grid * grid, f"期望 {grid*grid} token,得到 {N}"
    r = grid // out
    z = z.view(B, out, r, out, r, C).mean((2, 4))          # [B, out, out, C]
    return z.reshape(B, out * out * C)
