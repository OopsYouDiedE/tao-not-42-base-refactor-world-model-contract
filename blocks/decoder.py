"""L1 primitive 积木库 - 卷积图像解码器 (blocks/decoder.py)。

对外接口:
    ConvDecoder — DreamerV3/DRAMA 式卷积解码器(Linear 投影到 min_res 小图,逐级 ×stride 上采样)。

与 blocks/encoder.py:ConvEncoder 镜像。复用 blocks/conv.py 的 Conv2dSamePad / ImgChLayerNorm。
上采样用 "resize(Upsample)+same-conv" 而非 ConvTranspose:免去 output_padding 的写死、且无棋盘伪影。
本块产出**原始图像张量**(无激活封顶),重建似然由 blocks.distributions.MSEDist / SymlogDist 在外部封装。
所有尺寸/层数/卷积核/步长/起始分辨率/激活/归一化经构造参数注入,**不写死**;默认值对齐 DRAMA 解码器
(depths=(64,64,48,32,16) 为编码器倒序, kernel=5, SiLU)。

输出分辨率 = min_res × stride^len(depths)(与对称 ConvEncoder 的下采样级数一致);由调参方负责对齐。
"""
import torch
import torch.nn as nn

from blocks.conv import Conv2dSamePad, ImgChLayerNorm


class ConvDecoder(nn.Module):
    """卷积图像解码器(Linear → reshape 小图 → 逐级 resize-conv 上采样)。

    输入接受任意前导维 [..., feat_dim](展平后还原),故潜向量带不带时间维都可。

    Args:
        feat_dim: 输入特征/潜向量维度。
        out_channels: 重建图像通道数。
        depths: 每级通道数序列;depths[0] 为 min_res 处的通道数,后续各级逐步收窄。
                级数 = len(depths),对应同样级数的 ×stride 上采样。
        kernel: same-pad 卷积核大小。
        stride: 每级上采样倍率(Upsample scale_factor)。
        min_res: Linear 投影后起始方形分辨率(min_res × min_res)。
        act: 激活层**类**(可调用,无参构造),默认 nn.SiLU。
        norm: 是否在每级(末级除外)卷积后插入通道维 LayerNorm(I7)。
        upsample_mode: Upsample 插值模式(nearest / bilinear 等)。
    """

    def __init__(self, feat_dim, out_channels=3, depths=(64, 64, 48, 32, 16),
                 kernel=5, stride=2, min_res=4, act=nn.SiLU, norm=True,
                 upsample_mode="nearest"):
        super().__init__()
        self.depths = tuple(depths)
        self.min_res = min_res
        self.base_ch = self.depths[0]
        self.fc = nn.Linear(feat_dim, self.base_ch * min_res * min_res)
        layers = []
        c = self.base_ch
        for d in self.depths[1:]:
            layers.append(nn.Upsample(scale_factor=stride, mode=upsample_mode))
            layers.append(Conv2dSamePad(c, d, kernel, stride=1, bias=not norm))
            if norm:
                layers.append(ImgChLayerNorm(d))
            layers.append(act())
            c = d
        # 末级:再上采样一次后映射到图像通道,不接归一化/激活(输出原始像素值)
        layers.append(nn.Upsample(scale_factor=stride, mode=upsample_mode))
        layers.append(Conv2dSamePad(c, out_channels, kernel, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, feat):
        *lead, f = feat.shape
        x = self.fc(feat.reshape(-1, f))
        x = x.reshape(-1, self.base_ch, self.min_res, self.min_res)
        x = self.layers(x)
        return x.reshape(*lead, *x.shape[1:])
