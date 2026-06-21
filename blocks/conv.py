"""L1 primitive 积木库 - 卷积底层算子 (blocks/conv.py)。

对外接口:
    Conv2dSamePad  — 带 TensorFlow 式 "same" 动态填充的 Conv2d 子类。
    ImgChLayerNorm — 在通道维(NCHW→NHWC→NCHW)上做 LayerNorm 的图像归一化。

来源:原样照抄 NM512/dreamerv3-torch 的 `networks.py`(MIT,见 blocks/NOTICE.dreamerv3)。
与任务无关的可复用卷积算子,被 DreamerV3 的 ConvEncoder/ConvDecoder 复用;
按本仓 blocks/net 分层约定,从原 networks.py 物理拆到 blocks/。类体保持 1:1 原样。
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


# 以下两类原样照抄 NM512/dreamerv3-torch 的 networks.py(类体逐字 1:1,见 blocks/NOTICE.dreamerv3)。
class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
