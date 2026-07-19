"""提供条件化逐通道仿射调制。"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureWiseLinearModulation(nn.Module):
    """用 ``[B,C]`` 条件调制任意 ``[B,...,C]`` 特征。"""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Linear(channels, channels)
        self.bias = nn.Linear(channels, channels)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """返回与 ``features`` 相同 Shape 和 Dtype 的有界尺度仿射结果。"""
        scale = torch.tanh(self.scale(condition))
        bias = self.bias(condition)
        while scale.ndim < features.ndim:
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return features * (1.0 + scale) + bias
