"""提供 Pre-LN 残差前馈算子。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualFeedForwardBlock(nn.Module):
    """以四倍隐藏层变换最后一维并保持输入 Shape。"""

    def __init__(self, channels: int):
        super().__init__()
        self.normalization = nn.LayerNorm(channels)
        self.layers = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """变换 ``[...,C]`` 特征并保持 Shape 与 Dtype。"""
        return value + self.layers(self.normalization(value))
