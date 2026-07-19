"""提供 Pre-LN Transformer 编码器。"""

from __future__ import annotations

import torch
import torch.nn as nn


class PreNormalizationTransformerEncoder(nn.Module):
    """保持 token Shape 的 Pre-LN Transformer 编码器。"""

    def __init__(self, channels: int, heads: int, layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            channels,
            heads,
            dim_feedforward=4 * channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False,
        )
        self.output_normalization = nn.LayerNorm(channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """编码 ``[B,L,C]`` token 并保持 Shape 与 Dtype。"""
        return self.output_normalization(self.body(tokens))
