"""提供带残差的 Pre-Norm 交叉注意力。"""

from __future__ import annotations

import torch
import torch.nn as nn


class PreNormalizationCrossAttention(nn.Module):
    """以独立归一化处理查询和上下文，再返回残差查询。"""

    def __init__(self, channels: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_normalization = nn.LayerNorm(channels)
        self.context_normalization = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """融合 ``[B,Q,C]`` 查询与 ``[B,K,C]`` 上下文，返回 ``[B,Q,C]``。"""
        normalized_context = self.context_normalization(context)
        attended, _ = self.attention(
            self.query_normalization(query),
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        return query + attended
