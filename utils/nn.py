"""通用神经网络辅助函数。"""
import torch.nn as nn

__all__ = ["gn"]


def gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with the largest valid group divisor for the given channel count."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)
