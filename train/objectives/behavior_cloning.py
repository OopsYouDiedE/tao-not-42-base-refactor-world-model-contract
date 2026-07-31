"""行为克隆监督项的框架无关归约函数。"""

from collections.abc import Sequence

import numpy as np


def masked_mean(losses: Sequence[float], mask: Sequence[bool | float]) -> float:
    """对有效动作位置的逐项损失求均值。"""
    values = np.asarray(losses, dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("losses 和 mask 必须是形状相同的一维序列")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("losses 和 mask 必须只包含有限数值")
    if (weights < 0).any():
        raise ValueError("mask 权重不能为负数")
    weight_sum = float(weights.sum())
    if weight_sum == 0:
        raise ValueError("mask 必须至少选择一个动作位置")
    return float(np.dot(values, weights) / weight_sum)
