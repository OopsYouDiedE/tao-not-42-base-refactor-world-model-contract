"""组内相对优势计算。"""

from collections import defaultdict
from collections.abc import Hashable, Sequence

import numpy as np


def grouped_relative_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """按组中心化奖励，并可用组内总体标准差归一化。"""
    if len(rewards) != len(group_ids):
        raise ValueError("rewards 和 group_ids 长度必须相同")
    if epsilon <= 0:
        raise ValueError("epsilon 必须大于零")

    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rewards 必须是一维有限数值")

    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("每个相对优势组至少需要两个样本")

    advantages = np.empty_like(values)
    for indices in groups.values():
        group_rewards = values[indices]
        centered = group_rewards - group_rewards.mean()
        if normalize:
            standard_deviation = group_rewards.std()
            centered = (
                np.zeros_like(centered)
                if standard_deviation < epsilon
                else centered / (standard_deviation + epsilon)
            )
        advantages[indices] = centered
    return advantages
