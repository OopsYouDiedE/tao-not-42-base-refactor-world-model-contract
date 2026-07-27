# -*- coding: utf-8 -*-
"""守卫的逐 tick 求值器与内建像素通道计算（帧级反应性的运行时实现）。

对外接口：
    compute_pixel_channels — 由相邻帧与段起始帧算出 pixel.change / pixel.drift。
    GuardMonitor — 持有 sustain 计数，逐 tick 喂入通道值，返回首个命中的守卫。

设计要点：求值是纯数值比较，单 tick 成本可忽略，因此可以在**不做任何推理**的前提下获得
帧级中断能力。这正是"必须一轮推理才能反应，但绝不能逐帧推理"的解法：语义由大模型隔秒
给出，反应由守卫每 tick 兜住。
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np

from control_contract.decision_segment import (
    Guard,
    GuardComparison,
    PIXEL_CHANGE_CHANNEL,
    PIXEL_DRIFT_CHANNEL,
)
from control_contract.segment_compiler import GuardPlan


def compute_pixel_channels(
    current_frame: np.ndarray,
    previous_frame: Optional[np.ndarray],
    segment_start_frame: Optional[np.ndarray],
) -> Dict[str, float]:
    """计算内建像素通道值。

    Parameters
    ----------
    current_frame : np.ndarray
        当前观测帧，[H, W, 3]，uint8 或 float。
    previous_frame : Optional[np.ndarray]
        上一 tick 的帧，形状同上；None（段首）时 pixel.change 记 0。
    segment_start_frame : Optional[np.ndarray]
        本段第一帧；None 时 pixel.drift 记 0。

    Returns
    -------
    Dict[str, float]
        ``{"pixel.change": 0..1, "pixel.drift": 0..1}``，均为平均绝对差 / 255。
    """
    current = current_frame.astype(np.float32)
    channels = {PIXEL_CHANGE_CHANNEL: 0.0, PIXEL_DRIFT_CHANNEL: 0.0}
    if previous_frame is not None and previous_frame.shape == current_frame.shape:
        channels[PIXEL_CHANGE_CHANNEL] = float(
            np.abs(current - previous_frame.astype(np.float32)).mean() / 255.0)
    if segment_start_frame is not None and segment_start_frame.shape == current_frame.shape:
        channels[PIXEL_DRIFT_CHANNEL] = float(
            np.abs(current - segment_start_frame.astype(np.float32)).mean() / 255.0)
    return channels


class GuardMonitor:
    """一段守卫计划的逐 tick 求值器，持有各守卫的 sustain 连续计数与段起始基线。

    用法：段开始时构造一次，之后每 tick 调用 ``evaluate``；返回非 None 即应立即截断本段、
    抓取观测并请求下一轮推理。
    """

    def __init__(self, guard_plans: List[GuardPlan]):
        """按守卫计划初始化计数器。

        Parameters
        ----------
        guard_plans : List[GuardPlan]
            compile_segment 产出的守卫计划。
        """
        self._plans = list(guard_plans)
        self._streaks = [0] * len(self._plans)
        self._baselines: List[Optional[float]] = [None] * len(self._plans)

    def evaluate(self, channel_values: Mapping[str, float]) -> Optional[Guard]:
        """喂入本 tick 的通道值，返回首个满足 sustain 要求的守卫。

        Parameters
        ----------
        channel_values : Mapping[str, float]
            通道名 → 本 tick 数值。缺失的通道视为该守卫本 tick 不成立（计数清零）。

        Returns
        -------
        Optional[Guard]
            命中的守卫；无命中返回 None。声明顺序即优先级。
        """
        for index, plan in enumerate(self._plans):
            guard = plan.guard
            if guard.channel not in channel_values:
                self._streaks[index] = 0
                continue
            value = float(channel_values[guard.channel])
            if self._baselines[index] is None:
                self._baselines[index] = value
            if guard.comparison is GuardComparison.BELOW:
                satisfied = value < guard.threshold
            elif guard.comparison is GuardComparison.ABOVE:
                satisfied = value > guard.threshold
            else:
                baseline = self._baselines[index]
                satisfied = abs(value - baseline) > guard.threshold
            if satisfied:
                self._streaks[index] += 1
                if self._streaks[index] >= plan.sustain_ticks:
                    return guard
            else:
                self._streaks[index] = 0
        return None
