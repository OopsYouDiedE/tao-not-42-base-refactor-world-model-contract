# -*- coding: utf-8 -*-
"""守卫求值器与内建像素通道单测：帧级反应性不依赖任何游戏集成。"""
import numpy as np
import pytest

from control_contract.decision_segment import (
    Guard,
    GuardComparison,
    PIXEL_CHANGE_CHANNEL,
    PIXEL_DRIFT_CHANNEL,
)
from control_contract.guard_monitor import GuardMonitor, compute_pixel_channels
from control_contract.segment_compiler import GuardPlan


def _frame(value: int, shape=(4, 4, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_pixel_channels_zero_without_history():
    """段首无历史帧时两个通道均为 0。"""
    channels = compute_pixel_channels(_frame(100), None, None)
    assert channels[PIXEL_CHANGE_CHANNEL] == 0.0
    assert channels[PIXEL_DRIFT_CHANNEL] == 0.0


def test_pixel_change_measures_adjacent_difference():
    """pixel.change 为相邻帧平均绝对差 / 255。"""
    channels = compute_pixel_channels(_frame(155), _frame(100), _frame(100))
    assert channels[PIXEL_CHANGE_CHANNEL] == pytest.approx(55 / 255)


def test_pixel_drift_measures_distance_from_segment_start():
    """pixel.drift 相对段起始帧累积，可与 change 不同。"""
    channels = compute_pixel_channels(_frame(200), _frame(190), _frame(100))
    assert channels[PIXEL_CHANGE_CHANNEL] == pytest.approx(10 / 255)
    assert channels[PIXEL_DRIFT_CHANNEL] == pytest.approx(100 / 255)


def test_shape_mismatch_is_ignored_not_raised():
    """分辨率变化时不抛错，退化为 0（运行时不因抓帧尺寸抖动而崩）。"""
    channels = compute_pixel_channels(_frame(120), _frame(100, (8, 8, 3)), None)
    assert channels[PIXEL_CHANGE_CHANNEL] == 0.0


def test_instant_guard_trips_on_first_satisfying_tick():
    """sustain=0（换算为 1 tick）的守卫瞬时命中。"""
    guard = Guard(channel=PIXEL_CHANGE_CHANNEL, comparison=GuardComparison.ABOVE,
                  threshold=0.2, label="something happened")
    monitor = GuardMonitor([GuardPlan(guard=guard, sustain_ticks=1)])
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.1}) is None
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.5}) is guard


def test_sustain_requires_consecutive_ticks():
    """sustain 要求连续成立，中断即清零，抑制抖动误触。"""
    guard = Guard(channel=PIXEL_CHANGE_CHANNEL, comparison=GuardComparison.BELOW,
                  threshold=0.01, sustain_ms=150, label="stuck")
    monitor = GuardMonitor([GuardPlan(guard=guard, sustain_ticks=3)])
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.001}) is None
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.001}) is None
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.9}) is None      # 打断，计数清零
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.001}) is None
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.001}) is None
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.001}) is guard


def test_delta_above_uses_segment_baseline():
    """delta_above 以本段首次求值的通道值为基线。"""
    guard = Guard(channel="health", comparison=GuardComparison.DELTA_ABOVE, threshold=0.2)
    monitor = GuardMonitor([GuardPlan(guard=guard, sustain_ticks=1)])
    assert monitor.evaluate({"health": 1.0}) is None
    assert monitor.evaluate({"health": 0.9}) is None
    assert monitor.evaluate({"health": 0.7}) is guard


def test_missing_channel_never_trips():
    """通道缺失时守卫不命中且计数清零，不误触发。"""
    guard = Guard(channel="health", comparison=GuardComparison.BELOW, threshold=0.5)
    monitor = GuardMonitor([GuardPlan(guard=guard, sustain_ticks=1)])
    assert monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.0}) is None


def test_declaration_order_is_priority():
    """多个守卫同时命中时按声明顺序返回第一个。"""
    first = Guard(channel=PIXEL_CHANGE_CHANNEL, comparison=GuardComparison.ABOVE,
                  threshold=0.1, label="first")
    second = Guard(channel=PIXEL_DRIFT_CHANNEL, comparison=GuardComparison.ABOVE,
                   threshold=0.1, label="second")
    monitor = GuardMonitor([
        GuardPlan(guard=first, sustain_ticks=1), GuardPlan(guard=second, sustain_ticks=1)])
    tripped = monitor.evaluate({PIXEL_CHANGE_CHANNEL: 0.9, PIXEL_DRIFT_CHANNEL: 0.9})
    assert tripped is first


def test_no_guards_never_trips():
    """无守卫的段永不被中断。"""
    assert GuardMonitor([]).evaluate({PIXEL_CHANGE_CHANNEL: 1.0}) is None
