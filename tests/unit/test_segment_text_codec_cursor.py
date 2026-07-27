# -*- coding: utf-8 -*-
"""GUI 光标编译的回归测试。

守的是一组**实测标定**（见 segment_text_codec 顶部常量注释）：光标只由相机增量驱动、
只走整数像素（1 px = 0.15°）、按绝对位置语义"到位后停住"、按 E 后复位并延迟 2 tick。
v3 那次合成失败的根因就在这里——提示词承诺绝对位置，编译器却按相对增量走，且 y 轴
和 Mouse 的 pitch 一起被取反。
"""
from __future__ import annotations

import pytest

from rl_training_environments.craftground.segment_text_codec import (
    CURSOR_DEGREES_PER_PIXEL,
    CURSOR_HOME,
    CURSOR_SCREEN_HEIGHT_PIXELS,
    CURSOR_SCREEN_WIDTH_PIXELS,
    compile_parsed_segment,
    parse_segment_text,
)

AIM_CAP_DEGREES_PER_TICK = 18.0


def compile_text(text: str, cursor_start=CURSOR_HOME):
    return compile_parsed_segment(parse_segment_text(text), cursor_start=cursor_start)


def camera_totals(compiled):
    """整段累计的相机增量。光标位移最终就落在这两个通道上。"""
    return (sum(action["camera_yaw"] for action in compiled.actions),
            sum(action["camera_pitch"] for action in compiled.actions))


def pixels_to_degrees(delta_screen: float, screen_pixels: int) -> float:
    """按整数像素折算期望度数——光标不走小数像素，期望值也不能按小数算。"""
    return round(delta_screen * screen_pixels) * CURSOR_DEGREES_PER_PIXEL


def test_absolute_target_is_reached_within_deadline():
    """两个 Cursor 项都按绝对位置到位，段末停在最后一个目标上。"""
    compiled = compile_text(
        "for: 190/20s\ntap: 40/20s Mouse_L, 100/20s Mouse_L\n"
        "Cursor: 30/20s 0.383,0.706, 70/20s 0.496,0.586\nwhy: 取木板再合成\n")
    assert compiled.cursor_end == pytest.approx((0.496, 0.586), abs=2e-3)
    assert compiled.cursor_truncated_x == pytest.approx(0.0, abs=1e-3)
    assert compiled.cursor_truncated_y == pytest.approx(0.0, abs=1e-3)


def test_repeating_same_target_does_not_drift():
    """同一位置写三次，第一段到位后不应再有任何位移。

    这是绝对语义与相对语义的判别式：按相对增量走会累积成 3 倍位移。
    """
    compiled = compile_text(
        "for: 90/20s\nCursor: 30/20s 0.4,0.7, 60/20s 0.4,0.7, 90/20s 0.4,0.7\nwhy: 停住\n")
    assert compiled.cursor_end == pytest.approx((0.4, 0.7), abs=1e-6)
    expected_yaw = pixels_to_degrees(0.4 - 0.5, CURSOR_SCREEN_WIDTH_PIXELS)
    expected_pitch = pixels_to_degrees(0.7 - 0.5, CURSOR_SCREEN_HEIGHT_PIXELS)
    yaw, pitch = camera_totals(compiled)
    assert yaw == pytest.approx(expected_yaw, abs=0.01)
    assert pitch == pytest.approx(expected_pitch, abs=0.01)


def test_cursor_down_maps_to_positive_camera_pitch():
    """光标向下 = camera_pitch 正方向。这是实测出来的，与 Mouse 的 pitch 口径相反。"""
    compiled = compile_text("for: 20/20s\nCursor: 20/20s 0.5,0.9\nwhy: 下移\n")
    _, pitch = camera_totals(compiled)
    assert pitch > 0
    assert pitch == pytest.approx(
        pixels_to_degrees(0.9 - 0.5, CURSOR_SCREEN_HEIGHT_PIXELS), abs=0.01)


def test_mouse_and_cursor_pitch_are_not_negated_together():
    """Mouse 的 +pitch=抬头要取负，Cursor 的 +y=向下不取负；两者不能先相加再取反。"""
    compiled = compile_text(
        "for: 20/20s\nMouse: 20/20s +0,+18\nCursor: 20/20s 0.5,0.9\nwhy: 同时动\n")
    _, pitch = camera_totals(compiled)
    expected = -AIM_CAP_DEGREES_PER_TICK + pixels_to_degrees(
        0.9 - 0.5, CURSOR_SCREEN_HEIGHT_PIXELS)
    assert pitch == pytest.approx(expected, abs=0.01)


def test_speed_cap_truncates_and_is_accounted():
    """1 tick 跨整屏做不到，必须截断并把差额记进台账，不许假装成功。"""
    compiled = compile_text("for: 1/20s\nCursor: 1/20s 1.0,1.0\nwhy: 冲太快\n")
    assert compiled.cursor_truncated_x > 0.3
    assert compiled.cursor_truncated_y > 0.1
    max_pixels = int(AIM_CAP_DEGREES_PER_TICK / CURSOR_DEGREES_PER_PIXEL)
    assert max_pixels == 120


def test_cursor_state_continues_across_segments():
    """背包不关，光标停在上段末尾；本段目标等于起点时应零位移。"""
    compiled = compile_text(
        "for: 20/20s\nCursor: 20/20s 0.4,0.7\nwhy: 原地\n", cursor_start=(0.4, 0.7))
    yaw, pitch = camera_totals(compiled)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)


def test_pressing_inventory_resets_cursor_to_center():
    """按 E 重开背包，光标复位到屏幕正中，之后的规划从正中起算。"""
    compiled = compile_text(
        "for: 40/20s\ntap: 2/20s E\nCursor: 40/20s 0.5,0.5\nwhy: 开背包\n",
        cursor_start=(0.9, 0.1))
    yaw, pitch = camera_totals(compiled)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)


def test_out_of_range_cursor_warns():
    """写成像素值（320,180）而不是归一化，要告警而不是静默钳位。"""
    parsed = parse_segment_text("for: 20/20s\nCursor: 20/20s 320,180\nwhy: 写错单位\n")
    assert any("超出 0..1" in warning for warning in parsed.warnings)
