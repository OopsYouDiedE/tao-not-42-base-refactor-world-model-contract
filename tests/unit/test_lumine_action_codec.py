"""Lumine 动作编解码的不变量测试：run-length 语义、钳位与脏输入容错。"""

from __future__ import annotations

import numpy as np
import pytest

from bc_datasets.minestudio.lumine_action_codec import (
    MOUSE_DELTA_LIMIT,
    SCROLL_LIMIT,
    decode_lumine_action,
    encode_lumine_action,
    press_release_events,
)


def _window(num_frames: int = 4, **keys: list[int]) -> dict[str, np.ndarray]:
    """构造一个动作窗口，未给出的键全零，camera 默认不动。"""
    window: dict[str, np.ndarray] = {
        "camera": np.zeros((num_frames, 2), dtype=np.float64),
    }
    for name, values in keys.items():
        field = name.replace("hotbar_", "hotbar.")
        window[field] = np.asarray(values, dtype=np.int64)
    return window


def test_held_key_is_not_repressed_across_chunks() -> None:
    """键在相邻 chunk 连续出现时只按下一次，中途不重按。"""
    window = _window(4, forward=[1, 1, 1, 1])
    action = encode_lumine_action(window)
    events = press_release_events(action.chunks)
    assert events[0] == (frozenset({"W"}), frozenset())
    for pressed, released in events[1:]:
        assert not pressed
        assert not released


def test_absent_key_is_released() -> None:
    """键在某 chunk 缺席即视为在该 chunk 松开。"""
    window = _window(4, jump=[1, 1, 0, 0])
    events = press_release_events(encode_lumine_action(window).chunks)
    assert events[2] == (frozenset(), frozenset({"space"}))


def test_camera_accumulates_over_window_and_maps_to_pixels() -> None:
    """窗口内相机增量累加后按 0.15 度/像素换算，列序 [pitch, yaw] → (Y, X)。"""
    window = _window(4)
    window["camera"] = np.array(
        [[0.15, 0.30], [0.15, 0.30], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64,
    )
    action = encode_lumine_action(window)
    assert action.mouse_delta_y == 2  # pitch 0.30 度 / 0.15 = 2 像素
    assert action.mouse_delta_x == 4  # yaw 0.60 度 / 0.15 = 4 像素


def test_mouse_delta_is_clamped() -> None:
    """极端相机增量被钳到 ±MOUSE_DELTA_LIMIT，不产生越界数值。"""
    window = _window(4)
    window["camera"] = np.full((4, 2), 1000.0, dtype=np.float64)
    action = encode_lumine_action(window)
    assert action.mouse_delta_x == MOUSE_DELTA_LIMIT
    assert action.mouse_delta_y == MOUSE_DELTA_LIMIT


def test_multi_frame_chunk_keeps_short_press() -> None:
    """一个 chunk 覆盖多帧时，任一帧按下即记为按住，短按不丢。"""
    window = _window(4, attack=[0, 1, 0, 0])
    action = encode_lumine_action(window, frames_per_chunk=2)
    assert len(action.chunks) == 2
    assert action.chunks[0].keys == ("mouse_left",)
    assert action.chunks[1].keys == ()


def test_roundtrip_is_identity() -> None:
    """编码 → 文本 → 解码得到完全相同的结构。"""
    window = _window(4, forward=[1, 1, 1, 0], jump=[0, 1, 1, 0], attack=[1, 0, 0, 1])
    window["camera"] = np.array(
        [[0.3, -0.6], [0.0, 0.0], [-0.15, 0.15], [0.0, 0.0]], dtype=np.float64,
    )
    action = encode_lumine_action(window)
    assert decode_lumine_action(action.to_text(), expected_chunks=4) == action


def test_decode_pads_and_truncates_to_expected_chunks() -> None:
    """chunk 数不足补空、超出截断，输出始终定长。"""
    short = decode_lumine_action("<|action_start|>0 0 0 ; W<|action_end|>", expected_chunks=4)
    assert len(short.chunks) == 4
    assert short.chunks[3].keys == ()
    long_text = "<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>"
    assert len(decode_lumine_action(long_text, expected_chunks=4).chunks) == 4


def test_decode_drops_unknown_keys_and_bad_numbers() -> None:
    """未知键名与非法数值被丢弃，解码结果仍结构合法。"""
    text = "<|action_start|>abc xyz 99 ; W bogus_key ; nonsense<|action_end|>"
    action = decode_lumine_action(text, expected_chunks=2)
    assert action.mouse_delta_x == 0
    assert action.mouse_delta_y == 0
    assert action.scroll_delta == SCROLL_LIMIT  # 99 钳到 5
    assert action.chunks[0].keys == ("W",)
    assert action.chunks[1].keys == ()


def test_decode_rejects_text_without_markers() -> None:
    """没有成对标记时明确报错，不猜测内容。"""
    with pytest.raises(ValueError):
        decode_lumine_action("0 0 0 ; W")


def test_encode_rejects_indivisible_window() -> None:
    """窗口帧数不能被 chunk 帧数整除时报错，不静默丢帧。"""
    with pytest.raises(ValueError):
        encode_lumine_action(_window(5, forward=[1, 1, 1, 1, 1]), frames_per_chunk=2)
