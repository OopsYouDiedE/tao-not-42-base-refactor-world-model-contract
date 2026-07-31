"""Lumine 动作编解码的不变量测试：run-length 语义、钳位与脏输入容错。"""

from __future__ import annotations

import numpy as np
import pytest

from lumine.action_codec import (
    MOUSE_DELTA_LIMIT,
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


def test_camera_is_encoded_inside_each_tick() -> None:
    """相机增量按 tick 编码为 Mouse dx dy，不再累计到窗口 header。"""
    window = _window(4)
    window["camera"] = np.array(
        [[0.15, 0.30], [0.15, 0.30], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    action = encode_lumine_action(window)
    assert action.chunks[0].mouse == (2, 1)
    assert action.chunks[1].mouse == (2, 1)
    assert action.chunks[2].mouse == (0, 0)
    assert "Mouse 2 1" in action.to_text()


def test_mouse_delta_is_clamped() -> None:
    """极端相机增量被钳到 ±MOUSE_DELTA_LIMIT，不产生越界数值。"""
    window = _window(4)
    window["camera"] = np.full((4, 2), 1000.0, dtype=np.float64)
    action = encode_lumine_action(window)
    assert all(chunk.mouse == (MOUSE_DELTA_LIMIT, MOUSE_DELTA_LIMIT) for chunk in action.chunks)


def test_multi_frame_chunk_keeps_short_press() -> None:
    """一个 chunk 覆盖多帧时，任一帧按下即记为按住，短按不丢。"""
    window = _window(4, attack=[0, 1, 0, 0])
    action = encode_lumine_action(window, frames_per_chunk=2)
    assert len(action.chunks) == 2
    assert action.chunks[0].keys == ("MouseLeft",)
    assert action.chunks[1].keys == ()


def test_roundtrip_is_identity() -> None:
    """编码 → 文本 → 解码得到完全相同的结构。"""
    window = _window(4, forward=[1, 1, 1, 0], jump=[0, 1, 1, 0], attack=[1, 0, 0, 1])
    window["camera"] = np.array(
        [[0.3, -0.6], [0.0, 0.0], [-0.15, 0.15], [0.0, 0.0]],
        dtype=np.float64,
    )
    action = encode_lumine_action(window)
    assert decode_lumine_action(action.to_text(), expected_chunks=4) == action


def test_decode_pads_and_truncates_to_expected_chunks() -> None:
    """chunk 数不足补空、超出截断，输出始终定长。"""
    short = decode_lumine_action("<|action_start|> ; W <|action_end|>", expected_chunks=4)
    assert len(short.chunks) == 4
    assert short.chunks[3].keys == ()
    long_text = "<|action_start|> ; W ; W ; W ; W ; W ; W <|action_end|>"
    assert len(decode_lumine_action(long_text, expected_chunks=4).chunks) == 4


def test_decode_drops_unknown_keys_and_bad_numbers() -> None:
    """未知键名与非法数值被丢弃，解码结果仍结构合法。"""
    text = "<|action_start|> ; Mouse abc xyz ; W bogus_key ; nonsense Scroll 99<|action_end|>"
    action = decode_lumine_action(text, expected_chunks=2)
    assert action.chunks[0].mouse == (0, 0)
    assert action.chunks[1].keys == ("W",)


def test_named_mouse_can_mix_with_keys_in_one_tick() -> None:
    """需要连续移动时允许 Mouse 与按键混写，解析后仍属于同一 tick。"""
    text = "<|action_start|> ; Mouse 35 30 W D <|action_end|>"
    action = decode_lumine_action(text)
    assert action.chunks[0].mouse == (35, 30)
    assert action.chunks[0].keys == ("W", "D")
    assert decode_lumine_action(action.to_text()) == action


def test_chunk_count_is_variable_without_expected_chunks() -> None:
    short = decode_lumine_action("<|action_start|> ; W <|action_end|>")
    long = decode_lumine_action("<|action_start|> ; W ; W ; Mouse 4 -2 <|action_end|>")
    assert len(short.chunks) == 1
    assert len(long.chunks) == 3


def test_decode_rejects_text_without_markers() -> None:
    """没有成对标记时明确报错，不猜测内容。"""
    with pytest.raises(ValueError):
        decode_lumine_action("; W")


def test_encode_rejects_indivisible_window() -> None:
    """窗口帧数不能被 chunk 帧数整除时报错，不静默丢帧。"""
    with pytest.raises(ValueError):
        encode_lumine_action(_window(5, forward=[1, 1, 1, 1, 1]), frames_per_chunk=2)
