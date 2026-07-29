"""预训练窗口时间布局的约束测试。"""

from __future__ import annotations

import pytest

from bc_datasets.minestudio.lumine_pretraining_dataset import (
    FRAMES_PER_SECOND,
    WindowLayout,
    _observation_frame_indices,
)


def test_default_layout_matches_lumine_perception_rate() -> None:
    """默认布局对应 Lumine 的 5Hz 感知率：20Hz 下 4 帧 = 200ms。"""
    layout = WindowLayout()
    assert layout.window_frames * 1000 // FRAMES_PER_SECOND == 200
    assert layout.chunks_per_window == 4


def test_coarser_motor_step_reduces_chunk_count() -> None:
    """一个 chunk 覆盖 2 帧（100ms）时窗口只出 2 个 chunk。"""
    layout = WindowLayout(window_frames=4, frames_per_chunk=2)
    assert layout.chunks_per_window == 2


def test_layout_rejects_indivisible_chunk_size() -> None:
    """chunk 帧数不整除窗口帧数时直接报错。"""
    with pytest.raises(ValueError):
        WindowLayout(window_frames=4, frames_per_chunk=3)


def test_observation_indices_are_ascending_with_current_frame_last() -> None:
    """历史帧按感知步回溯，时间升序，当前帧在末位。"""
    layout = WindowLayout(window_frames=4, history_windows=3)
    assert _observation_frame_indices(20, layout) == [8, 12, 16, 20]


def test_observation_indices_drop_negative_history() -> None:
    """episode 开头历史不足时丢掉越界帧，不产生负下标。"""
    layout = WindowLayout(window_frames=4, history_windows=3)
    assert _observation_frame_indices(4, layout) == [0, 4]


def test_non_history_layout_yields_only_current_frame() -> None:
    """history_windows=0 时只给当前帧（Lumine 的 non-history 配方）。"""
    assert _observation_frame_indices(40, WindowLayout()) == [40]
