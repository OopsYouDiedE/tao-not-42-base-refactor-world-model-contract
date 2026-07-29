"""流式加载的样本切分与 worker 推算测试：不碰真实 LMDB。"""

from __future__ import annotations

import pytest

from train.lumine_streaming_dataset import (
    StreamingSettings,
    _observation_frame_indices,
    _sample_positions,
    resolve_worker_count,
)

_GIBIBYTE = 1024 ** 3


def test_default_settings_match_lumine_perception_rate() -> None:
    """默认布局对应 Lumine 的 5Hz 感知率：20Hz 下 4 帧 = 200ms。"""
    settings = StreamingSettings()
    assert settings.window_frames == 4
    assert settings.stride_frames == 4


def test_settings_reject_indivisible_chunk_size() -> None:
    """chunk 帧数不整除窗口帧数时报错，与落盘路径同一约束。"""
    with pytest.raises(ValueError):
        StreamingSettings(window_frames=4, frames_per_chunk=3)


def test_sample_positions_do_not_run_past_episode_end() -> None:
    """最后一个窗口必须完整落在 episode 内，不产生越界样本。"""
    positions = _sample_positions({"ep": 10}, ["ep"], StreamingSettings())
    assert positions == [("ep", 0), ("ep", 4)]  # start=8 时 8+4>10，丢弃


def test_sample_positions_reserve_history_span() -> None:
    """有历史帧时起始帧后移，保证回溯不越界。"""
    settings = StreamingSettings(history_windows=2)
    positions = _sample_positions({"ep": 20}, ["ep"], settings)
    assert positions[0] == ("ep", 8)  # 2 个窗口 × 4 帧
    assert all(start >= 8 for _, start in positions)


def test_sample_positions_skip_too_short_episodes() -> None:
    """帧数不足一个窗口的 episode 不产出样本，也不报错。"""
    assert _sample_positions({"ep": 3}, ["ep"], StreamingSettings()) == []


def test_sample_positions_only_cover_requested_episodes() -> None:
    """只切给定 episode，防止验证集 episode 漏进训练集。"""
    frames = {"train_ep": 20, "validation_ep": 20}
    positions = _sample_positions(frames, ["train_ep"], StreamingSettings())
    assert {episode for episode, _ in positions} == {"train_ep"}


def test_overlapping_stride_produces_more_samples() -> None:
    """stride 小于窗口时窗口重叠，样本数增加。"""
    dense = _sample_positions({"ep": 20}, ["ep"], StreamingSettings(stride_frames=1))
    sparse = _sample_positions({"ep": 20}, ["ep"], StreamingSettings())
    assert len(dense) > len(sparse)


def test_observation_indices_match_offline_path() -> None:
    """观测帧下标与落盘路径一致：时间升序，当前帧在末位。"""
    settings = StreamingSettings(history_windows=3)
    assert _observation_frame_indices(20, settings) == [8, 12, 16, 20]


def test_observation_indices_drop_negative_history() -> None:
    """episode 开头历史不足时丢掉越界帧，不产生负下标。"""
    settings = StreamingSettings(history_windows=3)
    assert _observation_frame_indices(4, settings) == [0, 4]


def test_worker_count_leaves_a_core_for_main_process() -> None:
    """内存充裕时按核心数决定，留一个核心给主进程做 collate。"""
    assert resolve_worker_count(
        logical_cores=8, available_memory_bytes=64 * _GIBIBYTE,
    ) == 7


def test_worker_count_is_capped_by_memory() -> None:
    """内存紧张时以内存为准，不按核心数硬开。"""
    assert resolve_worker_count(
        logical_cores=32, available_memory_bytes=4 * _GIBIBYTE,
    ) == 4


def test_worker_count_never_drops_below_one() -> None:
    """单核或内存极少时仍返回 1，不返回 0 或负数。"""
    assert resolve_worker_count(logical_cores=1, available_memory_bytes=_GIBIBYTE) == 1
    assert resolve_worker_count(logical_cores=2, available_memory_bytes=1024) == 1


def test_worker_count_respects_hard_ceiling() -> None:
    """核心与内存都很富余时仍受上限约束。"""
    assert resolve_worker_count(
        logical_cores=256, available_memory_bytes=1024 * _GIBIBYTE,
    ) == 16


def test_worker_count_falls_back_to_cores_when_memory_unknown() -> None:
    """内存读不到时只按核心数决定，不猜测内存大小。"""
    assert resolve_worker_count(logical_cores=4) == 3
