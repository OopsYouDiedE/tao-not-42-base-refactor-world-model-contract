"""变长动作段和动作—图片对齐校验。"""

from __future__ import annotations

import numpy as np
import pytest

from tao.protocols.action import (
    ActionSegment,
    compress_action_ticks,
    expand_action_segments,
    validate_action_image_alignment,
)


def _actions(length: int = 5) -> dict[str, np.ndarray]:
    return {
        "camera": np.asarray(
            [[0, 0], [0, 0], [1, 2], [0, 0], [-2, 1]][:length],
            dtype=np.int64,
        ),
        "forward": np.ones(length, dtype=np.int64),
    }


def test_compression_preserves_irregular_mouse_positions() -> None:
    segments = compress_action_ticks(
        [
            {"keys": ("W",), "mouse": (0, 0)},
            {"keys": ("W",), "mouse": (0, 0)},
            {"keys": ("W",), "mouse": (2, 1)},
            {"keys": ("W",), "mouse": (0, 0)},
        ],
    )
    assert segments == [
        ActionSegment(2, ("W",), (0, 0)),
        ActionSegment(1, ("W",), (2, 1)),
        ActionSegment(1, ("W",), (0, 0)),
    ]
    assert expand_action_segments(segments) == [
        {"keys": ("W",), "mouse": (0, 0)},
        {"keys": ("W",), "mouse": (0, 0)},
        {"keys": ("W",), "mouse": (2, 1)},
        {"keys": ("W",), "mouse": (0, 0)},
    ]


def test_alignment_accepts_same_episode_frame_count() -> None:
    report = validate_action_image_alignment(_actions(), np.zeros((5, 8, 8, 3), dtype=np.uint8))
    assert report["frames"] == 5
    assert report["duration_ms"] == 250
    assert report["segments"] == 4


def test_alignment_rejects_mismatched_image_count() -> None:
    with pytest.raises(ValueError, match="不一致"):
        validate_action_image_alignment(_actions(), np.zeros((4, 8, 8, 3), dtype=np.uint8))


def test_alignment_rejects_expected_window_mismatch() -> None:
    with pytest.raises(ValueError, match="期望值"):
        validate_action_image_alignment(
            _actions(),
            np.zeros((5, 8, 8, 3), dtype=np.uint8),
            expected_frames=4,
        )
