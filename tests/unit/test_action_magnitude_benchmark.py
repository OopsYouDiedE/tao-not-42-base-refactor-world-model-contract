"""WASD 帧数与鼠标局部幅度测试。"""

from __future__ import annotations

import random

import numpy as np

from bc_datasets.minestudio.action_magnitude_benchmark import (
    build_key_frame_distractors,
    build_mouse_local_distractors,
)


def test_key_distractors_change_total_by_requested_frames() -> None:
    actions = {
        "camera": np.zeros((40, 2), dtype=np.float64),
        "forward": np.array([1] * 20 + [0] * 20, dtype=np.int64),
    }
    distractors, metadata = build_key_frame_distractors(actions, "forward", random.Random(3))
    counts = sorted(int(candidate["forward"].sum()) for candidate in distractors)
    assert len(distractors) == 3
    assert all(count - 20 in {-8, -4, 4, 8} for count in counts)
    assert metadata["true_value"] == 20


def test_key_candidates_are_pairwise_equidistant() -> None:
    actions = {
        "camera": np.zeros((40, 2), dtype=np.float64),
        "forward": np.array([1] * 20 + [0] * 20, dtype=np.int64),
    }
    distractors, _ = build_key_frame_distractors(actions, "forward", random.Random(11))
    candidates = [actions, *distractors]
    distances = []
    for left_index in range(4):
        for right_index in range(left_index + 1, 4):
            distances.append(int(np.count_nonzero(
                candidates[left_index]["forward"] != candidates[right_index]["forward"]
            )))
    assert len(set(distances)) == 1


def test_mouse_distractors_only_scale_selected_positions() -> None:
    actions = {"camera": np.ones((12, 2), dtype=np.float64)}
    distractors, metadata = build_mouse_local_distractors(actions, 1, random.Random(5))
    selected = set(metadata["selected_frame_offsets"])
    for candidate, multipliers in zip(distractors, metadata["multipliers_by_candidate"]):
        multiplier_by_index = dict(zip(metadata["selected_frame_offsets"], multipliers))
        for index in range(12):
            expected = multiplier_by_index.get(index, 1.0)
            assert candidate["camera"][index, 1] == expected
            assert candidate["camera"][index, 0] == 1.0


def test_mouse_candidates_are_pairwise_equidistant() -> None:
    actions = {"camera": np.ones((20, 2), dtype=np.float64)}
    distractors, _ = build_mouse_local_distractors(actions, 0, random.Random(7))
    candidates = [actions, *distractors]
    distances = []
    for left_index in range(4):
        for right_index in range(left_index + 1, 4):
            distances.append(float(np.abs(
                candidates[left_index]["camera"][:, 0]
                - candidates[right_index]["camera"][:, 0]
            ).sum()))
    assert np.allclose(distances, distances[0])
