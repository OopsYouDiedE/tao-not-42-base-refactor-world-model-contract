"""动作四选一测试集的候选唯一性与序列化测试。"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from bc_datasets.minestudio.action_choice_benchmark import (
    action_signature,
    build_magnitude_distractors,
    classify_action_type,
    classify_detailed_action_type,
    counterfactual_candidates,
    is_informative_action,
    serialize_action,
    shuffled_choices,
)


def _actions(yaw: float, forward: int = 0) -> dict[str, np.ndarray]:
    return {
        "camera": np.asarray([[0.0, yaw], [0.0, 0.0]], dtype=np.float64),
        "forward": np.asarray([forward, forward], dtype=np.int64),
    }


def test_signature_distinguishes_camera_and_keys() -> None:
    assert action_signature(_actions(1.0)) != action_signature(_actions(2.0))
    assert action_signature(_actions(1.0)) != action_signature(_actions(1.0, forward=1))


def test_informative_action_rejects_exact_noop() -> None:
    assert not is_informative_action(_actions(0.0))
    assert is_informative_action(_actions(1.0))
    assert is_informative_action(_actions(0.0, forward=1))


def test_action_types_are_mutually_exclusive() -> None:
    assert classify_action_type(_actions(1.0)) == "camera_only"
    assert classify_action_type(_actions(1.0, forward=1)) == "movement"
    interaction = _actions(1.0)
    interaction["attack"] = np.ones(2, dtype=np.int64)
    assert classify_action_type(interaction) == "interaction"
    interaction["forward"] = np.ones(2, dtype=np.int64)
    assert classify_action_type(interaction) == "mixed"
    assert classify_action_type(_actions(0.0)) == "noop"


def test_detailed_action_types_cover_control_families() -> None:
    camera = _actions(1.0)
    assert classify_detailed_action_type(camera) == "camera_only"
    movement = _actions(1.0, forward=1)
    assert classify_detailed_action_type(movement) == "locomotion"
    movement["jump"] = np.ones(2, dtype=np.int64)
    assert classify_detailed_action_type(movement) == "jump"
    attack = _actions(1.0)
    attack["attack"] = np.ones(2, dtype=np.int64)
    assert classify_detailed_action_type(attack) == "attack"
    attack["forward"] = np.ones(2, dtype=np.int64)
    assert classify_detailed_action_type(attack) == "move_attack"


def test_magnitude_camera_levels_are_five_degrees_apart() -> None:
    actions = {"camera": np.tile(np.array([[0.0, 0.5]]), (20, 1))}
    distractors, metadata = build_magnitude_distractors(
        actions, "camera_only", correct_rank=2, randomizer=random.Random(1)
    )
    assert metadata["levels"] == [0.0, 5.0, 10.0, 15.0]
    assert metadata["correct_rank"] == 2
    assert len(distractors) == 3


def test_magnitude_key_levels_change_duration_by_quarter_window() -> None:
    actions = {
        "camera": np.zeros((20, 2), dtype=np.float64),
        "forward": np.array([1] * 5 + [0] * 15, dtype=np.int64),
    }
    distractors, metadata = build_magnitude_distractors(
        actions, "locomotion", correct_rank=1, randomizer=random.Random(1)
    )
    assert metadata["levels"] == [0, 5, 10, 15]
    assert sorted(int(candidate["forward"].sum()) for candidate in distractors) == [0, 10, 15]


def test_serialized_action_keeps_every_frame() -> None:
    serialized = serialize_action(_actions(1.5, forward=1))
    assert len(serialized["frames"]) == 2
    assert serialized["frames"][0]["camera_yaw_degrees"] == 1.5
    assert serialized["frames"][0]["held_keys"] == ["W"]


def test_choices_have_one_answer_and_are_reproducible() -> None:
    candidates = [_actions(value) for value in (1.0, 2.0, 3.0, 4.0)]
    first = shuffled_choices(candidates[0], candidates[1:], random.Random(7))
    second = shuffled_choices(candidates[0], candidates[1:], random.Random(7))
    assert first == second
    choices, answer = first
    assert set(choices) == {"A", "B", "C", "D"}
    assert answer in choices


def test_choices_can_fix_answer_position_for_balanced_benchmark() -> None:
    candidates = [_actions(value) for value in (1.0, 2.0, 3.0, 4.0)]
    choices, answer = shuffled_choices(
        candidates[0], candidates[1:], random.Random(7), correct_label="B"
    )
    assert set(choices) == {"A", "B", "C", "D"}
    assert answer == "B"


def test_choices_reject_duplicate_actions() -> None:
    with pytest.raises(ValueError, match="互不相同"):
        shuffled_choices(_actions(1.0), [_actions(1.0), _actions(2.0), _actions(3.0)], random.Random(1))


def test_counterfactual_candidates_include_direction_scale_and_time() -> None:
    actions = _actions(1.5, forward=1)
    candidates = dict(counterfactual_candidates(actions))
    assert candidates["reverse_yaw"]["camera"][0, 1] == -1.5
    assert candidates["scale_camera_0.5"]["camera"][0, 1] == 0.75
    assert np.array_equal(candidates["swap_forward_back"]["back"], np.ones(2))
    assert np.array_equal(candidates["reverse_timeline"]["camera"], actions["camera"][::-1])


def test_counterfactual_candidates_handle_key_only_action() -> None:
    actions = _actions(0.0, forward=1)
    correct = json.dumps(serialize_action(actions), sort_keys=True)
    visible = {
        json.dumps(serialize_action(candidate), sort_keys=True)
        for _, candidate in counterfactual_candidates(actions, random.Random(3))
        if json.dumps(serialize_action(candidate), sort_keys=True) != correct
    }
    assert len(visible) >= 3
