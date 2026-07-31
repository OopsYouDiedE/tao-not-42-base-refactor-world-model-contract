"""最小轨迹数据工具链的生成规则与审核门槛。"""

from __future__ import annotations

import numpy as np

from dataset.trajectory.generate_questions import (
    AI_REVIEW_PROMPT,
    HUMAN_REVIEW_PROMPT,
    OUTPUT_CONTRACT,
    TASK_PROMPTS,
    TASK_TYPES,
    action_node_frames,
    automatic_quality_reasons,
    build_question_record,
    infer_action_intent,
    normalize_gui_clicks,
    source_frames,
    target_interval_for,
)

ACTION = "<|action_start|> ; Mouse 35 30 ; W <|action_end|>"


def test_three_task_types_and_prompts_are_stable() -> None:
    assert TASK_TYPES == (
        "demonstration_optimization",
        "image_sequence_to_action",
        "history_to_future_action",
        "single_frame_intent_to_action",
    )
    assert "cursor in GUI" in OUTPUT_CONTRACT["mouse"]
    assert "rising-edge click pulses" in AI_REVIEW_PROMPT
    assert "reviewed_optimized_demonstration" in HUMAN_REVIEW_PROMPT
    assert (
        "one valid action block for each adjacent image pair"
        in TASK_PROMPTS["image_sequence_to_action"]
    )


def test_task_frame_boundaries() -> None:
    assert source_frames("history_to_future_action", 20) == [8, 12, 16, 20]
    assert source_frames("image_sequence_to_action", 20) == [20, 21, 22, 23, 24]
    assert source_frames("demonstration_optimization", 20) == [20, 24, 28, 32]
    assert source_frames("single_frame_intent_to_action", 20) == [20]
    question = build_question_record(
        "q",
        "demonstration_optimization",
        "episode",
        20,
        ["a", "b", "c", "d"],
        [ACTION] * 4,
    )
    assert question["target_interval"] == [20, 32]
    assert target_interval_for("image_sequence_to_action", 20) == [20, 24]
    assert target_interval_for("history_to_future_action", 20) == [20, 24]
    assert target_interval_for("single_frame_intent_to_action", 20) == [20, 24]


def test_reference_action_source_covers_target_interval() -> None:
    question = build_question_record(
        "q",
        "history_to_future_action",
        "episode",
        20,
        ["a", "b", "c", "d"],
    )
    assert question["target_interval"] == [20, 24]


def test_action_segments_follow_each_image_gap() -> None:
    assert action_node_frames(
        "image_sequence_to_action",
        [10, 15, 18],
        [10, 18],
    ) == [10, 15, 18]
    assert action_node_frames(
        "history_to_future_action",
        [2, 7, 12],
        [12, 19],
    ) == [12, 19]


def test_gui_held_click_becomes_pulses_but_gameplay_is_held() -> None:
    actions = {"camera": np.zeros((5, 2)), "attack": np.asarray([1, 1, 1, 0, 1])}
    gui = normalize_gui_clicks(actions, [{"isGuiOpen": True}] * 5)
    world = normalize_gui_clicks(actions, [{"isGuiOpen": False}] * 5)
    assert gui["attack"].tolist() == [1, 0, 0, 0, 1]
    assert world["attack"].tolist() == [1, 1, 1, 0, 1]


def test_automatic_filter_rejects_weak_gui_transition_without_click() -> None:
    images = [np.full((8, 8, 3), 80, dtype=np.uint8) for _ in range(5)]
    images[-1] = np.full((8, 8, 3), 100, dtype=np.uint8)
    actions = {"camera": np.zeros((4, 2)), "attack": np.zeros(4), "use": np.zeros(4)}
    reasons = automatic_quality_reasons(
        images,
        actions,
        [{"isGuiOpen": True}] * 5,
        "image_sequence_to_action",
    )
    assert "gui_change_without_click" in reasons


def test_automatic_filter_rejects_dark_and_camera_outlier() -> None:
    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    actions = {"camera": np.asarray([[500, 0]]), "attack": np.zeros(1)}
    reasons = automatic_quality_reasons(
        images,
        actions,
        [{"isGuiOpen": False}] * 4,
        "history_to_future_action",
    )
    assert {"image_too_dark", "camera_outlier"}.issubset(reasons)


def test_single_frame_intent_rejects_static_and_describes_movement() -> None:
    static = {"camera": np.zeros((4, 2)), "forward": np.zeros(4)}
    assert infer_action_intent(static) is None
    movement = {"camera": np.zeros((4, 2)), "forward": np.ones(4), "sprint": np.ones(4)}
    intent, category = infer_action_intent(movement)
    assert category == "movement"
    assert "move forward" in intent and "sprint" in intent
