"""四类带图训练任务 Demo 的数据契约测试。"""

from __future__ import annotations

from pathlib import Path

from bc_datasets.multimodal_task_demo import DEMO_CASES

DEMO_DIRECTORY = Path(__file__).parents[2] / "bc_datasets" / "multimodal_task_demo"


def test_demo_covers_all_requested_task_types() -> None:
    assert {case.task_type for case in DEMO_CASES} == {
        "action_optimization",
        "inverse_action_generation",
        "future_action_choice",
        "macro_intent_classification",
    }


def test_every_case_has_chronological_images_on_disk() -> None:
    for case in DEMO_CASES:
        assert len(case.images) >= 2
        assert all((DEMO_DIRECTORY / image).is_file() for image in case.images)


def test_action_optimization_receives_original_action() -> None:
    case = next(case for case in DEMO_CASES if case.task_type == "action_optimization")
    assert "original_action" in case.inputs
    assert str(case.answer).startswith("<|action_start|>")
    assert str(case.answer).endswith("<|action_end|>")


def test_inverse_generation_never_receives_original_action() -> None:
    case = next(case for case in DEMO_CASES if case.task_type == "inverse_action_generation")
    assert "original_action" not in case.inputs
    assert "previous_action" not in case.inputs
    assert case.inputs["optimization_rule"]


def test_future_prediction_is_strict_four_choice_without_actions() -> None:
    case = next(case for case in DEMO_CASES if case.task_type == "future_action_choice")
    choices = case.inputs["choices"]
    assert tuple(choices) == ("A", "B", "C", "D")
    assert len(set(choices.values())) == 4
    assert case.answer in choices
    assert "original_action" not in case.inputs
    assert "previous_action" not in case.inputs


def test_macro_intent_uses_distinct_controlled_labels() -> None:
    case = next(case for case in DEMO_CASES if case.task_type == "macro_intent_classification")
    choices = case.inputs["choices"]
    assert tuple(choices) == ("A", "B", "C", "D")
    assert len(set(choices.values())) == 4
    assert case.answer in choices


def test_every_case_records_answer_aware_and_blind_assessments() -> None:
    for case in DEMO_CASES:
        assert case.answer_aware_assessment
        assert case.blind_answer is not None
        assert case.blind_assessment
