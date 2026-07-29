"""三类轨迹题的时间边界、审核准入和做题测试。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from datasets.minestudio_finetune.generate_questions import (
    build_question_record,
    source_frames,
    write_dataset_readme,
)
from datasets.minestudio_finetune.question_schema import OUTPUT_CONTRACT, REVIEW_DIMENSIONS, TASK_TYPES
from datasets.minestudio_finetune.review_questions import approved_question_ids, structural_review
from datasets.minestudio_finetune.test_answers import evaluate_responses, parse_answer


ACTION = "<|action_start|> ; Mouse 35 30 ; W ; W D Mouse 4 -2 <|action_end|>"


def test_three_required_task_types_are_stable() -> None:
    assert TASK_TYPES == (
        "demonstration_optimization",
        "image_to_action",
        "history_to_future_action",
    )
    assert "cursor in GUI" in OUTPUT_CONTRACT["mouse"]


def test_prediction_frames_do_not_cross_target_start() -> None:
    assert source_frames("image_to_action", 20) == [20]
    assert source_frames("history_to_future_action", 20) == [8, 12, 16, 20]


def test_optimization_frames_cover_chronological_sequence() -> None:
    assert source_frames("demonstration_optimization", 20) == [20, 24, 28, 32]
    question = build_question_record(
        "q", "demonstration_optimization", "episode", 20, ["a", "b", "c", "d"],
        [ACTION, ACTION, ACTION, ACTION],
    )
    assert question["target_interval"] == [20, 36]


def test_structural_review_rejects_future_leakage(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), "black").save(image_path)
    question = build_question_record(
        "q", "history_to_future_action", "episode", 20, [image_path.name],
    )
    question["source"]["image_frames"] = [8, 12, 16, 21]
    review = structural_review(question, tmp_path)
    assert review["decision"] == "reject"
    assert "future_leakage" in review["reasons"]


def _review(sample_id: str, kind: str, decision: str = "approve", score: int = 5) -> dict:
    return {
        "id": sample_id,
        "reviewer_kind": kind,
        "decision": decision,
        "scores": {dimension: score for dimension in REVIEW_DIMENSIONS},
    }


def test_training_requires_structure_ai_and_human_approval() -> None:
    questions = [{"id": "q"}]
    structure = [{"id": "q", "decision": "pass"}]
    assert approved_question_ids(
        questions, structure, [_review("q", "ai")], [_review("q", "human")],
    ) == {"q"}
    assert approved_question_ids(
        questions, structure, [_review("q", "ai", score=2)], [_review("q", "human")],
    ) == set()


def test_answer_parser_requires_exact_block_count() -> None:
    assert len(parse_answer([ACTION], 1)) == 1
    try:
        parse_answer([], 1)
    except ValueError as error:
        assert "恰好包含 1 个" in str(error)
    else:
        raise AssertionError("空答案应被拒绝")


def test_answer_evaluation_marks_reference_as_non_unique() -> None:
    questions = [{"id": "q", "task_type": "image_to_action"}]
    key = [{"id": "q", "reference_action_sequence": [ACTION]}]
    responses = [{"id": "q", "answer": [ACTION]}]
    result = evaluate_responses(questions, key, responses)[0]
    assert result["format_valid"] is True
    assert result["reference_similarity"] == 1.0
    assert result["requires_semantic_review"] is True


def test_generated_readme_contains_images_question_answer_and_review(tmp_path: Path) -> None:
    question = build_question_record("q", "image_to_action", "episode", 20, ["images/q.jpg"])
    answer = {"id": "q", "reference_action_sequence": [ACTION]}
    review = {"id": "q", "decision": "pass", "reasons": []}
    write_dataset_readme(tmp_path, [question], [answer], [review])
    report = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "![q frame 20](images/q.jpg)" in report
    assert question["prompt"] in report
    assert ACTION in report
    assert '\"decision\": \"pass\"' in report
