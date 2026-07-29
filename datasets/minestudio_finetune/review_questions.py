"""生成 AI 审核请求、执行结构审计并按人工/AI 双审结果筛选题目。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from datasets.action_codec import decode_lumine_action
from datasets.minestudio_finetune.generate_questions import write_dataset_readme
from datasets.minestudio_finetune.question_schema import (
    HARD_REJECTION_REASONS,
    REVIEW_DIMENSIONS,
    TASK_TYPES,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number} 不是合法 JSON") from error
    return records


def structural_review(question: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    """执行不依赖模型的硬规则检查，返回可审计结果。"""
    reasons: list[str] = []
    if question.get("task_type") not in TASK_TYPES:
        reasons.append("unknown_task_type")
    images = question.get("images")
    if not isinstance(images, list) or not images:
        reasons.append("missing_image")
        images = []
    for relative in images:
        path = dataset_root / str(relative)
        if not path.is_file():
            reasons.append("missing_image")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            reasons.append("corrupt_image")
    source = question.get("source") or {}
    frames = source.get("image_frames") or []
    if frames != sorted(frames) or len(frames) != len(set(frames)):
        reasons.append("non_monotonic_frames")
    target = question.get("target_interval") or []
    if question.get("task_type") == "history_to_future_action":
        if len(target) == 2 and any(frame > target[0] for frame in frames):
            reasons.append("future_leakage")
        if "raw_action_sequence" in (question.get("inputs") or {}):
            reasons.append("future_leakage")
    raw = (question.get("inputs") or {}).get("raw_action_sequence", [])
    try:
        for block in raw:
            if not decode_lumine_action(block).chunks:
                raise ValueError("动作块没有任何 chunk")
    except (TypeError, ValueError):
        reasons.append("invalid_action_contract")
    unique_reasons = sorted(set(reasons))
    return {
        "id": question.get("id"),
        "reviewer": "deterministic_structure_audit_v1",
        "decision": "reject" if unique_reasons else "pass",
        "hard_rejection": bool(HARD_REJECTION_REASONS.intersection(unique_reasons)),
        "reasons": unique_reasons,
    }


def ai_review_request(question: dict[str, Any]) -> dict[str, Any]:
    """构造可交给任意视觉模型的审核请求；本函数不调用外部 API。"""
    rubric = [
        {"dimension": name, "criterion": criterion, "score": "integer 1-5"}
        for name, criterion in REVIEW_DIMENSIONS.items()
    ]
    return {
        "id": question["id"],
        "images": question["images"],
        "system": (
            "You review Minecraft trajectory-training questions. Inspect every supplied image and "
            "the complete question. Return JSON only. A score below 3 on any dimension requires "
            "reject. Uncertain visual evidence requires revise or reject, never an invented claim."
        ),
        "question_under_review": question,
        "rubric": rubric,
        "required_response": {
            "id": question["id"],
            "reviewer_kind": "ai",
            "decision": "approve | revise | reject",
            "scores": {name: "1-5" for name in REVIEW_DIMENSIONS},
            "reasons": ["specific evidence tied to images or fields"],
            "suggested_revision": "string or null",
        },
    }


def human_review_template(question: dict[str, Any]) -> dict[str, Any]:
    """生成与 AI 审核同量表的人工审核空表。"""
    return {
        "id": question["id"],
        "reviewer_kind": "human",
        "decision": "revise",
        "scores": {dimension: 0 for dimension in REVIEW_DIMENSIONS},
        "reasons": [],
        "suggested_revision": None,
    }


def valid_review(record: dict[str, Any], reviewer_kind: str) -> bool:
    if record.get("reviewer_kind") != reviewer_kind:
        return False
    if record.get("decision") not in {"approve", "revise", "reject"}:
        return False
    scores = record.get("scores")
    return isinstance(scores, dict) and all(
        isinstance(scores.get(dimension), int) and 1 <= scores[dimension] <= 5
        for dimension in REVIEW_DIMENSIONS
    )


def approved_question_ids(
    questions: Iterable[dict[str, Any]],
    structure_reviews: Iterable[dict[str, Any]],
    ai_reviews: Iterable[dict[str, Any]],
    human_reviews: Iterable[dict[str, Any]],
) -> set[str]:
    """只有结构通过、AI 批准、人工批准且各维度至少 3 分的题目可以进入训练。"""
    structure = {record["id"]: record for record in structure_reviews}
    ai = {record["id"]: record for record in ai_reviews if valid_review(record, "ai")}
    human = {
        record["id"]: record for record in human_reviews if valid_review(record, "human")
    }
    approved: set[str] = set()
    for question in questions:
        sample_id = question["id"]
        reviews = (ai.get(sample_id), human.get(sample_id))
        if structure.get(sample_id, {}).get("decision") != "pass" or None in reviews:
            continue
        if all(
            review["decision"] == "approve" and min(review["scores"].values()) >= 3
            for review in reviews
            if review is not None
        ):
            approved.add(sample_id)
    return approved


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="审计并筛选 MineStudio 三类轨迹题")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ai-reviews", type=Path)
    parser.add_argument("--human-reviews", type=Path)
    arguments = parser.parse_args()
    questions_path = arguments.dataset_dir / "questions.jsonl"
    questions = read_jsonl(questions_path)
    structure = [structural_review(question, arguments.dataset_dir) for question in questions]
    _write_jsonl(arguments.dataset_dir / "structure_reviews.jsonl", structure)
    _write_jsonl(
        arguments.dataset_dir / "ai_review_requests.jsonl",
        (ai_review_request(question) for question in questions),
    )
    _write_jsonl(
        arguments.dataset_dir / "human_review_template.jsonl",
        (human_review_template(question) for question in questions),
    )
    answer_key_path = arguments.dataset_dir / "answer_key.jsonl"
    if answer_key_path.is_file():
        write_dataset_readme(
            arguments.dataset_dir, questions, read_jsonl(answer_key_path), structure,
        )
    if arguments.ai_reviews and arguments.human_reviews:
        approved = approved_question_ids(
            questions, structure, read_jsonl(arguments.ai_reviews),
            read_jsonl(arguments.human_reviews),
        )
        selected = [
            {**question, "review_status": "approved", "include_in_training": True}
            for question in questions if question["id"] in approved
        ]
        _write_jsonl(arguments.dataset_dir / "questions_approved.jsonl", selected)
    print(json.dumps({
        "questions": len(questions),
        "structure_passed": sum(item["decision"] == "pass" for item in structure),
        "ai_review_requests": len(questions),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
