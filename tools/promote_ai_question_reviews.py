"""Promote a complete AI question audit into the first-round review ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    questions_path = args.dataset_dir / "questions.jsonl"
    preannotations_path = args.dataset_dir / "ai_question_preannotations.jsonl"
    reviews_path = args.dataset_dir / "question_reviews.jsonl"
    questions = read_jsonl(questions_path)
    preannotations = {row["id"]: row for row in read_jsonl(preannotations_path)}
    question_ids = {row["id"] for row in questions}
    if set(preannotations) != question_ids:
        raise ValueError("AI preannotation IDs do not exactly cover questions")

    reviews = []
    for question in questions:
        preannotation = preannotations[question["id"]]
        decision = preannotation.get("decision")
        if decision not in {"approve", "reject"}:
            raise ValueError(f"{question['id']}: invalid decision {decision!r}")
        if question["task_type"] == "single_frame_intent_to_action" and decision == "approve":
            intent = str(preannotation.get("suggested_intent", "")).strip()
            if not intent:
                raise ValueError(f"{question['id']}: approved single-frame item lacks intent")
            question.setdefault("inputs", {})["intent"] = intent
            question["inputs"]["intent_status"] = "ai_preannotated_pending_second_round_review"
        reviews.append({
            "id": question["id"],
            "decision": decision,
            "reason": preannotation["reason"],
            "reasons": preannotation.get("reasons", [preannotation["reason"]]),
            "review_kind": "accepted_ai_question_review",
            "accepted_from": "ai_question_preannotation",
            "image_frames": question["source"]["image_frames"],
            "target_interval": question["target_interval"],
        })

    write_jsonl(questions_path, questions)
    write_jsonl(reviews_path, reviews)
    print(json.dumps({
        "total": len(reviews),
        "approve": sum(row["decision"] == "approve" for row in reviews),
        "reject": sum(row["decision"] == "reject" for row in reviews),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
