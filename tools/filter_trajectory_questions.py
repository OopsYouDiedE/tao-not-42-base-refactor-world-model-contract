"""从 MineStudio 轨迹候选集中一致剔除已拒绝题目。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets.minestudio_finetune.generate_questions import (
    _read_jsonl,
    _write_jsonl,
    validate_generated_dataset,
    write_dataset_readme,
)


ACTIVE_FILES = (
    "questions.jsonl",
    "answer_key.jsonl",
    "ai_review_requests.jsonl",
    "human_review_templates.jsonl",
)


def read_rejections(path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    for record in records:
        if not record.get("id") or not record.get("reasons"):
            raise ValueError("拒绝记录必须包含 id 和 reasons")
    return records


def filter_rejected(dataset: Path, rejection_path: Path) -> dict[str, Any]:
    root = Path(dataset)
    rejections = read_rejections(rejection_path)
    rejected_ids = {record["id"] for record in rejections}
    questions = _read_jsonl(root / "questions.jsonl")
    answers = _read_jsonl(root / "answer_key.jsonl")
    known_ids = {question["id"] for question in questions}
    unknown = rejected_ids - known_ids
    if unknown:
        raise ValueError(f"拒绝记录包含未知题目：{sorted(unknown)}")
    rejected_questions = [question for question in questions if question["id"] in rejected_ids]
    for filename in ACTIVE_FILES:
        records = _read_jsonl(root / filename)
        _write_jsonl(root / filename, [record for record in records if record["id"] not in rejected_ids])
    removed_images: list[str] = []
    for question in rejected_questions:
        for relative in question["images"]:
            image_path = root / relative
            if image_path.is_file():
                image_path.unlink()
                removed_images.append(relative)
    remaining_questions = _read_jsonl(root / "questions.jsonl")
    remaining_answers = _read_jsonl(root / "answer_key.jsonl")
    write_dataset_readme(root, remaining_questions, remaining_answers)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_counts = Counter(question["task_type"] for question in remaining_questions)
    manifest.update({
        "sample_count": len(remaining_questions),
        "samples_per_type_after_screening": dict(task_counts),
        "screening_rejected_count": len(rejected_ids),
        "screening_rejections": rejection_path.name,
    })
    manifest["validation"] = validate_generated_dataset(root)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "remaining_samples": len(remaining_questions),
        "rejected_samples": len(rejected_ids),
        "removed_images": len(removed_images),
        "task_counts": dict(task_counts),
        "validation": manifest["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="剔除 MineStudio 轨迹候选题")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(
        filter_rejected(arguments.dataset_dir, arguments.rejections),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
