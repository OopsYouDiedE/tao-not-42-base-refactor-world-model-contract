"""把双审通过的轨迹题、答案和 JPEG 图片打包为单个 HDF5。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _approved_review(record: dict[str, Any]) -> bool:
    scores = record.get("scores")
    scores_are_valid = (
        scores is None
        or (
            isinstance(scores, dict)
            and bool(scores)
            and all(isinstance(score, int) and score >= 3 for score in scores.values())
        )
    )
    return record.get("decision") == "approve" and scores_are_valid


def pack_approved_questions(
    dataset_directory: Path,
    output_path: Path,
    ai_reviews_path: Path | None = None,
    human_reviews_path: Path | None = None,
    final_reviews_path: Path | None = None,
) -> dict[str, Any]:
    """根据 AI 与人工双审结果，只打包双方批准且拥有合法答案的题目。"""
    root = Path(dataset_directory)
    questions = _read_jsonl(root / "questions.jsonl")
    answers = {record["id"]: record for record in _read_jsonl(root / "answer_key.jsonl")}
    if final_reviews_path is not None:
        final_reviews = {record["id"]: record for record in _read_jsonl(final_reviews_path)}
        ai_reviews = human_reviews = final_reviews
    else:
        ai_reviews = {
            record["id"]: record
            for record in _read_jsonl(ai_reviews_path or root / "ai_reviews.jsonl")
        }
        human_reviews = {
            record["id"]: record
            for record in _read_jsonl(human_reviews_path or root / "human_reviews.jsonl")
        }
    approved = [
        question for question in questions
        if _approved_review(ai_reviews.get(question["id"], {}))
        and _approved_review(human_reviews.get(question["id"], {}))
    ]
    if not approved:
        raise ValueError("没有同时通过 AI 与人工审核的题目")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as archive:
        archive.attrs["format"] = "minestudio_trajectory_sft_v1"
        samples = archive.create_group("samples", track_order=True)
        for index, question in enumerate(approved):
            sample_id = question["id"]
            answer = answers.get(sample_id)
            if answer is None:
                raise ValueError(f"题目 {sample_id} 缺少答案")
            ai_sequence = ai_reviews[sample_id].get("reviewed_answer_sequence")
            reviewed_sequence = human_reviews[sample_id].get("reviewed_answer_sequence")
            if not ai_sequence or not reviewed_sequence:
                raise ValueError(f"题目 {sample_id} 没有 AI 与人工双审后的动作答案")
            if ai_sequence != reviewed_sequence:
                raise ValueError(f"题目 {sample_id} 的 AI 与人工最终动作答案不一致")
            reference_kind = (
                "reviewed_optimized_demonstration"
                if question["task_type"] == "demonstration_optimization"
                else "reviewed_optimized_action_sequence"
            )
            answer = {
                **answer,
                "reference_action_sequence": reviewed_sequence,
                "answer_reason": human_reviews[sample_id].get("reason", ""),
                "reference_kind": reference_kind,
            }
            question = {
                **question, "review_status": "approved", "include_in_training": True,
            }
            group = samples.create_group(f"{index:08d}")
            group.attrs["id"] = sample_id
            group.attrs["question_json"] = json.dumps(question, ensure_ascii=False)
            group.attrs["answer_json"] = json.dumps(answer, ensure_ascii=False)
            image_group = group.create_group("images", track_order=True)
            for image_index, relative in enumerate(question["images"]):
                image_path = root / relative
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                payload = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
                image_group.create_dataset(f"{image_index:04d}", data=payload, compression="gzip")
        archive.attrs["sample_count"] = len(approved)
    return {"format": "minestudio_trajectory_sft_v1", "sample_count": len(approved), "path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="把双审通过的 MineStudio 轨迹题打包为 HDF5")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ai-reviews", type=Path)
    parser.add_argument("--human-reviews", type=Path)
    parser.add_argument("--final-reviews", type=Path)
    arguments = parser.parse_args()
    result = pack_approved_questions(
        arguments.dataset_dir, arguments.output, arguments.ai_reviews, arguments.human_reviews,
        arguments.final_reviews,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
