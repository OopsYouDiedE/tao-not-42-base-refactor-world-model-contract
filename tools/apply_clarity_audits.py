"""Apply visually reviewed keyframe exceptions to a trajectory review dataset."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from tools.trajectory_human_review import ReviewStore


KEEP_VERDICTS = {"keep_regular", "retain_current_frames"}


def audit_verdict(record: dict[str, Any]) -> str:
    for field in ("verdict", "status", "audit_decision"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"{record.get('id', '<unknown>')}: audit verdict is missing")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def bounded_minor_adjustment(record: dict[str, Any], maximum_offset: int = 3) -> list[int]:
    regular = [int(frame) for frame in record["current_regular_frames"]]
    desired = [int(frame) for frame in record["recommended_frames"]]
    choices = [
        range(max(regular[index] - maximum_offset, regular[0] + index),
              min(regular[index] + maximum_offset, regular[-1] - (len(regular) - index - 1)) + 1)
        for index in range(1, len(regular) - 1)
    ]
    candidates = (
        [regular[0], *middle, regular[-1]]
        for middle in itertools.product(*choices)
        if all(right > left for left, right in zip(middle, middle[1:]))
    )
    return min(
        candidates,
        key=lambda frames: (
            sum(abs(frame - target) for frame, target in zip(frames, desired)),
            sum(abs(frame - baseline) for frame, baseline in zip(frames, regular)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    args = parser.parse_args()

    recommendations: dict[str, dict[str, Any]] = {}
    for audit_path in args.audit:
        for record in read_jsonl(audit_path):
            record["normalized_verdict"] = audit_verdict(record)
            if record["normalized_verdict"] not in KEEP_VERDICTS:
                recommendations[record["id"]] = record

    preannotation_path = args.dataset_dir / "ai_question_preannotations.jsonl"
    preannotations = read_jsonl(preannotation_path)
    preannotation_by_id = {record["id"]: record for record in preannotations}
    missing = set(recommendations) - set(preannotation_by_id)
    if missing:
        raise ValueError(f"audit IDs absent from preannotations: {sorted(missing)[:5]}")

    store = ReviewStore(args.dataset_dir, raw_dataset_directory=args.raw_dataset_dir)
    applied: list[dict[str, Any]] = []
    try:
        for sample_id, audit in recommendations.items():
            question = store.questions[store.index_by_id[sample_id]]
            before = list(question["source"]["image_frames"])
            after = (
                bounded_minor_adjustment(audit)
                if audit["normalized_verdict"] == "minor_adjust"
                else [int(frame) for frame in audit["recommended_frames"]]
            )
            if len(after) != len(before) or after[0] != before[0] or after[-1] != before[-1]:
                raise ValueError(f"{sample_id}: recommendation changed the action window")
            if any(right <= left for left, right in zip(after, after[1:])):
                raise ValueError(f"{sample_id}: recommendation is not strictly increasing")

            preannotation = preannotation_by_id[sample_id]
            preannotation["image_frames"] = after
            rounds = preannotation.get("rounds", {})
            candidates = rounds.get("candidate_image_frames")
            selected_round = rounds.get("selected_round")
            if candidates and isinstance(selected_round, int) and 1 <= selected_round <= len(candidates):
                candidates[selected_round - 1] = after

            target_end = (
                int(question["target_interval"][1])
                if question["task_type"] == "history_to_future_action"
                else None
            )
            if after != before:
                store.save_adjustment(sample_id, after, target_end, rewrite_dataset=False)
            applied.append({
                "id": sample_id,
                "verdict": audit["normalized_verdict"],
                "before": before,
                "after": after,
            })

        store._rewrite_dataset_files()
        write_jsonl(preannotation_path, preannotations)
    finally:
        store.close()

    print(json.dumps({"applied": len(applied), "examples": applied[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
