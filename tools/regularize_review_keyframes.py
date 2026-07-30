"""Regularize reviewed keyframes while preserving each approved action window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.trajectory_human_review import ReviewStore


MULTI_FRAME_TASKS = {
    "demonstration_optimization",
    "history_to_future_action",
    "image_sequence_to_action",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def evenly_spaced_frames(start: int, end: int, count: int) -> list[int]:
    """Return integer nodes whose adjacent gaps differ by at most one tick."""
    if count < 2 or end - start < count - 1:
        raise ValueError(f"cannot place {count} frames in [{start}, {end}]")
    intervals = count - 1
    return [start + round(position * (end - start) / intervals) for position in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    preannotation_path = args.dataset_dir / "ai_question_preannotations.jsonl"
    preannotations = read_jsonl(preannotation_path)
    store = ReviewStore(args.dataset_dir, raw_dataset_directory=args.raw_dataset_dir)
    changed: list[dict[str, Any]] = []
    try:
        for preannotation in preannotations:
            question = store.questions[store.index_by_id[preannotation["id"]]]
            if question["task_type"] not in MULTI_FRAME_TASKS:
                continue
            current = [int(frame) for frame in preannotation["image_frames"]]
            regularized = evenly_spaced_frames(current[0], current[-1], len(current))
            if regularized == current:
                continue

            preannotation["image_frames"] = regularized
            rounds = preannotation.get("rounds", {})
            candidates = rounds.get("candidate_image_frames")
            selected_round = rounds.get("selected_round")
            if candidates and isinstance(selected_round, int) and 1 <= selected_round <= len(candidates):
                candidates[selected_round - 1] = regularized

            target_end = (
                int(preannotation["target_interval"][1])
                if question["task_type"] == "history_to_future_action"
                else None
            )
            store.save_adjustment(
                preannotation["id"], regularized, target_end, rewrite_dataset=False,
            )
            changed.append({"id": preannotation["id"], "before": current, "after": regularized})

        store._rewrite_dataset_files()
        write_jsonl(preannotation_path, preannotations)
    finally:
        store.close()

    print(json.dumps({"changed": len(changed), "examples": changed[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
