"""重新生成第二轮动作候选，并把 AI 合格结果正式写入轨迹数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets.action_codec import decode_lumine_action
from datasets.minestudio_data.load import TrajectoryReader
from tools.trajectory_action_review import optimize_action_sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="提升 AI 合格的第二轮动作与意图标注")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_dir
    questions = read_jsonl(root / "questions.jsonl")
    answers = read_jsonl(root / "answer_key.jsonl")
    answer_by_id = {row["id"]: row for row in answers}
    first_reviews = {row["id"]: row for row in read_jsonl(root / "question_reviews.jsonl")}
    old_candidates = {
        row["id"]: row for row in read_jsonl(root / "second_round_preannotations.jsonl")
    }
    approved = [row for row in questions if first_reviews[row["id"]]["decision"] == "approve"]

    reader = TrajectoryReader([args.raw_dataset_dir], ["action", "meta_info"], 320, 180)
    episodes = set(reader.episode_names())
    candidates: list[dict[str, Any]] = []
    try:
        for question in approved:
            sample_id = question["id"]
            blocks = answer_by_id[sample_id]["reference_action_sequence"]
            start, end = question["target_interval"]
            gui_flags = None
            episode = question["source"]["episode"]
            if episode in episodes:
                metadata = reader.readers["meta_info"].read_frames(episode, start, end - start)
                gui_flags = [bool(item.get("isGuiOpen")) for item in metadata]
            sequence, stats = optimize_action_sequence(blocks, gui_flags)
            expected = question["inputs"]["action_block_ticks"]
            actual = [len(decode_lumine_action(block).chunks) for block in sequence]
            if actual != expected:
                raise ValueError(f"{sample_id} 的动作 tick {actual} 与题面 {expected} 不一致")
            old = old_candidates.get(sample_id, {})
            record = {
                "id": sample_id,
                "task_type": question["task_type"],
                "answer_sequence": sequence,
                "answer_reason": (
                    "非 GUI 鼠标微动按动作块中点分区，累计到最近图像边界；"
                    "逐 tick 按键、动作块时长、GUI 点击顺序和点击落点保持不变。"
                ),
                "optimization_stats": stats,
                "ai_decision": "approve",
                "preannotation_kind": "ai_second_round_answer_preannotation",
            }
            for key in ("suggested_intent", "intent_reason"):
                if old.get(key):
                    record[key] = old[key]
            candidates.append(record)
    finally:
        reader.close()

    candidate_by_id = {row["id"]: row for row in candidates}
    reviews = []
    for question in approved:
        sample_id = question["id"]
        candidate = candidate_by_id[sample_id]
        reviews.append({
            "id": sample_id,
            "decision": "approve",
            "reason": candidate["answer_reason"],
            "review_kind": "accepted_ai_second_round_action_review",
            "accepted_from": "ai_second_round_answer_preannotation",
            "reviewed_answer_sequence": candidate["answer_sequence"],
            "reference_kind": (
                "reviewed_optimized_demonstration"
                if question["task_type"] == "demonstration_optimization"
                else "reviewed_optimized_action_sequence"
            ),
            **(
                {"reviewed_intent": candidate["suggested_intent"]}
                if question["task_type"] == "single_frame_intent_to_action" else {}
            ),
        })
        question["review_status"] = "approved"
        question["include_in_training"] = True
        if question["task_type"] == "single_frame_intent_to_action":
            question["inputs"]["intent"] = candidate["suggested_intent"]
            question["inputs"]["intent_status"] = "accepted_ai_second_round_review"
        answer = answer_by_id[sample_id]
        answer["reference_action_sequence"] = candidate["answer_sequence"]
        answer["answer_reason"] = candidate["answer_reason"]
        answer["reference_kind"] = reviews[-1]["reference_kind"]

    write_jsonl(root / "second_round_preannotations.jsonl", candidates)
    write_jsonl(root / "action_reviews.jsonl", reviews)
    write_jsonl(root / "questions.jsonl", questions)
    write_jsonl(root / "answer_key.jsonl", answers)
    print(json.dumps({"promoted": len(reviews), "total_questions": len(questions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
