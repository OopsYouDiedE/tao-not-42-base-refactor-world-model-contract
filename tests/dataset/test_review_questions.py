"""Gradio 人工审核的数据保存与输入校验测试。"""

from __future__ import annotations

import json
from pathlib import Path

from dataset.trajectory.review_questions import ReviewStore, format_reference_actions


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def dataset(root: Path) -> None:
    write_jsonl(
        root / "questions.jsonl",
        [
            {
                "id": "q",
                "task_type": "image_sequence_to_action",
                "images": [],
                "source": {"episode": "e", "image_frames": []},
                "target_interval": [0, 4],
                "prompt": "infer",
                "inputs": {},
            }
        ],
    )
    write_jsonl(
        root / "answer_key.jsonl",
        [
            {
                "id": "q",
                "reference_action_sequence": ["action"],
            }
        ],
    )


def test_review_store_saves_and_resumes(tmp_path: Path) -> None:
    dataset(tmp_path)
    store = ReviewStore(tmp_path)
    record = {**store.default_review("q"), "decision": "reject", "reasons": ["dark"]}
    store.save(record)
    assert ReviewStore(tmp_path).review("q")["decision"] == "reject"
    assert not (tmp_path / "question_reviews.jsonl.tmp").exists()


def test_preannotation_does_not_count_as_human_review(tmp_path: Path) -> None:
    dataset(tmp_path)
    write_jsonl(
        tmp_path / "ai_question_preannotations.jsonl",
        [
            {
                "id": "q",
                "decision": "reject",
                "reason": "目标不清楚",
                "review_kind": "ai_question_preannotation",
            }
        ],
    )
    store = ReviewStore(tmp_path)
    assert store.review("q")["decision"] == "pending"
    assert store.displayed_review("q")["decision"] == "reject"
    assert store.counts()["pending"] == 1


def test_human_intent_is_written_into_single_frame_question(tmp_path: Path) -> None:
    dataset(tmp_path)
    question = json.loads((tmp_path / "questions.jsonl").read_text())
    question["task_type"] = "single_frame_intent_to_action"
    question["inputs"] = {"intent": "", "intent_status": "pending_human_authoring"}
    write_jsonl(tmp_path / "questions.jsonl", [question])
    store = ReviewStore(tmp_path)
    store.save_intent("q", "持续挖掘准星指向的石块")
    updated = json.loads((tmp_path / "questions.jsonl").read_text())
    assert updated["inputs"] == {
        "intent": "持续挖掘准星指向的石块",
        "intent_status": "human_authored",
    }


def test_complete_reference_actions_are_rendered() -> None:
    rendered = format_reference_actions(
        ["tick one", "tick two"],
        node_frames=[20, 25, 28],
        unoptimized=True,
    )
    assert "动作块 1" in rendered
    assert "tick one" in rendered
    assert "动作块 2" in rendered
    assert "tick two" in rendered
    assert "原始录制" in rendered
    assert "图像帧 20 → 25" in rendered
    assert "序列长度 5 tick" in rendered
    assert "图像帧 25 → 28" in rendered
    assert "序列长度 3 tick" in rendered
