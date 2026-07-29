"""双审轨迹题到视觉 SFT messages 的适配测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from train.trajectory_question_dataset import load_approved_question_conversations


ACTION = "<|action_start|> ; MouseLeft ; ; MouseLeft <|action_end|>"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_approved_sequence_question_becomes_multimodal_messages(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    Image.new("RGB", (8, 8), "black").save(tmp_path / "images/0.jpg")
    question = {
        "id": "q",
        "task_type": "image_sequence_to_action",
        "prompt": "infer",
        "images": ["images/0.jpg"],
        "inputs": {},
        "review_status": "approved",
        "include_in_training": True,
    }
    answer = {
        "id": "q",
        "reference_action_sequence": [ACTION],
        "reference_kind": "recorded_human_demonstration",
    }
    _write_jsonl(tmp_path / "questions_approved.jsonl", [question])
    _write_jsonl(tmp_path / "answer_key.jsonl", [answer])
    conversation = load_approved_question_conversations(tmp_path)[0]
    assert conversation["messages"][0]["content"][0]["type"] == "image"
    assert json.loads(conversation["messages"][1]["content"][0]["text"]) == [ACTION]


def test_unreviewed_optimization_answer_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    Image.new("RGB", (8, 8), "black").save(tmp_path / "images/0.jpg")
    question = {
        "id": "q",
        "task_type": "demonstration_optimization",
        "prompt": "optimize",
        "images": ["images/0.jpg"],
        "inputs": {"raw_action_sequence": [ACTION]},
        "review_status": "approved",
        "include_in_training": True,
    }
    answer = {
        "id": "q",
        "reference_action_sequence": [ACTION],
        "reference_kind": "recorded_human_demonstration",
    }
    _write_jsonl(tmp_path / "questions_approved.jsonl", [question])
    _write_jsonl(tmp_path / "answer_key.jsonl", [answer])
    with pytest.raises(ValueError, match="审核后的优化答案"):
        load_approved_question_conversations(tmp_path)
