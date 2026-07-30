"""双审准入题的 HDF5 打包和视觉 SFT 加载测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from datasets.minestudio_finetune.load_hdf5 import format_question_prompt, load_hdf5_conversations
from datasets.minestudio_finetune.pack_hdf5 import pack_approved_questions
from datasets.minestudio_finetune.sft_protocol import format_assistant_response, sanitize_intent


ACTION = "<|action_start|> ; MouseLeft ; ; MouseLeft <|action_end|>"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _dataset(root: Path, task_type: str = "image_sequence_to_action") -> None:
    (root / "images").mkdir()
    Image.new("RGB", (8, 8), "green").save(root / "images/0.jpg")
    question = {
        "id": "q", "task_type": task_type, "prompt": "infer", "images": ["images/0.jpg"],
        "inputs": {}, "review_status": "pending_human_and_ai_review",
        "include_in_training": False,
    }
    answer = {
        "id": "q", "reference_action_sequence": [ACTION],
        "reference_kind": "recorded_human_demonstration",
    }
    review = {
        "id": "q", "decision": "approve", "scores": {"quality": 5},
        "reason": "动作与图像一致", "reviewed_answer_sequence": [ACTION],
    }
    _write_jsonl(root / "questions.jsonl", [question])
    _write_jsonl(root / "answer_key.jsonl", [answer])
    _write_jsonl(root / "ai_reviews.jsonl", [review])
    _write_jsonl(root / "human_reviews.jsonl", [review])


def test_pack_and_load_multimodal_messages(tmp_path: Path) -> None:
    _dataset(tmp_path)
    archive = tmp_path / "train.h5"
    assert pack_approved_questions(tmp_path, archive)["sample_count"] == 1
    conversation = load_hdf5_conversations(archive)[0]
    assert conversation["messages"][0]["content"][0]["type"] == "image"
    answer_text = conversation["messages"][1]["content"][0]["text"]
    actions, reason = answer_text.split("\nReason: ", 1)
    assert json.loads(actions) == [ACTION]
    assert reason == "动作与图像一致"


def test_pack_rejects_missing_reviewed_answer(tmp_path: Path) -> None:
    _dataset(tmp_path, "demonstration_optimization")
    review = {"id": "q", "decision": "approve", "reason": "动作与图像一致"}
    _write_jsonl(tmp_path / "human_reviews.jsonl", [review])
    with pytest.raises(ValueError, match="双审后的动作答案"):
        pack_approved_questions(tmp_path, tmp_path / "train.h5")


def test_pack_rejects_unapproved_question(tmp_path: Path) -> None:
    _dataset(tmp_path)
    review = json.loads((tmp_path / "human_reviews.jsonl").read_text())
    review["decision"] = "reject"
    _write_jsonl(tmp_path / "human_reviews.jsonl", [review])
    with pytest.raises(ValueError, match="没有同时通过"):
        pack_approved_questions(tmp_path, tmp_path / "train.h5")


def test_pack_accepts_approval_without_scores(tmp_path: Path) -> None:
    _dataset(tmp_path)
    review = {
        "id": "q", "decision": "approve", "reason": "画面与动作一致",
        "reviewed_answer_sequence": [ACTION],
    }
    _write_jsonl(tmp_path / "ai_reviews.jsonl", [review])
    _write_jsonl(tmp_path / "human_reviews.jsonl", [review])
    assert pack_approved_questions(tmp_path, tmp_path / "train.h5")["sample_count"] == 1


def test_pack_uses_reviewed_answer_for_every_task(tmp_path: Path) -> None:
    _dataset(tmp_path)
    reviewed = "<|action_start|> ; W ; W ; W <|action_end|>"
    review = {
        "id": "q", "decision": "approve", "reason": "采用规范移动动作",
        "reviewed_answer_sequence": [reviewed],
    }
    _write_jsonl(tmp_path / "ai_reviews.jsonl", [review])
    _write_jsonl(tmp_path / "human_reviews.jsonl", [review])
    archive = tmp_path / "train.h5"
    pack_approved_questions(tmp_path, archive)
    answer_text = load_hdf5_conversations(archive)[0]["messages"][1]["content"][0]["text"]
    actions, reason = answer_text.split("\nReason: ", 1)
    assert json.loads(actions) == [reviewed]
    assert reason == "采用规范移动动作"


def test_format_question_prompt_includes_public_timing() -> None:
    prompt = format_question_prompt({
        "prompt": "infer", "inputs": {"action_block_ticks": [5, 8], "intent": "挖掘石块"},
    })
    assert "Required action-block tick counts: [5, 8]" in prompt
    assert "Intent: 挖掘石块" in prompt
    assert "do not return nested tick arrays" in prompt
    assert "Reason:" in prompt


def test_future_prompt_hides_reference_tick_count_and_asks_for_suitable_duration() -> None:
    prompt = format_question_prompt({
        "task_type": "single_frame_intent_to_action",
        "prompt": "legacy prompt",
        "inputs": {"action_block_ticks": [60], "intent": "向前疾跑（59 tick，向右转向）"},
    })
    assert "Required action-block tick counts" not in prompt
    assert "59 tick" not in prompt
    assert "Intent: 向前疾跑（向右转向）" in prompt
    assert "suitable number" in prompt
    assert "up to 60 ticks" in prompt


def test_action_first_response_starts_with_independently_parseable_json() -> None:
    question = {
        "task_type": "history_to_future_action", "inputs": {},
    }
    answer = {"reference_action_sequence": [ACTION], "answer_reason": ""}
    response = format_assistant_response(question, answer)
    actions, reason = response.split("\nReason: ", 1)
    assert json.loads(actions) == [ACTION]
    assert reason


def test_sanitize_intent_removes_tick_answer_without_dropping_direction() -> None:
    assert sanitize_intent("沿路线疾跑（39 tick，向右平视）") == "沿路线疾跑（向右平视）"
