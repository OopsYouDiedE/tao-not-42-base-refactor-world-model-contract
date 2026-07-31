from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.trajectory.review_actions import ActionReviewStore, optimize_action_sequence
from lumine.action_codec import decode_lumine_action


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def dataset(root: Path) -> None:
    question = {
        "id": "q",
        "task_type": "image_sequence_to_action",
        "images": ["a", "b"],
        "source": {"image_frames": [10, 16], "episode": "e"},
        "target_interval": [10, 16],
    }
    action = "<|action_start|> ; Mouse 1 0 W ; Mouse 2 0 W ; W ; W ; <|action_end|>"
    write_jsonl(root / "questions.jsonl", [question])
    write_jsonl(root / "answer_key.jsonl", [{"id": "q", "reference_action_sequence": [action]}])
    write_jsonl(root / "question_reviews.jsonl", [{"id": "q", "decision": "approve"}])


def test_gameplay_compression_preserves_held_duration_and_merges_mouse() -> None:
    action = (
        "<|action_start|> ; Mouse 1 0 W ; Mouse 2 0 W ; W ; "
        "Mouse 4 1 W ; Mouse 5 2 W <|action_end|>"
    )
    optimized, stats = optimize_action_sequence([action], [False] * 5)
    assert stats["raw_ticks"] == 5
    assert stats["optimized_ticks"] == 5
    assert stats["compressed_held_ticks"] == 0
    chunks = decode_lumine_action(optimized[0]).chunks
    assert [chunk.mouse for chunk in chunks] == [(3, 0), (0, 0), (0, 0), (0, 0), (9, 3)]
    assert all(chunk.keys == ("W",) for chunk in chunks)


def test_gui_compression_merges_cursor_path_into_click() -> None:
    action = "<|action_start|> ; Mouse 2 1 ; Mouse 3 2 ; Mouse 1 0 MouseLeft ; <|action_end|>"
    optimized, stats = optimize_action_sequence([action], [True] * 4)
    assert stats["optimized_ticks"] == 4
    assert "Mouse 6 3 MouseLeft" in optimized[0]


def test_store_only_includes_approved_and_saves_modified_answer(tmp_path: Path) -> None:
    dataset(tmp_path)
    store = ActionReviewStore(tmp_path)
    assert len(store.questions) == 1
    candidate = store.candidates["q"]["answer_sequence"]
    store.save("q", json.dumps(candidate), "", "approve", "题目成立，压缩回答准确")
    review = json.loads((tmp_path / "action_reviews.jsonl").read_text())
    assert review["reference_kind"] == "reviewed_optimized_action_sequence"


def test_save_rejects_wrong_block_count(tmp_path: Path) -> None:
    dataset(tmp_path)
    store = ActionReviewStore(tmp_path)
    with pytest.raises(ValueError, match="需要 1 个动作块"):
        store.save(
            "q", json.dumps(["<|action_start|> ; W <|action_end|>"] * 2), "", "approve", "理由"
        )
