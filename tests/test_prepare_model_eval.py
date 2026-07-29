"""模型盲测包必须隔离答案并保留图片顺序。"""

from __future__ import annotations

import json
from pathlib import Path

from datasets.minestudio_finetune.prepare_model_eval import prepare_model_eval


def test_prepare_model_eval_excludes_answers(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = dataset / "blind"
    dataset.mkdir()
    question = {
        "id": "q",
        "task_type": "image_sequence_to_action",
        "prompt": "infer",
        "images": ["images/0.jpg", "images/1.jpg"],
        "inputs": {"images_chronological": True},
        "output_contract": {"chunk_count": "variable"},
    }
    (dataset / "questions.jsonl").write_text(json.dumps(question) + "\n", encoding="utf-8")
    result = prepare_model_eval(dataset, output)
    request = json.loads((output / "requests.jsonl").read_text(encoding="utf-8"))
    assert result["contains_answer_key"] is False
    assert request["images"] == ["../images/0.jpg", "../images/1.jpg"]
    assert "answer" not in request
