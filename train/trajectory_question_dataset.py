"""把双审通过的三类轨迹题转换为视觉 SFT messages。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def load_approved_question_conversations(
    dataset_directory: Path,
    maximum_samples: int | None = None,
) -> list[dict[str, list[dict[str, Any]]]]:
    """读取双审准入题和隔离答案，生成训练器可消费的多图对话。"""
    root = Path(dataset_directory)
    questions_path = root / "questions_approved.jsonl"
    answers_path = root / "answer_key.jsonl"
    if not questions_path.is_file() or not answers_path.is_file():
        raise FileNotFoundError("训练需要 questions_approved.jsonl 和 answer_key.jsonl")
    answers = {
        record["id"]: record
        for record in _read_jsonl(answers_path)
    }
    conversations: list[dict[str, list[dict[str, Any]]]] = []
    for question in _read_jsonl(questions_path):
        if not question.get("include_in_training") or question.get("review_status") != "approved":
            raise ValueError(f"题目 {question.get('id')} 未完成训练准入")
        answer = answers.get(question["id"])
        if answer is None:
            raise ValueError(f"题目 {question['id']} 缺少参考答案")
        if question["task_type"] == "demonstration_optimization" and answer.get(
            "reference_kind"
        ) != "reviewed_optimized_demonstration":
            raise ValueError(
                f"优化题 {question['id']} 没有审核后的优化答案，不能用原始轨迹训练",
            )
        content: list[dict[str, Any]] = []
        for relative in question["images"]:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"题目图片缺失：{path}")
            content.append({"type": "image", "image": Image.open(path).convert("RGB")})
        prompt = question["prompt"]
        raw = question.get("inputs", {}).get("raw_action_sequence")
        if raw:
            prompt += "\nRaw action sequence:\n" + json.dumps(raw, ensure_ascii=False)
        content.append({"type": "text", "text": prompt})
        conversations.append({
            "messages": [
                {"role": "user", "content": content},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": json.dumps(
                            answer["reference_action_sequence"], ensure_ascii=False,
                        ),
                    }],
                },
            ],
        })
        if maximum_samples is not None and len(conversations) >= maximum_samples:
            break
    if not conversations:
        raise ValueError("questions_approved.jsonl 中没有可训练题目")
    return conversations


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
