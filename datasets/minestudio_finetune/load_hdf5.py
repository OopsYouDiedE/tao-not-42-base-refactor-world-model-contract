"""从轨迹题 HDF5 生成 Unsloth 视觉 SFT messages。"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import h5py
from PIL import Image


def format_question_prompt(question: dict[str, Any]) -> str:
    """构造训练和盲测共用的公开模型提示词。"""
    prompt = question["prompt"]
    ticks = question.get("inputs", {}).get("action_block_ticks")
    if ticks:
        prompt += "\nRequired action-block tick counts: " + json.dumps(ticks)
    prompt += (
        "\nAction format example for a 3-tick block: "
        '"<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>". '
        "Each JSON array item must be one string action block; do not return nested tick arrays."
    )
    raw = question.get("inputs", {}).get("raw_action_sequence")
    if raw:
        prompt += "\nRaw action sequence:\n" + json.dumps(raw, ensure_ascii=False)
    intent = question.get("inputs", {}).get("intent")
    if intent:
        prompt += f"\nIntent: {intent}"
    return prompt


def load_hdf5_conversations(
    archive_path: Path,
    maximum_samples: int | None = None,
) -> list[dict[str, list[dict[str, Any]]]]:
    """加载完整 HDF5，并将图像放在文本提示词之前。"""
    conversations: list[dict[str, list[dict[str, Any]]]] = []
    with h5py.File(archive_path, "r") as archive:
        if archive.attrs.get("format") != "minestudio_trajectory_sft_v1":
            raise ValueError("不是受支持的 MineStudio 轨迹 SFT HDF5")
        for group in archive["samples"].values():
            question = json.loads(group.attrs["question_json"])
            answer = json.loads(group.attrs["answer_json"])
            content: list[dict[str, Any]] = []
            for dataset in group["images"].values():
                image = Image.open(io.BytesIO(dataset[()].tobytes())).convert("RGB")
                content.append({"type": "image", "image": image})
            prompt = format_question_prompt(question)
            content.append({"type": "text", "text": prompt})
            conversations.append({"messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{
                    "type": "text",
                    "text": json.dumps(answer["reference_action_sequence"], ensure_ascii=False),
                }]},
            ]})
            if maximum_samples is not None and len(conversations) >= maximum_samples:
                break
    if not conversations:
        raise ValueError("HDF5 中没有训练样本")
    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 MineStudio 轨迹 SFT HDF5")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--maximum-samples", type=int)
    arguments = parser.parse_args()
    conversations = load_hdf5_conversations(arguments.archive, arguments.maximum_samples)
    print(json.dumps({"sample_count": len(conversations)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
