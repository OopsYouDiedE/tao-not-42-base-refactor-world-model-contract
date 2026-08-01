"""从轨迹题 HDF5 生成 Unsloth 视觉 SFT messages。"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any

import h5py
from PIL import Image

from dataset.organization.sft_protocol import (
    format_assistant_response,
    format_question_prompt,
)


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
            conversations.append(
                {
                    "messages": [
                        {"role": "user", "content": content},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": format_assistant_response(question, answer),
                                }
                            ],
                        },
                    ]
                }
            )
            if maximum_samples is not None and len(conversations) >= maximum_samples:
                break
    if not conversations:
        raise ValueError("HDF5 中没有训练样本")
    return conversations


def split_conversations(
    conversations: list[dict[str, list[dict[str, Any]]]],
    validation_ratio: float = 0.1,
    seed: int = 3407,
) -> tuple[list[dict[str, list[dict[str, Any]]]], list[dict[str, list[dict[str, Any]]]]]:
    """按固定种子划分互斥的训练集和验证集。"""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")
    if len(conversations) < 2:
        raise ValueError("训练/验证划分至少需要两个样本")
    indices = list(range(len(conversations)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, round(len(indices) * validation_ratio))
    validation_indices = set(indices[:validation_size])
    training = [
        sample for index, sample in enumerate(conversations) if index not in validation_indices
    ]
    validation = [
        sample for index, sample in enumerate(conversations) if index in validation_indices
    ]
    return training, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 MineStudio 轨迹 SFT HDF5")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--maximum-samples", type=int)
    arguments = parser.parse_args()
    conversations = load_hdf5_conversations(arguments.archive, arguments.maximum_samples)
    print(json.dumps({"sample_count": len(conversations)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
