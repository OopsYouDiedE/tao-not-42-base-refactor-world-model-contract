"""Lumine 预训练样本 → Unsloth 视觉 SFT 的对话格式数据集。

对外接口：
    DEFAULT_INSTRUCTION — 动作预测任务的默认指令文本。
    SubsetName — 可读取的子集名。
    build_conversation — 单条 Lumine 样本 → messages 对话。
    load_lumine_conversations — 读 ``samples_<子集>.jsonl``，产出可直接喂 SFTTrainer 的数据集。

对话布局遵循 Unsloth 视觉微调的硬约束：**图像必须排在文本指令之前**，assistant 回复
只含 Lumine 动作串，使模型的监督目标就是动作 token 本身。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image

SubsetName = Literal["train", "validation"]

DEFAULT_INSTRUCTION = (
    "You are controlling a Minecraft player. Given the current view"
    " (and preceding frames if provided), output the actions to execute over the next"
    " 200 ms in Lumine format: mouse dx dy dz, then one key set per 50 ms chunk"
    " separated by ';'. Keys held across consecutive chunks stay pressed."
)


def build_conversation(
    sample: dict[str, Any],
    dataset_root: Path,
    instruction: str = DEFAULT_INSTRUCTION,
    include_previous_action: bool = True,
    loaded_images: list[Image.Image] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """把一条 Lumine 样本转成 messages 对话。

    Parameters
    ----------
    sample : dict
        样本 JSONL 的一行，含 ``image_paths`` / ``action_text`` /
        ``previous_action_text``。
    dataset_root : Path
        样本 JSONL 所在目录，用于解析 ``image_paths`` 的相对路径。
    instruction : str
        任务指令文本。
    include_previous_action : bool
        是否把上一窗口的动作串带进 prompt。带上等于给模型动作历史；
        本项目此前的零样本探测显示无动作历史时模型完全无法对齐真值。
    loaded_images : list of PIL.Image.Image or None
        已在内存中的观测帧，时间升序。给定时直接使用并忽略 ``image_paths``——
        流式路径从 LMDB 解码出帧，不经过磁盘。

    Returns
    -------
    dict
        ``{"messages": [...]}``，user 内容里图像在前、文本在后。

    Raises
    ------
    FileNotFoundError
        ``image_paths`` 指向的帧文件不存在。
    """
    content: list[dict[str, Any]] = []
    if loaded_images is not None:
        for image in loaded_images:
            content.append({"type": "image", "image": image})
    else:
        for relative in sample["image_paths"]:
            path = dataset_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"观测帧缺失：{path}")
            content.append({"type": "image", "image": Image.open(path).convert("RGB")})

    text = instruction
    previous = sample.get("previous_action_text") or ""
    if include_previous_action and previous:
        text = f"{instruction}\nPrevious 200 ms executed: {previous}"
    content.append({"type": "text", "text": text})

    return {
        "messages": [
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["action_text"]}],
            },
        ],
    }


def load_lumine_conversations(
    dataset_directory: Path,
    subset: SubsetName = "train",
    instruction: str = DEFAULT_INSTRUCTION,
    include_previous_action: bool = True,
    maximum_samples: int | None = None,
) -> list[dict[str, list[dict[str, Any]]]]:
    """读取 Lumine 预训练目录的一个子集，产出 SFTTrainer 可直接消费的对话列表。

    Parameters
    ----------
    dataset_directory : Path
        ``build_pretraining_dataset`` 的输出目录，需含 ``samples_<子集>.jsonl``。
    subset : {"train", "validation"}
        要读取的子集。文件名与 ``build_pretraining_dataset`` 的产物一致。
    instruction : str
        任务指令文本。
    include_previous_action : bool
        是否在 prompt 中带上一窗口动作。
    maximum_samples : int or None
        最多读取的样本数，None 表示全部。

    Returns
    -------
    list of dict
        每项形如 ``{"messages": [...]}``。

    Raises
    ------
    FileNotFoundError
        目录下没有该子集的样本文件。
    ValueError
        样本里没有观测帧——视觉 SFT 必须有图像，需用带 ``image`` 模态的数据重新构建。
    """
    samples_path = Path(dataset_directory) / f"samples_{subset}.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"找不到 {samples_path}；先跑 bc_datasets.minestudio.lumine_pretraining_dataset",
        )
    conversations: list[dict[str, list[dict[str, Any]]]] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not sample["image_paths"]:
                raise ValueError(
                    "样本没有观测帧：视觉 SFT 需要图像。请下载 image 模态后重建数据"
                    "（不要加 --no-images）",
                )
            conversations.append(
                build_conversation(
                    sample,
                    samples_path.parent,
                    instruction=instruction,
                    include_previous_action=include_previous_action,
                ),
            )
            if maximum_samples is not None and len(conversations) >= maximum_samples:
                break
    if not conversations:
        raise ValueError(f"{samples_path} 中没有任何样本")
    return conversations
