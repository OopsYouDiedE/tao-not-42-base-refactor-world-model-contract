"""用视觉大模型(默认 Claude Sonnet)为 MineStudio 窗口生成动作的语言化理由标注。

对外接口：main（命令行入口）。

协议：从 MineStudio 数据集按 5fps 采样画面（原始 20fps，每 4 帧取 1 帧），把该组 4 帧的
真值动作聚合成一个 5fps 动作（二值键取 OR、相机取偏离中性最大分量）。把采样帧序列与其
真值聚合动作一起交给视觉大模型，让它**解释每个已发生动作在画面语境下为何合理**，产出
``Type/Key/Action/Explanation`` 结构化标注。真值动作是硬事实，模型只补语言理由，不改动作。
忽略抖动：相机微偏（|分量|≤1）与偶发键不算关键动作。输出 JSONL，每行一个窗口的标注。

设计取舍见 data_pipelines/annotations/AGENTS.md。API key 从环境变量 ANTHROPIC_API_KEY 读，
不硬编码（AGENTS §101）。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import torch
from PIL import Image

from data_pipelines.minestudio.groups import get_dataset_group
from data_pipelines.minestudio.dataset import MineStudioLMDBDataset
from net.action_token_codec import CAMERA_NEUTRAL_BIN, StructuredAction
from train.minecraft.action_supervision import (
    CAMERA_SCALE,
    DEGREES_PER_MOUSE_PIXEL,
    vpt_actions_to_structured,
)

SOURCE_FPS = 20
CAMERA_JITTER_BIN = 1  # |bin-中性| <= 此值视为抖动，不算关键相机动作


def aggregate_actions(frames: list[StructuredAction]) -> StructuredAction:
    """把一组连续帧聚合成一个采样步动作：键取 OR、相机取偏离中性最大分量。"""
    keys = {key: any(frame.keys[key] for frame in frames) for key in frames[0].keys}

    def dominant(values: list[int]) -> int:
        return max(values, key=lambda value: abs(value - CAMERA_NEUTRAL_BIN))

    return StructuredAction(
        camera_yaw_bin=dominant([frame.camera_yaw_bin for frame in frames]),
        camera_pitch_bin=dominant([frame.camera_pitch_bin for frame in frames]),
        keys=keys,
    )


def describe_action(action: StructuredAction) -> str:
    """把聚合动作压成人类可读串，抖动级相机不计入（供模型理解真值动作）。"""
    parts = [symbol for key, symbol in (
        ("forward", "forward"), ("back", "back"),
        ("left", "strafe-left"), ("right", "strafe-right"),
    ) if action.keys[key]]
    parts += [key for key in ("jump", "sneak", "sprint", "attack", "use",
                              "drop", "inventory") if action.keys[key]]
    parts += [f"hotbar-{index}" for index in range(1, 10)
              if action.keys[f"hotbar.{index}"]]
    yaw = action.camera_yaw_bin - CAMERA_NEUTRAL_BIN
    pitch = action.camera_pitch_bin - CAMERA_NEUTRAL_BIN
    camera_parts = []
    if abs(yaw) > CAMERA_JITTER_BIN:
        camera_parts.append(f"turn-{'right' if yaw > 0 else 'left'}")
    if abs(pitch) > CAMERA_JITTER_BIN:
        camera_parts.append(f"look-{'down' if pitch > 0 else 'up'}")
    parts += camera_parts
    return ", ".join(parts) if parts else "no-op (idle / jitter only)"


def sample_window(
    sample: dict,
    stride: int,
    sampled_steps: int,
) -> tuple[list[Image.Image], list[StructuredAction]]:
    """把一个数据窗口按 stride 降采样为(帧图, 聚合动作)序列。"""
    images = sample["img"]
    structured = vpt_actions_to_structured(sample["act_agg"])
    frames: list[Image.Image] = []
    actions: list[StructuredAction] = []
    for step in range(sampled_steps):
        raw = step * stride
        array = images[raw].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        frames.append(Image.fromarray(array))
        actions.append(aggregate_actions(structured[raw:raw + stride]))
    return frames, actions


def _image_block(image: Image.Image) -> dict:
    """把 PIL 图编码为 Anthropic 消息的 base64 image block。"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


_SYSTEM = (
    "You are an expert Minecraft action annotator. You are given a 5fps sequence of "
    "gameplay frames and, for each frame, the ground-truth aggregated controls the "
    "player actually executed. Explain WHY each already-executed action is reasonable "
    "given what is visible on screen. You are not guessing the action — it is fact. "
    "Ignore jitter: tiny camera nudges and incidental keys are not key actions. Ground "
    "every explanation in visible on-screen evidence (terrain, blocks, mobs, UI). "
    "Output the Type/Key/Action/Explanation format exactly as instructed."
)

_FORMAT_HINT = (
    "For each step output:\n"
    "Step N:\n"
    "Scene: <one line: what is visible>\n"
    "Type: keyboard|mouse\n"
    "Key: <e.g. w, space, right_click, camera>\n"
    "Action: press|hold|move\n"
    "Explanation: <why this action fits the scene; cite what is visible>\n"
    "(Repeat Type/Key/Action/Explanation per concurrent control; if only jitter "
    "remains after ignoring it, write a single block noting no key action.)"
)


def build_message_content(
    task_text: str,
    frames: list[Image.Image],
    actions: list[StructuredAction],
) -> list[dict]:
    """构造多模态 user 消息：任务 + 逐步(图像 + 真值动作) + 格式要求。"""
    content: list[dict] = [
        {"type": "text", "text": f"Task goal: {task_text}\n\n"
         f"Sampled at 5fps. {len(frames)} steps follow."},
    ]
    for step, (frame, action) in enumerate(zip(frames, actions)):
        content.append({"type": "text",
                        "text": f"--- Step {step} --- ground-truth action: "
                                f"{describe_action(action)}"})
        content.append(_image_block(frame))
    content.append({"type": "text", "text": _FORMAT_HINT})
    return content


def annotate_window(client, model: str, task_text: str,
                    frames: list[Image.Image],
                    actions: list[StructuredAction]) -> str:
    """调用视觉大模型为一个采样窗口产出标注文本。"""
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM,
        messages=[{"role": "user",
                   "content": build_message_content(task_text, frames, actions)}],
    )
    return "".join(block.text for block in response.content
                   if block.type == "text")


def main() -> None:
    """从数据集采样若干窗口，逐窗口调模型产出动作理由标注，写 JSONL。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", required=True,
                        help="已下载的 dataset-group 目录，例如 runs/data/minestudio/10xx")
    parser.add_argument("--dataset-group", default="10xx", choices=("7xx", "9xx", "10xx"))
    parser.add_argument("--model", default="claude-sonnet-5",
                        help="视觉大模型标识（Anthropic）")
    parser.add_argument("--windows", type=int, default=20, help="标注的窗口数")
    parser.add_argument("--sampled-steps", type=int, default=8,
                        help="每窗口的 5fps 步数")
    parser.add_argument("--stride", type=int, default=SOURCE_FPS // 5,
                        help="降采样步长（20fps→5fps 为 4）")
    parser.add_argument("--image-height", type=int, default=252)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--output", default="runs/annotations/minestudio_10xx.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.stride < 1 or arguments.sampled_steps < 1 or arguments.windows < 1:
        raise ValueError("stride、sampled-steps、windows 必须大于零")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("需要环境变量 ANTHROPIC_API_KEY 才能调用标注模型")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    group = get_dataset_group(arguments.dataset_group)
    window = arguments.stride * arguments.sampled_steps
    dataset = MineStudioLMDBDataset(
        data_directory=arguments.data_directory, sequence_length=window,
        image_size=(arguments.image_height, arguments.image_width),
        task_text=group.task_text,
        camera_max_degrees=CAMERA_SCALE * DEGREES_PER_MOUSE_PIXEL, split="all",
    )
    if len(dataset) == 0:
        raise RuntimeError("数据窗口为空，检查下载分片")
    generator = torch.Generator().manual_seed(arguments.seed)
    indices = torch.randperm(len(dataset), generator=generator)[
        :arguments.windows].tolist()
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for position, index in enumerate(indices):
            frames, actions = sample_window(
                dataset[index], arguments.stride, arguments.sampled_steps)
            annotation = annotate_window(
                client, arguments.model, group.task_text, frames, actions)
            record = {
                "window_index": int(index),
                "task": group.task_text,
                "fps": 5,
                "actions": [describe_action(action) for action in actions],
                "annotation": annotation,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"event": "annotated", "position": position,
                              "window_index": int(index)}, ensure_ascii=False),
                  flush=True)
    print(json.dumps({"event": "done", "windows": len(indices),
                      "output": str(output_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
