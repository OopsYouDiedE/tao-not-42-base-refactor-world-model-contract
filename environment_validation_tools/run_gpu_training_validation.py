"""编排真实视觉模型的 GPU 推理、BC 与相对优势训练验收。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from shared_tools import atomic_write_json, atomic_write_text

DEFAULT_MODEL = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"


def _run(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], check=True)


def _prepare_dataset(output: Path) -> tuple[Path, Path]:
    dataset = output / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    image_path = dataset / "tree.png"
    image = Image.new("RGB", (224, 224), (65, 130, 70))
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((70, 30, 150, 210), fill=(110, 75, 45))
    drawing.ellipse((88, 5, 132, 49), fill=(45, 155, 65))
    image.save(image_path)
    row = {
        "image_paths": [image_path.name],
        "prompt": (
            "Return one valid standard-input-action/v1 sequence that moves forward for one tick."
        ),
        "action_text": "Device KeyboardMouse\nTick 0\n<action>W</action>",
    }
    atomic_write_text(dataset / "train.jsonl", json.dumps(row, ensure_ascii=False) + "\n")
    return dataset, image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行真实 2B 视觉模型 GPU 训练全链路验收")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("runs/gpu_training_validation"))
    parser.add_argument("--bc-steps", type=int, default=1)
    parser.add_argument("--rlhf-epochs", type=int, default=1)
    arguments = parser.parse_args()
    output = arguments.output
    dataset, image = _prepare_dataset(output)
    bc_output = output / "bc"
    adapter = bc_output / "lora_adapter"
    _run(
        [
            "-m",
            "behavior_cloning_training.train",
            "--model",
            arguments.model,
            "--dataset",
            str(dataset),
            "--output",
            str(bc_output),
            "--maximum-samples",
            "1",
            "--micro-batch",
            "1",
            "--gradient-accumulation",
            "1",
            "--max-steps",
            str(arguments.bc_steps),
            "--max-sequence-length",
            "512",
            "--lora-rank",
            "4",
            "--lora-alpha",
            "4",
            "--load-in-4bit",
        ]
    )
    execution = output / "execution" / "execution.json"
    _run(
        [
            "-m",
            "environment_validation_tools.generate_gpu_validation_execution",
            "--model",
            arguments.model,
            "--adapter",
            str(adapter),
            "--image",
            str(image),
            "--intent",
            "Move forward for one tick.",
            "--output",
            str(execution),
            "--load-in-4bit",
        ]
    )
    rlhf_output = output / "rlhf"
    _run(
        [
            "-m",
            "relative_advantage_comparison_training.train_policy",
            "--model",
            arguments.model,
            "--adapter",
            str(adapter),
            "--execution",
            str(execution),
            "--intent",
            "Move forward for one tick.",
            "--output",
            str(rlhf_output),
            "--epochs",
            str(arguments.rlhf_epochs),
            "--load-in-4bit",
        ]
    )
    result = {
        "model": arguments.model,
        "device_validation": "real GPU model inference, BC update, and 2+6 policy update",
        "behavior_cloning": json.loads(
            (bc_output / "training_result.json").read_text(encoding="utf-8")
        ),
        "execution": str(execution),
        "relative_advantage_training": json.loads(
            (rlhf_output / "training_result.json").read_text(encoding="utf-8")
        ),
    }
    atomic_write_json(output / "validation_result.json", result)
    atomic_write_text(
        output / "REPORT.md",
        "# GPU 训练验收\n\n"
        f"- 模型：`{arguments.model}`\n"
        "- 路径：真实视觉推理、行为克隆更新、2+6 相对优势更新\n"
        f"- 结果：`{output / 'validation_result.json'}`\n",
    )
    print(output / "REPORT.md")


if __name__ == "__main__":
    main()
