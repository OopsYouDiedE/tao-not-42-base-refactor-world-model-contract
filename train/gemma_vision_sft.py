"""Gemma 4 视觉 SFT 入口：在 Lumine 动作数据上微调 Gemma 4 主干。

对外接口：
    main — 命令行入口。

用法::

    python -m train.gemma_vision_sft --model gemma-4-26B-A4B-it \
        --dataset-dir runs/datasets/lumine-10xx --output-dir runs/trains/sft-gemma

Gemma 4 的注意点：MoE 主干（26B-A4B）无官方 4bit 变体，走 bf16 LoRA；chat template 的
content 必须是列表而非裸字符串（本项目的对话构造已满足）。E2B/E4B 的训练 loss 落在
13–15 属正常，26B/31B 约 1–3，视觉任务约 3–5。
"""

from __future__ import annotations

from train.command_line import build_argument_parser, run_from_arguments
from train.unsloth_vision_sft import GEMMA_MODELS


def main() -> None:
    """解析命令行并跑 Gemma 4 视觉 SFT。"""
    parser = build_argument_parser(
        description="在 Lumine 动作数据上微调 Gemma 4 视觉主干",
        model_choices=GEMMA_MODELS,
        default_model="gemma-4-26B-A4B-it",
    )
    run_from_arguments(parser.parse_args())


if __name__ == "__main__":
    main()
