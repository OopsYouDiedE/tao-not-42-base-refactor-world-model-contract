"""两族主干共用的训练命令行参数与入口构造。

对外接口：
    build_argument_parser — 按候选模型构造 parser。
    run_from_arguments — 解析结果 → 训练调用。

Gemma 与 Qwen 的入口只差候选模型与默认值，参数解析逻辑不重复实现。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_argument_parser(
    description: str,
    model_choices: dict[str, str],
    default_model: str,
) -> argparse.ArgumentParser:
    """构造训练命令行 parser。

    Parameters
    ----------
    description : str
        命令行帮助里的一句话说明。
    model_choices : dict of str to str
        可选模型短名 → HuggingFace 名。
    default_model : str
        默认模型短名。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model", default=default_model, choices=sorted(model_choices),
        help="主干模型短名",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="Lumine 预训练数据目录",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/trains/sft"),
        help="checkpoint 输出目录",
    )
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA 秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--freeze-vision", action="store_true", help="冻结视觉塔，只训语言层",
    )
    parser.add_argument(
        "--micro-batch", type=int, default=8,
        help="单卡 micro-batch；96GB 卡实测 8 为吞吐/显存拐点",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max-steps", type=int, default=None, help="最大步数，覆盖 epochs")
    parser.add_argument("--epochs", type=float, default=1.0, help="训练轮数")
    parser.add_argument(
        "--max-sequence-length", type=int, default=2048, help="最大 token 数，含图像 token",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="4bit 加载；MoE 主干不建议开，走 bf16",
    )
    parser.add_argument(
        "--no-previous-action", action="store_true", help="prompt 不带上一窗口动作",
    )
    parser.add_argument("--maximum-samples", type=int, default=None, help="最多使用的样本数")
    return parser


def run_from_arguments(arguments: argparse.Namespace) -> None:
    """按解析结果跑训练并打印统计。"""
    # unsloth 必须在 transformers 之前完成补丁，因此训练模块延迟到此处导入。
    from train.unsloth_supervised_finetuning import (
        LoraSettings,
        TrainingSettings,
        run_supervised_finetuning,
    )

    result = run_supervised_finetuning(
        model=arguments.model,
        dataset_directory=arguments.dataset_dir,
        output_directory=arguments.output_dir,
        lora=LoraSettings(
            rank=arguments.lora_rank,
            alpha=arguments.lora_alpha,
            finetune_vision_layers=not arguments.freeze_vision,
        ),
        training=TrainingSettings(
            micro_batch_size=arguments.micro_batch,
            gradient_accumulation_steps=arguments.gradient_accumulation,
            learning_rate=arguments.learning_rate,
            max_steps=arguments.max_steps,
            num_train_epochs=arguments.epochs,
            max_sequence_length=arguments.max_sequence_length,
        ),
        load_in_4bit=arguments.load_in_4bit,
        include_previous_action=not arguments.no_previous_action,
        maximum_samples=arguments.maximum_samples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
