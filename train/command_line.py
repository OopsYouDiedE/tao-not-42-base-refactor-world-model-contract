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
        "--dataset-dir", type=Path, required=True,
        help="流式模式（默认）：MineStudio 数据集根目录；--no-streaming 时：Lumine 落盘数据目录",
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
    parser.add_argument(
        "--subset", default="train", choices=("train", "validation"),
        help="用于训练的子集",
    )
    parser.add_argument(
        "--no-streaming", action="store_true",
        help="改读 lumine_pretraining_dataset 的落盘产物；默认直接从 LMDB 流式加载",
    )
    parser.add_argument(
        "--dataloader-workers", type=int, default=None,
        help="并行数据加载 worker 数；默认按 CPU 核心数与可用内存推算",
    )
    parser.add_argument(
        "--holdout-level", default="prefix", choices=("prefix", "episode"),
        help="prefix：整个玩家留出，衡量跨玩家泛化；episode：按 episode 打散",
    )
    parser.add_argument(
        "--validation-ratio", type=float, default=0.1, help="验证集目标帧数占比",
    )
    parser.add_argument("--split-seed", type=int, default=3407, help="episode 粒度打散种子")
    return parser


def run_from_arguments(arguments: argparse.Namespace) -> None:
    """按解析结果跑训练并打印统计。"""

    # unsloth 必须在 transformers 之前完成补丁，因此训练模块延迟到此处导入。
    from train.unsloth_vision_sft import (
        LoRASettings,
        SFTSettings,
        run_vision_sft,
    )

    result = run_vision_sft(
        model=arguments.model,
        dataset_directory=arguments.dataset_dir,
        output_directory=arguments.output_dir,
        lora=LoRASettings(
            rank=arguments.lora_rank,
            alpha=arguments.lora_alpha,
            finetune_vision_layers=not arguments.freeze_vision,
        ),
        training=SFTSettings(
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
        subset=arguments.subset,
        streaming=not arguments.no_streaming,
        dataloader_workers=arguments.dataloader_workers,
        holdout_level=arguments.holdout_level,
        validation_ratio=arguments.validation_ratio,
        split_seed=arguments.split_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
