"""视觉行为克隆训练命令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import build_streaming_datasets, load_conversations
from .modeling import LoRASettings, SFTSettings, load_vision_model


def run_behavior_cloning(
    *,
    model: str,
    dataset: Path,
    output: Path,
    adapter: str | None = None,
    streaming: bool = False,
    lora: LoRASettings | None = None,
    training: SFTSettings | None = None,
    maximum_samples: int | None = None,
    load_in_4bit: bool = False,
) -> dict[str, Any]:
    settings = training or SFTSettings()
    loaded, processor = load_vision_model(
        model,
        adapter=adapter,
        lora=lora,
        load_in_4bit=load_in_4bit,
        max_sequence_length=settings.max_sequence_length,
    )
    if streaming:
        train_dataset, validation_dataset, dataset_stats = build_streaming_datasets(
            dataset, maximum_samples=maximum_samples
        )
    else:
        train_path = dataset / "train.jsonl" if dataset.is_dir() else dataset
        validation_path = dataset / "validation.jsonl" if dataset.is_dir() else None
        train_dataset = load_conversations(train_path, maximum_samples=maximum_samples)
        validation_dataset = (
            load_conversations(validation_path, maximum_samples=maximum_samples)
            if validation_path and validation_path.is_file()
            else None
        )
        dataset_stats = {
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset or ()),
            "protocol": "standard-input-action/v1",
        }
    try:
        from trl import SFTConfig, SFTTrainer
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as error:
        raise RuntimeError("behavior cloning requires trl and unsloth") from error
    configuration = SFTConfig(
        per_device_train_batch_size=settings.micro_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        output_dir=str(output),
        report_to="none",
        max_length=settings.max_sequence_length,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        **(
            {"max_steps": settings.max_steps}
            if settings.max_steps is not None
            else {"num_train_epochs": settings.epochs}
        ),
        **(
            {"eval_strategy": "epoch", "save_strategy": "epoch"}
            if validation_dataset is not None
            else {"save_strategy": "steps"}
        ),
    )
    trainer = SFTTrainer(
        model=loaded,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=configuration,
        data_collator=UnslothVisionDataCollator(loaded, processor),
    )
    metrics = trainer.train().metrics
    adapter_path = output / "lora_adapter"
    loaded.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    result = {
        "model": model,
        "source_adapter": adapter,
        "adapter": str(adapter_path),
        "dataset": str(dataset),
        "dataset_statistics": dataset_stats,
        "training_metrics": metrics,
        "protocol": "standard-input-action/v1",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="训练标准输入动作协议 v1 的视觉 BC LoRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    arguments = parser.parse_args()
    result = run_behavior_cloning(
        model=arguments.model,
        adapter=arguments.adapter,
        dataset=arguments.dataset,
        output=arguments.output,
        streaming=arguments.streaming,
        maximum_samples=arguments.maximum_samples,
        load_in_4bit=arguments.load_in_4bit,
        lora=LoRASettings(rank=arguments.lora_rank, alpha=arguments.lora_alpha),
        training=SFTSettings(
            micro_batch_size=arguments.micro_batch,
            gradient_accumulation_steps=arguments.gradient_accumulation,
            learning_rate=arguments.learning_rate,
            epochs=arguments.epochs,
            max_steps=arguments.max_steps,
            max_sequence_length=arguments.max_sequence_length,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
