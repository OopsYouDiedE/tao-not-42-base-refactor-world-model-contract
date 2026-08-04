"""Unsloth 视觉模型加载与行为克隆训练配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoRASettings:
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    seed: int = 3407


@dataclass(frozen=True)
class SFTSettings:
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    epochs: float = 1.0
    max_steps: int | None = None
    max_sequence_length: int = 2048


def _chat_template(model: str) -> str | None:
    lowered = model.lower()
    if "gemma-4" in lowered:
        return "gemma-4-thinking" if any(name in lowered for name in ("26b", "31b")) else "gemma-4"
    if "qwen" in lowered:
        return None
    raise ValueError(f"unsupported vision model family: {model!r}")


def load_vision_model(
    model: str,
    *,
    adapter: str | None = None,
    lora: LoRASettings | None = None,
    load_in_4bit: bool = False,
    max_sequence_length: int = 2048,
) -> tuple[Any, Any]:
    try:
        import unsloth  # noqa: F401
        from unsloth import FastVisionModel, get_chat_template
    except ImportError as error:
        raise RuntimeError("GPU training requires unsloth") from error
    settings = lora or LoRASettings()
    base_name = model
    if adapter:
        from peft import PeftConfig

        base_name = str(PeftConfig.from_pretrained(adapter).base_model_name_or_path)
    loaded, processor = FastVisionModel.from_pretrained(
        adapter or model,
        max_seq_length=max_sequence_length,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
        use_exact_model_name=True,
    )
    if adapter is None:
        loaded = FastVisionModel.get_peft_model(
            loaded,
            finetune_vision_layers=settings.finetune_vision_layers,
            finetune_language_layers=settings.finetune_language_layers,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=settings.dropout,
            bias="none",
            random_state=settings.seed,
            target_modules="all-linear",
        )
    template = _chat_template(base_name)
    return loaded, processor if template is None else get_chat_template(processor, template)
