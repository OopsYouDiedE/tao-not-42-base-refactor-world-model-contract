"""Unsloth 视觉 SFT 的共享训练流程（Gemma 与 Qwen 两族共用）。

对外接口：
    GEMMA_MODELS, QWEN_MODELS — 两族可用主干及其 chat template。
    LoraSettings — LoRA 注入配置。
    TrainingSettings — 训练超参。
    load_vision_model — 按模型名加载 FastVisionModel 并注入 LoRA。
    run_supervised_finetuning — 端到端训练并保存 adapter。

`import unsloth` 必须早于 transformers / trl 的重型导入，因此本模块把 unsloth 放在
文件顶部第一组 import——这是 unsloth 的补丁顺序要求，不是风格问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401  # 必须最先导入以完成对 transformers 的补丁
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

# Gemma 4 族：MoE 与稠密混编。26B-A4B 无官方 4bit 变体，MoE 建议走 bf16 LoRA。
GEMMA_MODELS: dict[str, str] = {
    "gemma-4-E2B-it": "unsloth/gemma-4-E2B-it",
    "gemma-4-E4B-it": "unsloth/gemma-4-E4B-it",
    "gemma-4-26B-A4B-it": "unsloth/gemma-4-26B-A4B-it",
    "gemma-4-31B-it": "unsloth/gemma-4-31B-it",
}

# Qwen3.6 族：原生多模态（Causal LM + Vision Encoder），思考模式是运行时开关。
QWEN_MODELS: dict[str, str] = {
    "Qwen3.6-27B": "unsloth/Qwen3.6-27B",
    "Qwen3.6-35B-A3B": "unsloth/Qwen3.6-35B-A3B",
}

# 两族的 chat template 名：Gemma 4 大模型走 thinking 模板，Qwen3.6 思考是运行时开关。
CHAT_TEMPLATES: dict[str, str] = {
    "gemma-4-E2B-it": "gemma-4",
    "gemma-4-E4B-it": "gemma-4",
    "gemma-4-26B-A4B-it": "gemma-4-thinking",
    "gemma-4-31B-it": "gemma-4-thinking",
    "Qwen3.6-27B": "qwen-3.6",
    "Qwen3.6-35B-A3B": "qwen-3.6",
}


@dataclass(frozen=True)
class LoraSettings:
    """LoRA 注入配置。

    Attributes
    ----------
    rank : int
        LoRA 秩。视觉任务用 32 起步；26B-A4B 上 r=16 约 1.88% 可训练参数。
    alpha : int
        LoRA alpha，一般与 rank 同量级。
    dropout : float
        LoRA dropout，0 表示不用。
    finetune_vision_layers : bool
        是否训练视觉塔。动作预测强依赖画面细节，默认开。
    finetune_language_layers : bool
        是否训练语言层。
    random_state : int
        LoRA 初始化随机种子。
    """

    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    random_state: int = 3407


@dataclass(frozen=True)
class TrainingSettings:
    """训练超参。

    Attributes
    ----------
    micro_batch_size : int
        单卡 micro-batch。本项目在 96GB Blackwell 上实测 8 为吞吐/显存拐点
        （峰值约 71GB，再往上收益枯竭且易 OOM）。
    gradient_accumulation_steps : int
        梯度累积步数。
    learning_rate : float
        学习率。短跑 2e-4，长跑降到 2e-5。
    max_steps : int or None
        最大步数；给定时覆盖 ``num_train_epochs``。
    num_train_epochs : float
        训练轮数，仅当 ``max_steps`` 为 None 时生效。
    max_sequence_length : int
        单样本最大 token 数，含图像 token。
    warmup_ratio : float
        warmup 占比。
    weight_decay : float
        权重衰减。
    max_gradient_norm : float
        梯度裁剪阈值。
    logging_steps : int
        日志间隔步数。
    save_steps : int
        checkpoint 间隔步数。
    seed : int
        训练随机种子。
    """

    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    max_steps: int | None = None
    num_train_epochs: float = 1.0
    max_sequence_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_gradient_norm: float = 0.3
    logging_steps: int = 1
    save_steps: int = 200
    seed: int = 3407


def resolve_model_name(model: str) -> tuple[str, str]:
    """把模型短名解析为 ``(HuggingFace 名, chat template 名)``。

    也接受完整 HuggingFace 名；此时按短名后缀匹配 chat template，匹配不到则报错，
    避免静默套用错模板（模板不一致是导出后效果变差的最常见原因）。
    """
    known = {**GEMMA_MODELS, **QWEN_MODELS}
    if model in known:
        return known[model], CHAT_TEMPLATES[model]
    for short_name, full_name in known.items():
        if model == full_name or model.endswith(short_name):
            return model, CHAT_TEMPLATES[short_name]
    raise ValueError(
        f"无法为 {model!r} 确定 chat template，可选短名：{', '.join(sorted(known))}",
    )


def load_vision_model(
    model: str,
    lora: LoraSettings,
    load_in_4bit: bool = False,
    max_sequence_length: int = 2048,
) -> tuple[Any, Any]:
    """加载视觉主干并注入 LoRA。

    Parameters
    ----------
    model : str
        模型短名或完整 HuggingFace 名。
    lora : LoraSettings
        LoRA 配置。
    load_in_4bit : bool
        是否 4bit 量化加载。MoE 主干（26B-A4B / 35B-A3B）不建议开，走 bf16。
    max_sequence_length : int
        最大序列长度，单位 token。

    Returns
    -------
    tuple
        ``(model, processor)``。

    Notes
    -----
    传完整本地快照路径时会带上 ``use_exact_model_name=True``：unsloth 默认会把模型名
    规范化成小写去找缓存，遇到大写目录名会判定缺分片并重新下载整个模型。
    """
    model_name, chat_template = resolve_model_name(model)
    loaded, processor = FastVisionModel.from_pretrained(
        model_name,
        max_seq_length=max_sequence_length,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
        use_exact_model_name=True,
    )
    loaded = FastVisionModel.get_peft_model(
        loaded,
        finetune_vision_layers=lora.finetune_vision_layers,
        finetune_language_layers=lora.finetune_language_layers,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias="none",
        random_state=lora.random_state,
        target_modules="all-linear",
    )
    from unsloth import get_chat_template

    processor = get_chat_template(processor, chat_template)
    return loaded, processor


def run_supervised_finetuning(
    model: str,
    dataset_directory: Path,
    output_directory: Path,
    lora: LoraSettings | None = None,
    training: TrainingSettings | None = None,
    load_in_4bit: bool = False,
    include_previous_action: bool = True,
    maximum_samples: int | None = None,
    subset: str = "train",
) -> dict[str, Any]:
    """端到端跑一次 Lumine 动作预测的视觉 SFT。

    Parameters
    ----------
    model : str
        模型短名或完整 HuggingFace 名。
    dataset_directory : Path
        Lumine 预训练数据目录（``build_pretrain_dataset`` 的输出）。
    output_directory : Path
        checkpoint 与 LoRA adapter 的输出目录。
    lora : LoraSettings or None
        LoRA 配置，None 用默认。
    training : TrainingSettings or None
        训练超参，None 用默认。
    load_in_4bit : bool
        是否 4bit 加载。
    include_previous_action : bool
        prompt 是否带上一窗口动作。
    maximum_samples : int or None
        最多使用的样本数。
    subset : str
        用于训练的子集名，对应 ``samples_<子集>.jsonl``。默认 ``"train"``。

    Returns
    -------
    dict
        训练统计：``train_runtime``、``train_loss`` 等 TRL 原始字段，加 ``num_samples``。
    """
    from train.lumine_conversation_dataset import load_lumine_conversations

    lora_settings = lora if lora is not None else LoraSettings()
    training_settings = training if training is not None else TrainingSettings()

    loaded_model, processor = load_vision_model(
        model,
        lora_settings,
        load_in_4bit=load_in_4bit,
        max_sequence_length=training_settings.max_sequence_length,
    )
    conversations = load_lumine_conversations(
        dataset_directory,
        subset=subset,  # type: ignore[arg-type]
        include_previous_action=include_previous_action,
        maximum_samples=maximum_samples,
    )

    configuration = SFTConfig(
        per_device_train_batch_size=training_settings.micro_batch_size,
        gradient_accumulation_steps=training_settings.gradient_accumulation_steps,
        learning_rate=training_settings.learning_rate,
        warmup_ratio=training_settings.warmup_ratio,
        weight_decay=training_settings.weight_decay,
        max_grad_norm=training_settings.max_gradient_norm,
        logging_steps=training_settings.logging_steps,
        save_strategy="steps",
        save_steps=training_settings.save_steps,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=training_settings.seed,
        output_dir=str(output_directory),
        report_to="none",
        max_length=training_settings.max_sequence_length,
        # 视觉微调的四个硬性要求：collator 自己处理图文，不能让 TRL 再插手。
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        **(
            {"max_steps": training_settings.max_steps}
            if training_settings.max_steps is not None
            else {"num_train_epochs": training_settings.num_train_epochs}
        ),
    )
    trainer = SFTTrainer(
        model=loaded_model,
        train_dataset=conversations,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(loaded_model, processor),
        args=configuration,
    )
    statistics = trainer.train()

    output_directory.mkdir(parents=True, exist_ok=True)
    adapter_directory = output_directory / "lora_adapter"
    loaded_model.save_pretrained(str(adapter_directory))
    processor.save_pretrained(str(adapter_directory))

    result = dict(statistics.metrics)
    result["num_samples"] = len(conversations)
    result["adapter_directory"] = str(adapter_directory)
    return result
