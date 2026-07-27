"""Unsloth 视觉 SFT 训练入口。

对外接口：
    GEMMA_MODELS, QWEN_MODELS — 两族可用主干的模型名。
    LoraSettings, TrainingSettings — LoRA 与训练超参配置对象。
    load_lumine_conversations — Lumine 预训练样本 → 对话格式数据集。
    run_supervised_finetuning — 通用视觉 SFT 训练循环。

两族主干共用同一套数据与训练流程，差别只在模型名与 chat template，因此
``gemma_vision_sft.py`` 与 ``qwen_vision_sft.py`` 只是两个命令行入口。
"""

from train.lumine_conversation_dataset import (
    build_conversation,
    load_lumine_conversations,
)
from train.unsloth_supervised_finetuning import (
    GEMMA_MODELS,
    QWEN_MODELS,
    LoraSettings,
    TrainingSettings,
    run_supervised_finetuning,
)

__all__ = [
    "GEMMA_MODELS",
    "QWEN_MODELS",
    "LoraSettings",
    "TrainingSettings",
    "build_conversation",
    "load_lumine_conversations",
    "run_supervised_finetuning",
]
