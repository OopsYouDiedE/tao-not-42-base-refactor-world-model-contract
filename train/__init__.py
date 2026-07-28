"""Unsloth 视觉 SFT 训练入口。

对外接口：
    GEMMA_MODELS, QWEN_MODELS — 两族可用主干的模型名。
    LoraSettings, TrainingSettings — LoRA 与训练超参配置对象。
    DEFAULT_INSTRUCTION — 动作预测任务的默认指令文本。
    build_conversation, load_lumine_conversations — Lumine 落盘样本 → 对话格式数据集。
    StreamingSettings, LumineStreamingDataset, build_streaming_dataset —
        从 LMDB 流式产出对话样本，不落盘中间产物（训练默认路径）。
    resolve_worker_count — 按 CPU 核心数与可用内存推算数据加载 worker 数。
    run_supervised_finetuning — 通用视觉 SFT 训练循环。

两族主干共用同一套数据与训练流程，差别只在模型名与 chat template，因此
``gemma_vision_sft.py`` 与 ``qwen_vision_sft.py`` 只是两个命令行入口。

训练侧的名字（``GEMMA_MODELS`` 等）经 ``__getattr__`` 惰性转发：``import unsloth``
要装整套 CUDA 栈，而数据侧的流式加载与对话组装不需要。直接在本文件顶部导入会让
``from train.lumine_streaming_dataset import ...`` 也被迫拉起 unsloth，数据管线机器
与纯 CPU 单元测试就都跑不了。
"""

from typing import Any

from train.lumine_conversation_dataset import (
    DEFAULT_INSTRUCTION,
    build_conversation,
    load_lumine_conversations,
)
from train.lumine_streaming_dataset import (
    LumineStreamingDataset,
    StreamingSettings,
    build_streaming_dataset,
    resolve_worker_count,
)

# 名字 → 所在模块。取用时才导入，见上方说明。
_LAZY_EXPORTS = {
    "GEMMA_MODELS": "train.unsloth_supervised_finetuning",
    "QWEN_MODELS": "train.unsloth_supervised_finetuning",
    "LoraSettings": "train.unsloth_supervised_finetuning",
    "TrainingSettings": "train.unsloth_supervised_finetuning",
    "run_supervised_finetuning": "train.unsloth_supervised_finetuning",
}


def __getattr__(name: str) -> Any:
    """惰性解析训练侧名字，避免导入本包就要求 unsloth 可用。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


__all__ = [
    "DEFAULT_INSTRUCTION",
    "GEMMA_MODELS",
    "QWEN_MODELS",
    "LoraSettings",
    "LumineStreamingDataset",
    "StreamingSettings",
    "TrainingSettings",
    "build_conversation",
    "build_streaming_dataset",
    "load_lumine_conversations",
    "resolve_worker_count",
    "run_supervised_finetuning",
]
