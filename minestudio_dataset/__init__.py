"""MineStudio 数据集的批量下载与 Lumine 预训练数据预处理。

对外接口：
    DATASET_REPOSITORIES — 数据集短名 → HuggingFace 仓库名。
    download_datasets — 批量下载指定数据集与模态的 LMDB 分片。
    ModalKernelReader — 单模态 LMDB 读取（image / action / meta_info）。
    TrajectoryReader — 多模态按 episode 对齐读取。
    encode_lumine_action — 一个感知窗口的动作 → Lumine run-length 动作串。
    decode_lumine_action — Lumine 动作串 → 逐 chunk 键集合与鼠标增量。
    build_pretrain_dataset — 批量产出 Lumine 格式预训练样本。
"""

from minestudio_dataset.huggingface_download import (
    DATASET_REPOSITORIES,
    download_datasets,
)
from minestudio_dataset.lmdb_modal_reader import ModalKernelReader, TrajectoryReader
from minestudio_dataset.lumine_action_codec import (
    LumineActionChunk,
    LumineWindowAction,
    decode_lumine_action,
    encode_lumine_action,
)
from minestudio_dataset.lumine_pretrain_builder import build_pretrain_dataset

__all__ = [
    "DATASET_REPOSITORIES",
    "LumineActionChunk",
    "LumineWindowAction",
    "ModalKernelReader",
    "TrajectoryReader",
    "build_pretrain_dataset",
    "decode_lumine_action",
    "download_datasets",
    "encode_lumine_action",
]
