"""MineStudio 数据集的批量下载与 Lumine 预训练数据预处理。

对外接口：
    DATASET_REPOSITORIES — 数据集短名 → HuggingFace 仓库名。
    download_datasets — 批量下载指定数据集与模态的 LMDB 分片。
    LMDBModalityReader — 单模态 LMDB 读取（image / action / meta_info）。
    TrajectoryReader — 多模态按 episode 对齐读取。
    encode_lumine_action — 一个感知窗口的动作 → Lumine run-length 动作串。
    decode_lumine_action — Lumine 动作串 → 逐 chunk 键集合与鼠标增量。
    build_pretraining_dataset — 批量产出 Lumine 格式预训练样本。
"""

from bc_datasets.minestudio.episode_split import (
    EpisodeIdentity,
    SplitResult,
    build_split,
    load_split,
    parse_episode_identity,
)
from bc_datasets.minestudio.huggingface_download import (
    DATASET_REPOSITORIES,
    download_datasets,
)
from bc_datasets.minestudio.lmdb_modality_reader import LMDBModalityReader, TrajectoryReader
from bc_datasets.minestudio.lumine_action_codec import (
    LumineActionChunk,
    LumineWindowAction,
    decode_lumine_action,
    encode_lumine_action,
)
from bc_datasets.minestudio.lumine_pretraining_dataset import build_pretraining_dataset

__all__ = [
    "DATASET_REPOSITORIES",
    "EpisodeIdentity",
    "LumineActionChunk",
    "LumineWindowAction",
    "LMDBModalityReader",
    "SplitResult",
    "TrajectoryReader",
    "build_pretraining_dataset",
    "build_split",
    "decode_lumine_action",
    "download_datasets",
    "encode_lumine_action",
    "load_split",
    "parse_episode_identity",
]
