"""项目数据集运行目录合同。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

DatasetStage = Literal[
    "external_dataset", "protocol_adapted_external_dataset", "behavior_cloning_dataset"
]
_REPO_ID = re.compile(r"^[^/\s]+/[^/\s]+$")


def dataset_id_from_repo_id(repo_id: str) -> str:
    """将 Hugging Face 仓库路径转换为项目 dataset_id。"""
    if _REPO_ID.fullmatch(repo_id) is None:
        raise ValueError(f"无效的 Hugging Face 数据集仓库路径：{repo_id!r}")
    return repo_id.replace("/", "_")


def dataset_path(repo_id: str, stage: DatasetStage, *, runs_root: Path = Path("runs")) -> Path:
    """返回指定处理阶段的数据集本地目录。"""
    return runs_root / stage / dataset_id_from_repo_id(repo_id)
