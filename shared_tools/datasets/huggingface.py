"""Hugging Face 数据集下载和显式发布边界。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPublishResult:
    repo_id: str
    source_directory: Path
    private: bool
    commit_url: str | None


def download_dataset_snapshot(
    repo_id: str,
    target: Path,
    *,
    revision: str | None = None,
    allow_patterns: tuple[str, ...] | None = None,
) -> Path:
    """使用 Hugging Face 官方客户端下载数据集快照。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("下载数据集需要 huggingface-hub") from error
    resolved = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return Path(resolved)


def publish_dataset(
    source_directory: Path,
    repo_id: str,
    *,
    private: bool,
    commit_message: str,
    confirm_publish: bool = False,
) -> DatasetPublishResult:
    """发布数据集目录；没有显式确认时拒绝产生外部写操作。"""
    if not confirm_publish:
        raise PermissionError("发布数据集必须显式传入 confirm_publish=True")
    if not source_directory.is_dir():
        raise FileNotFoundError(source_directory)
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("发布数据集需要 huggingface-hub") from error
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    result = api.upload_folder(
        folder_path=source_directory,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    return DatasetPublishResult(repo_id, source_directory.resolve(), private, str(result))
