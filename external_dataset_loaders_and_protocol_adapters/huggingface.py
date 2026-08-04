"""外部 Hugging Face 数据集下载。"""

from __future__ import annotations

from pathlib import Path


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
