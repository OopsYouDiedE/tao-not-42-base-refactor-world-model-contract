"""完整下载指定 MineStudio 数据范围与模态。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re

from huggingface_hub import HfApi, snapshot_download

from datasets.minestudio.groups import (
    MINESTUDIO_DATASET_GROUPS,
    get_dataset_group,
)


MINESTUDIO_MODALITIES = (
    "action", "meta_info", "image", "event", "motion", "segmentation",
)


@dataclass(frozen=True)
class MineStudioDownloadSelection:
    """一次完整数据下载的范围、模态和图像分片。"""

    dataset_group: str
    modalities: tuple[str, ...]
    image_shards: tuple[str, ...]
    allow_patterns: tuple[str, ...]


def _shard_sort_key(path: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", PurePosixPath(path).name)
    return (int(match.group(1)) if match else 2**31 - 1, path)


def image_shards_from_repository_files(repository_files: list[str]) -> tuple[str, ...]:
    """从 Hugging Face 文件列表提取稳定排序的完整图像 LMDB 目录。"""
    return tuple(sorted({
        "/".join(PurePosixPath(path).parts[:2])
        for path in repository_files
        if len(PurePosixPath(path).parts) >= 3
        and PurePosixPath(path).parts[0] == "image"
        and PurePosixPath(path).parts[-1] == "data.mdb"
    }, key=_shard_sort_key))


def select_complete_modalities(
    dataset_group: str,
    repository_files: list[str],
    modalities: tuple[str, ...],
    maximum_image_shards: int | None = None,
) -> MineStudioDownloadSelection:
    """选择完整模态；image 被选择时包含它的全部 LMDB 分片。

    Parameters
    ----------
    maximum_image_shards : int | None
        非空时只保留前若干个稳定排序的图像 LMDB 分片，用于快速实验或联调；
        默认 None 表示下载全部分片。动作与其它模态始终完整下载，因为它们体积小、
        且分片编号与图像不要求对齐（见 MineStudioLMDBDataset 的 episode 交集逻辑）。
    """
    if not modalities:
        raise ValueError("modalities 不能为空")
    if len(set(modalities)) != len(modalities):
        raise ValueError("modalities 不能重复")
    if maximum_image_shards is not None and maximum_image_shards < 1:
        raise ValueError("maximum_image_shards 必须大于零或为 None")
    unknown = set(modalities) - set(MINESTUDIO_MODALITIES)
    if unknown:
        raise ValueError(f"未知 MineStudio 模态: {', '.join(sorted(unknown))}")
    canonical_modalities = tuple(
        modality for modality in MINESTUDIO_MODALITIES if modality in modalities
    )
    available_modalities = {
        PurePosixPath(path).parts[0]
        for path in repository_files
        if PurePosixPath(path).parts
    }
    missing = set(canonical_modalities) - available_modalities
    if missing:
        raise RuntimeError(f"远端数据缺少模态: {', '.join(sorted(missing))}")
    image_shards = (
        image_shards_from_repository_files(repository_files)
        if "image" in canonical_modalities else ()
    )
    if "image" in canonical_modalities and not image_shards:
        raise RuntimeError("远端数据没有 image LMDB")
    if maximum_image_shards is not None:
        image_shards = image_shards[:maximum_image_shards]
    allow_patterns = tuple(
        f"{modality}/**"
        for modality in canonical_modalities
        if modality != "image"
    ) + tuple(f"{image_shard}/**" for image_shard in image_shards)
    return MineStudioDownloadSelection(
        dataset_group=dataset_group,
        modalities=canonical_modalities,
        image_shards=image_shards,
        allow_patterns=allow_patterns,
    )


def prepare_dataset_group(
    dataset_group: str,
    data_root: str | Path,
    modalities: tuple[str, ...] = ("image", "action"),
    maximum_workers: int = 8,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    maximum_image_shards: int | None = None,
) -> tuple[Path, MineStudioDownloadSelection]:
    """完整下载范围内选定模态，保留所有已经下载的数据。

    ``maximum_image_shards`` 非空时只下载前若干个图像 LMDB 分片，用于实验或联调；
    默认 None 表示完整下载。
    """
    if maximum_workers < 1:
        raise ValueError("maximum_workers 必须大于零")
    configuration = get_dataset_group(dataset_group)
    repository_files = HfApi().list_repo_files(
        repo_id=configuration.repository_id,
        repo_type="dataset",
        revision=revision,
    )
    selection = select_complete_modalities(
        dataset_group, repository_files, modalities, maximum_image_shards,
    )
    destination = Path(data_root).resolve() / dataset_group
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=configuration.repository_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        cache_dir=cache_directory,
        allow_patterns=list(selection.allow_patterns),
        max_workers=maximum_workers,
    )
    return destination, selection


def main() -> None:
    """提供完整 MineStudio 数据下载命令。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-group",
        choices=[configuration.dataset_group for configuration in MINESTUDIO_DATASET_GROUPS],
        default="10xx",
    )
    parser.add_argument("--data-root", default="runs/data/minestudio")
    parser.add_argument(
        "--modalities", nargs="+", choices=MINESTUDIO_MODALITIES,
        default=["image", "action"],
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--maximum-workers", type=int, default=8)
    parser.add_argument(
        "--max-image-shards", type=int, default=None,
        help="只下载前若干个图像 LMDB 分片供实验或联调；默认下载全部分片",
    )
    arguments = parser.parse_args()
    destination, selection = prepare_dataset_group(
        dataset_group=arguments.dataset_group,
        data_root=arguments.data_root,
        modalities=tuple(arguments.modalities),
        maximum_workers=arguments.maximum_workers,
        revision=arguments.revision,
        cache_directory=arguments.cache_directory,
        maximum_image_shards=arguments.max_image_shards,
    )
    print(json.dumps({
        "dataset_group": selection.dataset_group,
        "modalities": list(selection.modalities),
        "image_shards": list(selection.image_shards),
        "destination": str(destination),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
