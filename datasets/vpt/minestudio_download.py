"""按课程阶段下载必需动作库、可选元数据库和图像 LMDB 分片。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from huggingface_hub import HfApi, snapshot_download

from datasets.vpt.minestudio_curriculum import (
    curriculum_stage_names,
    get_curriculum_stage,
)


MINESTUDIO_MODALITIES = (
    "action", "meta_info", "image", "event", "motion", "segmentation",
)


@dataclass(frozen=True)
class MineStudioShardSelection:
    """一次阶段下载所选择的文件和图像分片。"""

    image_shard: str
    image_shard_count: int
    allow_patterns: tuple[str, ...]


@dataclass(frozen=True)
class MineStudioDownloadSelection:
    """一次可独立执行的数据模态与图像分片选择。"""

    modalities: tuple[str, ...]
    image_shards: tuple[str, ...]
    image_shard_count: int
    allow_patterns: tuple[str, ...]


def _shard_sort_key(path: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", PurePosixPath(path).name)
    return (int(match.group(1)) if match else 2**31 - 1, path)


def image_shards_from_repository_files(repository_files: list[str]) -> tuple[str, ...]:
    """从 Hugging Face 文件列表提取稳定排序的图像 LMDB 目录。"""
    return tuple(sorted({
        "/".join(PurePosixPath(path).parts[:2])
        for path in repository_files
        if len(PurePosixPath(path).parts) >= 3
        and PurePosixPath(path).parts[0] == "image"
        and PurePosixPath(path).parts[-1] == "data.mdb"
    }, key=_shard_sort_key))


def list_stage_image_shards(
    stage_name: str,
    api: HfApi | None = None,
    revision: str | None = None,
) -> tuple[str, ...]:
    """查询一个课程阶段公开的全部图像 LMDB 分片。"""
    stage = get_curriculum_stage(stage_name)
    repository_files = (api or HfApi()).list_repo_files(
        repo_id=stage.repository_id,
        repo_type="dataset",
        revision=revision,
    )
    image_shards = image_shards_from_repository_files(repository_files)
    if not image_shards:
        raise RuntimeError(f"{stage.repository_id} 中没有 image/*/data.mdb 分片")
    return image_shards


def select_stage_modalities(
    repository_files: list[str],
    modalities: tuple[str, ...],
    image_shard_indices: tuple[int, ...] = (),
    all_image_shards: bool = False,
) -> MineStudioDownloadSelection:
    """选择完整非图像模态，以及零个、多个或全部图像 LMDB 分片。"""
    if not modalities:
        raise ValueError("modalities 不能为空")
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
        raise RuntimeError(f"远端 revision 缺少模态: {', '.join(sorted(missing))}")
    available_image_shards = image_shards_from_repository_files(repository_files)
    if "image" not in canonical_modalities:
        if image_shard_indices or all_image_shards:
            raise ValueError("只有选择 image 模态时才能指定图像分片")
        selected_image_shards: tuple[str, ...] = ()
    elif all_image_shards:
        if image_shard_indices:
            raise ValueError("all-image-shards 不能与 image-shard-index 同时使用")
        if not available_image_shards:
            raise RuntimeError("仓库中没有 image/*/data.mdb 分片")
        selected_image_shards = available_image_shards
    else:
        if not image_shard_indices:
            raise ValueError("选择 image 时必须指定 image-shard-index 或 all-image-shards")
        if not available_image_shards:
            raise RuntimeError("仓库中没有 image/*/data.mdb 分片")
        if len(set(image_shard_indices)) != len(image_shard_indices):
            raise ValueError("image-shard-index 不能重复")
        for index in image_shard_indices:
            if not 0 <= index < len(available_image_shards):
                raise ValueError(
                    f"图像分片索引 {index} 越界；有效范围为 "
                    f"0..{len(available_image_shards) - 1}",
                )
        selected_image_shards = tuple(
            available_image_shards[index] for index in image_shard_indices
        )
    allow_patterns = tuple(
        f"{modality}/**"
        for modality in canonical_modalities
        if modality != "image"
    ) + tuple(f"{image_shard}/**" for image_shard in selected_image_shards)
    return MineStudioDownloadSelection(
        modalities=canonical_modalities,
        image_shards=selected_image_shards,
        image_shard_count=len(available_image_shards),
        allow_patterns=allow_patterns,
    )


def select_stage_shard(
    repository_files: list[str],
    image_shard_index: int,
    include_metadata_targets: bool = False,
) -> MineStudioShardSelection:
    """选择全部动作文件、可选元数据文件和一个图像 LMDB。"""
    modalities = (
        ("action", "meta_info", "image")
        if include_metadata_targets else ("action", "image")
    )
    selection = select_stage_modalities(
        repository_files,
        modalities=modalities,
        image_shard_indices=(image_shard_index,),
    )
    image_shard = selection.image_shards[0]
    return MineStudioShardSelection(
        image_shard=image_shard,
        image_shard_count=selection.image_shard_count,
        allow_patterns=selection.allow_patterns,
    )


def prune_unselected_image_shards(
    destination: Path,
    selected_image_shards: tuple[str, ...],
) -> list[str]:
    """只删除目标阶段目录内、已经完整发布且未选中的图像 LMDB。"""
    image_root = (destination.resolve() / "image").resolve()
    if not image_root.is_dir():
        return []
    selected_names = {
        PurePosixPath(image_shard).name for image_shard in selected_image_shards
    }
    removed = []
    for child in sorted(image_root.iterdir()):
        resolved = child.resolve()
        if (
            child.is_dir()
            and resolved.parent == image_root
            and child.name not in selected_names
            and (child / "data.mdb").is_file()
        ):
            shutil.rmtree(resolved)
            removed.append(child.name)
    return removed


def local_complete_image_shards(
    data_root: str | Path,
    stage_name: str,
) -> tuple[str, ...]:
    """返回指定数据盘中已完整发布的阶段图像 LMDB。"""
    stage = get_curriculum_stage(stage_name)
    image_root = Path(data_root).resolve() / stage.dataset_group / "image"
    if not image_root.is_dir():
        return ()
    return tuple(
        f"image/{path.name}"
        for path in sorted(image_root.iterdir(), key=lambda path: _shard_sort_key(path.name))
        if path.is_dir() and (path / "data.mdb").is_file()
    )


def prune_other_image_shards(destination: Path, selected_image_shard: str) -> list[str]:
    """保留一个图像 LMDB 的兼容入口。"""
    return prune_unselected_image_shards(destination, (selected_image_shard,))


def prune_completed_stage(data_root: str | Path, stage_name: str) -> bool:
    """删除已完成阶段的本地副本，但不触及其他阶段或 checkpoint。"""
    root = Path(data_root).resolve()
    stage = get_curriculum_stage(stage_name)
    target = (root / stage.dataset_group).resolve()
    if target.parent != root:
        raise RuntimeError("课程阶段目录逃逸 data_root，拒绝删除")
    if not target.exists():
        return False
    if not target.is_dir():
        raise RuntimeError(f"课程阶段路径不是目录: {target}")
    shutil.rmtree(target)
    return True


def prepare_stage_shard(
    stage_name: str,
    data_root: str | Path,
    image_shard_index: int,
    maximum_workers: int = 4,
    replace_image_shards: bool = False,
    include_metadata_targets: bool = False,
) -> tuple[Path, MineStudioShardSelection]:
    """下载阶段动作库、可选元数据库和一个图像分片，支持断点续传。"""
    modalities = (
        ("action", "meta_info", "image")
        if include_metadata_targets else ("action", "image")
    )
    destination, download_selection = prepare_stage_modalities(
        stage_name=stage_name,
        data_root=data_root,
        modalities=modalities,
        image_shard_indices=(image_shard_index,),
        maximum_workers=maximum_workers,
        replace_image_shards=replace_image_shards,
    )
    return destination, MineStudioShardSelection(
        image_shard=download_selection.image_shards[0],
        image_shard_count=download_selection.image_shard_count,
        allow_patterns=download_selection.allow_patterns,
    )


def prepare_stage_modalities(
    stage_name: str,
    data_root: str | Path,
    modalities: tuple[str, ...],
    image_shard_indices: tuple[int, ...] = (),
    all_image_shards: bool = False,
    maximum_workers: int = 4,
    replace_image_shards: bool = False,
    revision: str | None = None,
) -> tuple[Path, MineStudioDownloadSelection]:
    """把所选完整模态和图像分片下载到指定数据盘，复用本地完整文件。"""
    if maximum_workers < 1:
        raise ValueError("maximum_workers 必须大于零")
    stage = get_curriculum_stage(stage_name)
    api = HfApi()
    repository_files = api.list_repo_files(
        repo_id=stage.repository_id,
        repo_type="dataset",
        revision=revision,
    )
    selection = select_stage_modalities(
        repository_files,
        modalities,
        image_shard_indices=image_shard_indices,
        all_image_shards=all_image_shards,
    )
    if replace_image_shards and not selection.image_shards:
        raise ValueError("replace-image-shards 要求本次选择至少一个 image 分片")
    destination = Path(data_root).resolve() / stage.dataset_group
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=stage.repository_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        allow_patterns=list(selection.allow_patterns),
        max_workers=maximum_workers,
    )
    if replace_image_shards:
        prune_unselected_image_shards(destination, selection.image_shards)
    return destination, selection


def main() -> None:
    """提供无需 CUDA、可按模态和图像分片重复执行的下载入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=curriculum_stage_names(), required=True)
    parser.add_argument("--data-root", default="runs/data/minestudio")
    parser.add_argument(
        "--modalities", nargs="+", choices=MINESTUDIO_MODALITIES,
        default=["action", "image"],
    )
    parser.add_argument("--image-shard-index", type=int, nargs="+")
    parser.add_argument("--all-image-shards", action="store_true")
    parser.add_argument("--revision", default=None, help="Hugging Face branch/tag/commit")
    parser.add_argument("--list-only", action="store_true", help="只解析并打印选择，不下载")
    parser.add_argument("--maximum-workers", type=int, default=4)
    parser.add_argument(
        "--replace-image-shards", action="store_true",
        help="新分片完整下载后删除本阶段旧图像分片；非图像模态不会删除",
    )
    arguments = parser.parse_args()
    image_shard_indices = (
        tuple(arguments.image_shard_index)
        if arguments.image_shard_index is not None
        else ((0,) if "image" in arguments.modalities and not arguments.all_image_shards else ())
    )
    if arguments.list_only:
        stage = get_curriculum_stage(arguments.stage)
        repository_files = HfApi().list_repo_files(
            repo_id=stage.repository_id,
            repo_type="dataset",
            revision=arguments.revision,
        )
        selection = select_stage_modalities(
            repository_files,
            tuple(arguments.modalities),
            image_shard_indices=image_shard_indices,
            all_image_shards=arguments.all_image_shards,
        )
        destination = Path(arguments.data_root).resolve() / stage.dataset_group
    else:
        destination, selection = prepare_stage_modalities(
            stage_name=arguments.stage,
            data_root=arguments.data_root,
            modalities=tuple(arguments.modalities),
            image_shard_indices=image_shard_indices,
            all_image_shards=arguments.all_image_shards,
            maximum_workers=arguments.maximum_workers,
            replace_image_shards=arguments.replace_image_shards,
            revision=arguments.revision,
        )
    print(json.dumps({
        "stage": arguments.stage,
        "modalities": list(selection.modalities),
        "image_shards": list(selection.image_shards),
        "image_shard_count": selection.image_shard_count,
        "revision": arguments.revision or "main",
        "destination": str(destination),
        "downloaded": not arguments.list_only,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
