"""按课程阶段全量下载动作/元数据库，并只轮换图像 LMDB 分片。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil

from huggingface_hub import HfApi, snapshot_download

from datasets.vpt.minestudio_curriculum import (
    curriculum_stage_names,
    get_curriculum_stage,
)


@dataclass(frozen=True)
class MineStudioShardSelection:
    """一次阶段下载所选择的文件和图像分片。"""

    image_shard: str
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


def list_stage_image_shards(stage_name: str, api: HfApi | None = None) -> tuple[str, ...]:
    """查询一个课程阶段公开的全部图像 LMDB 分片。"""
    stage = get_curriculum_stage(stage_name)
    repository_files = (api or HfApi()).list_repo_files(
        repo_id=stage.repository_id,
        repo_type="dataset",
    )
    image_shards = image_shards_from_repository_files(repository_files)
    if not image_shards:
        raise RuntimeError(f"{stage.repository_id} 中没有 image/*/data.mdb 分片")
    return image_shards


def select_stage_shard(
    repository_files: list[str],
    image_shard_index: int,
) -> MineStudioShardSelection:
    """选择全部动作/元数据文件和一个图像 LMDB，不假设分片编号对齐。"""
    image_shards = image_shards_from_repository_files(repository_files)
    if not image_shards:
        raise RuntimeError("仓库中没有 image/*/data.mdb 分片")
    if not 0 <= image_shard_index < len(image_shards):
        raise ValueError(
            f"图像分片索引 {image_shard_index} 越界；有效范围为 "
            f"0..{len(image_shards) - 1}",
        )
    image_shard = image_shards[image_shard_index]
    return MineStudioShardSelection(
        image_shard=image_shard,
        image_shard_count=len(image_shards),
        allow_patterns=("action/**", "meta_info/**", f"{image_shard}/**"),
    )


def prune_other_image_shards(destination: Path, selected_image_shard: str) -> list[str]:
    """只删除目标阶段目录内、已经完整发布的其他图像 LMDB。"""
    image_root = (destination.resolve() / "image").resolve()
    if not image_root.is_dir():
        return []
    selected_name = PurePosixPath(selected_image_shard).name
    removed = []
    for child in image_root.iterdir():
        resolved = child.resolve()
        if (
            child.is_dir()
            and resolved.parent == image_root
            and child.name != selected_name
            and (child / "data.mdb").is_file()
        ):
            shutil.rmtree(resolved)
            removed.append(child.name)
    return removed


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
) -> tuple[Path, MineStudioShardSelection]:
    """全量下载阶段动作/元数据库并下载一个图像分片，支持断点续传。"""
    if maximum_workers < 1:
        raise ValueError("maximum_workers 必须大于零")
    stage = get_curriculum_stage(stage_name)
    api = HfApi()
    repository_files = api.list_repo_files(
        repo_id=stage.repository_id,
        repo_type="dataset",
    )
    selection = select_stage_shard(repository_files, image_shard_index)
    destination = Path(data_root).resolve() / stage.dataset_group
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=stage.repository_id,
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=list(selection.allow_patterns),
        max_workers=maximum_workers,
    )
    if replace_image_shards:
        prune_other_image_shards(destination, selection.image_shard)
    return destination, selection


def main() -> None:
    """提供 AutoDL 可重复执行的阶段分片下载入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=curriculum_stage_names(), required=True)
    parser.add_argument("--data-root", default="runs/data/minestudio")
    parser.add_argument("--image-shard-index", type=int, default=0)
    parser.add_argument("--maximum-workers", type=int, default=4)
    parser.add_argument(
        "--replace-image-shards", action="store_true",
        help="新分片完整下载后删除本阶段旧图像分片；动作库不会删除",
    )
    arguments = parser.parse_args()
    destination, selection = prepare_stage_shard(
        stage_name=arguments.stage,
        data_root=arguments.data_root,
        image_shard_index=arguments.image_shard_index,
        maximum_workers=arguments.maximum_workers,
        replace_image_shards=arguments.replace_image_shards,
    )
    print(
        f"stage={arguments.stage} image_shard={selection.image_shard} "
        f"index={arguments.image_shard_index}/{selection.image_shard_count - 1} "
        f"destination={destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
