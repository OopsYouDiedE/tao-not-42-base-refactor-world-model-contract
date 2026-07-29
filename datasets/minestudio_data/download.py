"""从 Hugging Face 下载 MineStudio 数据集。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

DATASET_REPOSITORIES = {
    name: f"CraftJarvis/minestudio-data-{name}-v110"
    for name in ("6xx", "7xx", "8xx", "9xx", "10xx")
}
MODALITY_NAMES = ("image", "action", "meta_info", "event", "segmentation", "motion")


def download_datasets(
    datasets: list[str],
    modalities: list[str],
    output_directory: Path,
    maximum_workers: int = 4,
    token: str | None = None,
    clean: bool = False,
    maximum_image_parts: int | None = None,
) -> dict[str, Path]:
    """默认增量下载；clean=True 时清除目标目录后重新下载。"""
    output_directory = Path(output_directory)
    results = {}
    for dataset in datasets:
        repository = DATASET_REPOSITORIES[dataset]
        target = output_directory / repository.rsplit("/", 1)[-1]
        if clean and target.exists():
            resolved = target.resolve()
            if resolved.parent != output_directory.resolve() or not resolved.name.startswith("minestudio-data-"):
                raise ValueError(f"拒绝清除非数据集目录：{resolved}")
            shutil.rmtree(resolved)
        allow_patterns = [f"{modality}/*" for modality in modalities if modality != "image"]
        if "image" in modalities and maximum_image_parts is not None:
            if maximum_image_parts < 1:
                raise ValueError("maximum_image_parts 必须大于零")
            files = list_repo_files(repository, repo_type="dataset", token=token)
            image_parts = sorted({
                "/".join(path.split("/")[:2])
                for path in files if path.startswith("image/part-")
            })[:maximum_image_parts]
            if not image_parts:
                raise RuntimeError(f"{repository} 没有可下载的 image 分片")
            allow_patterns.extend(f"{part}/*" for part in image_parts)
        elif "image" in modalities:
            allow_patterns.append("image/*")
        snapshot_download(
            repo_id=repository,
            repo_type="dataset",
            local_dir=target,
            allow_patterns=allow_patterns,
            max_workers=maximum_workers,
            token=token,
        )
        results[dataset] = target
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 MineStudio 数据集")
    parser.add_argument("--dataset", nargs="+", default=["10xx"], choices=DATASET_REPOSITORIES)
    parser.add_argument(
        "--modality", dest="modalities", nargs="+", choices=MODALITY_NAMES,
        help="下载指定模态；不设置时下载全部模态",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/datasets"))
    parser.add_argument("--maximum-workers", type=int, default=4)
    parser.add_argument("--token")
    parser.add_argument(
        "--maximum-image-parts", type=int,
        help="只下载排序后的前 N 个 image 编码分片；其他指定模态仍完整下载",
    )
    parser.add_argument("--clean", action="store_true", help="清除目标目录后重新下载")
    arguments = parser.parse_args()

    results = download_datasets(
        arguments.dataset,
        arguments.modalities or list(MODALITY_NAMES),
        arguments.output_dir,
        arguments.maximum_workers,
        arguments.token,
        arguments.clean,
        arguments.maximum_image_parts,
    )
    for dataset, directory in results.items():
        print(f"{dataset} -> {directory}")


if __name__ == "__main__":
    main()
