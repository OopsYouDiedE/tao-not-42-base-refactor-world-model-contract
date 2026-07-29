"""从 Hugging Face 下载 MineStudio 数据集。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

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
        snapshot_download(
            repo_id=repository,
            repo_type="dataset",
            local_dir=target,
            allow_patterns=[f"{modality}/*" for modality in modalities],
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
    parser.add_argument("--clean", action="store_true", help="清除目标目录后重新下载")
    arguments = parser.parse_args()

    results = download_datasets(
        arguments.dataset,
        arguments.modalities or list(MODALITY_NAMES),
        arguments.output_dir,
        arguments.maximum_workers,
        arguments.token,
        arguments.clean,
    )
    for dataset, directory in results.items():
        print(f"{dataset} -> {directory}")


if __name__ == "__main__":
    main()
