"""MineStudio 数据集的批量下载。

对外接口：
    DATASET_REPOSITORIES — 数据集短名（6xx…10xx）→ HuggingFace 仓库名。
    MODAL_NAMES — 可下载的模态目录名。
    list_available_parts — 列出仓库中某模态的全部分片号。
    download_datasets — 批量下载指定数据集 × 模态 × 分片到本地目录。
    main — 命令行入口。

MineStudio v1.1.0 的数据按模态解耦存放，仓库内布局为
``<modal>/part-<编号>/{data.mdb,lock.mdb}``（``event`` 无分片，``motion`` 多一层
``motion/motion-<编号>/``）。单个 image 分片可达 29GB，因此下载按分片粒度控制。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# 短名 → 仓库名。OpenAI VPT contractor data 转成 MineStudio 轨迹结构后的发布版本。
DATASET_REPOSITORIES: dict[str, str] = {
    "6xx": "CraftJarvis/minestudio-data-6xx-v110",
    "7xx": "CraftJarvis/minestudio-data-7xx-v110",
    "8xx": "CraftJarvis/minestudio-data-8xx-v110",
    "9xx": "CraftJarvis/minestudio-data-9xx-v110",
    "10xx": "CraftJarvis/minestudio-data-10xx-v110",
}

MODAL_NAMES: tuple[str, ...] = ("image", "action", "meta_info", "event", "segmentation", "motion")

# 无分片结构的模态：整个模态只有一个 LMDB。
_UNPARTITIONED_MODALS: frozenset[str] = frozenset({"event"})

_PART_PATTERN = re.compile(r"/(?:part|motion)-(\d+)/")


def _repository_of(dataset: str) -> str:
    """把数据集短名解析为 HuggingFace 仓库名。"""
    try:
        return DATASET_REPOSITORIES[dataset]
    except KeyError:
        known = ", ".join(sorted(DATASET_REPOSITORIES))
        raise ValueError(f"未知数据集 {dataset!r}，可选：{known}") from None


def list_available_parts(dataset: str, modal: str, token: str | None = None) -> list[int]:
    """列出仓库中某模态的全部分片号，升序返回。

    Parameters
    ----------
    dataset : str
        数据集短名，例如 ``"10xx"``。
    modal : str
        模态目录名，取值见 ``MODAL_NAMES``。
    token : str or None
        HuggingFace token；公开数据集可为 None。

    Returns
    -------
    list of int
        升序分片号。无分片结构的模态（``event``）返回空列表。
    """
    if modal not in MODAL_NAMES:
        raise ValueError(f"未知模态 {modal!r}，可选：{', '.join(MODAL_NAMES)}")
    files = HfApi().list_repo_files(
        repo_id=_repository_of(dataset), repo_type="dataset", token=token,
    )
    parts: set[int] = set()
    for name in files:
        if not name.startswith(f"{modal}/"):
            continue
        matched = _PART_PATTERN.search(f"/{name}")
        if matched is not None:
            parts.add(int(matched.group(1)))
    return sorted(parts)


def _allow_patterns(
    dataset: str,
    modal: str,
    maximum_parts: int | None,
    parts: list[int] | None,
    token: str | None,
) -> list[str]:
    """构造某模态的 allow_patterns：无分片模态整下，分片模态按选择下。"""
    if modal in _UNPARTITIONED_MODALS:
        return [f"{modal}/*"]
    selected = parts if parts is not None else list_available_parts(dataset, modal, token=token)
    if not selected:
        return []
    if maximum_parts is not None:
        selected = selected[:maximum_parts]
    # motion 模态在仓库里多一层：motion/motion/motion-<编号>/。
    prefix = f"{modal}/motion/motion-" if modal == "motion" else f"{modal}/part-"
    return [f"{prefix}{number}/*" for number in selected]


def download_datasets(
    datasets: list[str],
    modals: list[str],
    output_directory: Path,
    maximum_parts: int | None = None,
    parts: list[int] | None = None,
    maximum_workers: int = 4,
    token: str | None = None,
) -> dict[str, Path]:
    """批量下载数据集 × 模态 × 分片，落到 MineStudio 期望的目录结构。

    Parameters
    ----------
    datasets : list of str
        数据集短名列表，例如 ``["6xx", "10xx"]``。
    modals : list of str
        模态列表，例如 ``["image", "action", "meta_info"]``。
    output_directory : Path
        输出根目录；每个数据集落在 ``<root>/minestudio-data-<短名>-v110/``。
    maximum_parts : int or None
        每个模态最多下载的分片数（按分片号升序取前 N）。None 表示全下。
    parts : list of int or None
        显式指定分片号；给定时忽略 ``maximum_parts`` 的筛选来源。
    maximum_workers : int
        并行下载线程数。
    token : str or None
        HuggingFace token。

    Returns
    -------
    dict of str to Path
        数据集短名 → 本地目录。

    Notes
    -----
    单个 ``image`` 分片可达 29GB，跑全量前先用 ``maximum_parts`` 估盘。
    """
    if not datasets:
        raise ValueError("datasets 不能为空")
    if not modals:
        raise ValueError("modals 不能为空")
    results: dict[str, Path] = {}
    for dataset in datasets:
        repository = _repository_of(dataset)
        patterns: list[str] = []
        for modal in modals:
            patterns.extend(_allow_patterns(dataset, modal, maximum_parts, parts, token))
        if not patterns:
            raise ValueError(f"数据集 {dataset} 在所选模态下没有可下载文件")
        local_directory = output_directory / repository.split("/")[-1]
        snapshot_download(
            repo_id=repository,
            repo_type="dataset",
            local_dir=str(local_directory),
            allow_patterns=patterns,
            max_workers=maximum_workers,
            token=token,
        )
        results[dataset] = local_directory
    return results


def main() -> None:
    """命令行入口：批量下载 MineStudio LMDB 分片。"""
    parser = argparse.ArgumentParser(description="批量下载 MineStudio 数据集")
    parser.add_argument(
        "--dataset", nargs="+", default=["10xx"], choices=sorted(DATASET_REPOSITORIES),
        help="数据集短名，可多选",
    )
    parser.add_argument(
        "--modal", nargs="+", default=["image", "action", "meta_info"], choices=MODAL_NAMES,
        help="要下载的模态",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/minestudio"), help="输出根目录",
    )
    parser.add_argument(
        "--maximum-parts", type=int, default=None,
        help="每个模态最多下载的分片数；单个 image 分片可达 29GB",
    )
    parser.add_argument("--parts", type=int, nargs="+", default=None, help="显式指定分片号")
    parser.add_argument("--maximum-workers", type=int, default=4, help="并行下载线程数")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument(
        "--list-parts", action="store_true", help="只列出各模态分片号，不下载",
    )
    arguments = parser.parse_args()

    if arguments.list_parts:
        for dataset in arguments.dataset:
            for modal in arguments.modal:
                available = list_available_parts(dataset, modal, token=arguments.token)
                print(f"{dataset} {modal}: {available}")
        return

    results = download_datasets(
        datasets=arguments.dataset,
        modals=arguments.modal,
        output_directory=arguments.output_dir,
        maximum_parts=arguments.maximum_parts,
        parts=arguments.parts,
        maximum_workers=arguments.maximum_workers,
        token=arguments.token,
    )
    for dataset, directory in results.items():
        print(f"{dataset} -> {directory}")


if __name__ == "__main__":
    main()
