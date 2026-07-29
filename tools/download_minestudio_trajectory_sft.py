"""从 Hugging Face 下载公开的 MineStudio 轨迹 SFT HDF5。"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPOSITORY = "unjustify/minestudio-trajectory-sft-237"
DEFAULT_FILENAME = "minestudio-trajectory-sft-237.h5"


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 MineStudio 轨迹 SFT HDF5")
    parser.add_argument("--repo-id", default=DEFAULT_REPOSITORY)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/datasets"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=arguments.repo_id,
        repo_type="dataset",
        filename=arguments.filename,
        revision=arguments.revision,
        local_dir=arguments.output_dir,
    )
    print(downloaded)


if __name__ == "__main__":
    main()
