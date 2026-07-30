"""合并 MineStudio 轨迹 HDF5，并为后续批次添加 ID 命名空间。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description="合并两个 minestudio_trajectory_sft_v1 文件")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--append", type=Path, required=True)
    parser.add_argument("--append-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.base, "r") as base, h5py.File(args.append, "r") as append:
        for archive in (base, append):
            if archive.attrs.get("format") != "minestudio_trajectory_sft_v1":
                raise ValueError("输入文件格式不兼容")
        with h5py.File(args.output, "w") as output:
            output.attrs["format"] = "minestudio_trajectory_sft_v1"
            samples = output.create_group("samples", track_order=True)
            seen: set[str] = set()
            index = 0
            for source, prefix in ((base, ""), (append, args.append_prefix)):
                for group in source["samples"].values():
                    old_id = str(group.attrs["id"])
                    sample_id = prefix + old_id
                    if sample_id in seen:
                        raise ValueError(f"合并后 ID 重复：{sample_id}")
                    seen.add(sample_id)
                    destination = samples.create_group(f"{index:08d}")
                    destination.attrs["id"] = sample_id
                    question = json.loads(group.attrs["question_json"])
                    answer = json.loads(group.attrs["answer_json"])
                    question["id"] = sample_id
                    answer["id"] = sample_id
                    if prefix:
                        question.setdefault("source", {})["dataset_batch"] = prefix.rstrip("_")
                    destination.attrs["question_json"] = json.dumps(question, ensure_ascii=False)
                    destination.attrs["answer_json"] = json.dumps(answer, ensure_ascii=False)
                    image_group = destination.create_group("images", track_order=True)
                    for image_name, image_data in group["images"].items():
                        source.copy(image_data, image_group, name=image_name)
                    index += 1
            output.attrs["sample_count"] = index
            output.attrs["merged_base_count"] = len(base["samples"])
            output.attrs["merged_append_count"] = len(append["samples"])
    print(json.dumps({"sample_count": index, "unique_ids": len(seen)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
