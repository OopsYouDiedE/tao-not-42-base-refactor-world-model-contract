"""把现有轨迹 HDF5 升级为 action-first 提示词和理由协议。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from datasets.minestudio_finetune.sft_protocol import normalize_question, training_reason


def upgrade_archive(source_path: Path, output_path: Path) -> dict[str, int | str]:
    """复制存档，并规范化内嵌题面、意图、输出契约和行为理由。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source:
        if source.attrs.get("format") != "minestudio_trajectory_sft_v1":
            raise ValueError("输入文件格式不兼容")
        sample_count = int(source.attrs["sample_count"])
        with h5py.File(output_path, "w") as output:
            for name, value in source.attrs.items():
                output.attrs[name] = value
            output.attrs["response_protocol"] = "action_first_reason_optional_v1"
            samples = output.create_group("samples", track_order=True)
            for index, group in enumerate(source["samples"].values()):
                destination = samples.create_group(f"{index:08d}")
                for name, value in group.attrs.items():
                    if name not in {"question_json", "answer_json"}:
                        destination.attrs[name] = value
                question = normalize_question(json.loads(group.attrs["question_json"]))
                answer = json.loads(group.attrs["answer_json"])
                answer["answer_reason"] = training_reason(question, answer)
                destination.attrs["question_json"] = json.dumps(question, ensure_ascii=False)
                destination.attrs["answer_json"] = json.dumps(answer, ensure_ascii=False)
                image_group = destination.create_group("images", track_order=True)
                for image_name, image_data in group["images"].items():
                    source.copy(image_data, image_group, name=image_name)
    return {
        "format": "minestudio_trajectory_sft_v1",
        "response_protocol": "action_first_reason_optional_v1",
        "sample_count": sample_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="升级轨迹 HDF5 的 action-first 回答协议")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade_archive(args.source, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
