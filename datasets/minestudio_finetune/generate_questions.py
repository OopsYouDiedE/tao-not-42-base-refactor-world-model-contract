"""从 MineStudio 真实轨迹机器生成三类动作训练题。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.action_codec import encode_lumine_action
from datasets.minestudio_data.load import TrajectoryReader
from datasets.minestudio_finetune.question_schema import (
    OUTPUT_CONTRACT,
    TASK_PROMPTS,
    TASK_TYPES,
    TaskType,
)

WINDOW_FRAMES = 4
HISTORY_OFFSETS = (12, 8, 4, 0)
OPTIMIZATION_WINDOWS = 4


def is_informative_action(actions: dict[str, np.ndarray]) -> bool:
    """动作窗口包含按键或可见相机移动时返回真。"""
    camera = np.asarray(actions["camera"], dtype=np.float64)
    if float(np.abs(camera).sum()) >= 0.30:
        return True
    return any(
        bool(np.asarray(values).astype(bool).any())
        for field, values in actions.items()
        if field != "camera"
    )


def source_frames(task_type: TaskType, start: int) -> list[int]:
    """返回题面使用的帧号；预测题不会取目标区间的未来帧。"""
    if task_type == "image_to_action":
        return [start]
    if task_type == "history_to_future_action":
        return [start - offset for offset in HISTORY_OFFSETS]
    return [start + offset for offset in range(0, OPTIMIZATION_WINDOWS * WINDOW_FRAMES, WINDOW_FRAMES)]


def _prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空：{output}；确认后使用 --overwrite")
        resolved = output.resolve()
        if resolved == Path(resolved.anchor) or resolved.name in {"runs", "datasets"}:
            raise ValueError(f"拒绝清理宽泛目录：{resolved}")
        shutil.rmtree(resolved)
    (output / "images").mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_dataset_readme(
    output: Path,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    structure_reviews: list[dict[str, Any]] | None = None,
) -> None:
    """把真实题图、题面、参考轨迹和结构审核结果写成浏览报告。"""
    answer_by_id = {record["id"]: record for record in answers}
    review_by_id = {record["id"]: record for record in (structure_reviews or [])}
    lines = [
        "# MineStudio 轨迹训练题生成报告", "",
        "> 本报告由出题流程自动生成。图片与参考动作来自真实 MineStudio 轨迹。",
        "> 参考轨迹是一种人类示范，不是唯一正确答案。`answer_key.jsonl` 不应交给做题模型。",
        "", "## 汇总", "", "| 项目 | 数量 |", "|---|---:|",
        f"| 候选题目 | {len(questions)} |",
        f"| 结构审核完成 | {len(review_by_id)} |",
        f"| 结构审核通过 | {sum(r.get('decision') == 'pass' for r in review_by_id.values())} |",
        "",
    ]
    for question in questions:
        sample_id = question["id"]
        answer = answer_by_id.get(sample_id, {})
        review = review_by_id.get(sample_id)
        lines.extend([
            f"## {sample_id}", "", "| 字段 | 内容 |", "|---|---|",
            f"| 题型 | `{question['task_type']}` |",
            f"| 来源 episode | `{question['source']['episode']}` |",
            f"| 图片帧 | `{question['source']['image_frames']}` |",
            f"| 目标动作区间 | `{question['target_interval']}` |",
            f"| 初始训练准入 | `{question['include_in_training']}` |",
            f"| 结构审核 | `{review['decision'] if review else 'pending'}` |",
            "", "### 图片", "",
        ])
        for index, image_path in enumerate(question["images"]):
            frame = question["source"]["image_frames"][index]
            lines.extend([
                f"**图 {index + 1}，帧 {frame}**", "",
                f"![{sample_id} frame {frame}]({image_path})", "",
            ])
        lines.extend(["### 问题", "", question["prompt"], ""])
        raw = question.get("inputs", {}).get("raw_action_sequence")
        if raw:
            lines.extend(["### 待优化的原始动作序列", ""])
            for index, block in enumerate(raw, 1):
                lines.extend([f"动作块 {index}：", "", "```text", block, "```", ""])
        lines.extend([
            "### 参考答案轨迹", "",
            "参考类型：真实人类演示；该轨迹不是唯一合理答案。", "",
        ])
        for index, block in enumerate(answer.get("reference_action_sequence", []), 1):
            lines.extend([f"动作块 {index}：", "", "```text", block, "```", ""])
        if review is not None:
            lines.extend([
                "### 结构校验结果", "", "```json",
                json.dumps(review, ensure_ascii=False, indent=2), "```", "",
            ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_images(
    reader: TrajectoryReader,
    output: Path,
    sample_id: str,
    episode: str,
    frames: list[int],
) -> list[str]:
    paths: list[str] = []
    for index, frame_index in enumerate(frames):
        relative = f"images/{sample_id}_{index:02d}.jpg"
        frame = reader.readers["image"].read_frames(episode, frame_index, 1)[0]
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
            output / relative, quality=95,
        )
        paths.append(relative)
    return paths


def _action_blocks(
    reader: TrajectoryReader,
    episode: str,
    start: int,
    count: int,
) -> list[str]:
    return [
        encode_lumine_action(
            reader.readers["action"].read_frames(
                episode, start + index * WINDOW_FRAMES, WINDOW_FRAMES,
            ),
        ).to_text()
        for index in range(count)
    ]


def build_question_record(
    sample_id: str,
    task_type: TaskType,
    episode: str,
    start: int,
    images: list[str],
    raw_actions: list[str] | None = None,
) -> dict[str, Any]:
    """构造公开题面。来源定位保留给审核，目标动作只进入独立答案文件。"""
    frames = source_frames(task_type, start)
    inputs: dict[str, Any] = {"images_chronological": True}
    if raw_actions is not None:
        inputs["raw_action_sequence"] = raw_actions
    return {
        "id": sample_id,
        "task_type": task_type,
        "prompt": TASK_PROMPTS[task_type],
        "images": images,
        "inputs": inputs,
        "output_contract": OUTPUT_CONTRACT,
        "source": {"episode": episode, "image_frames": frames},
        "target_interval": [
            start,
            start + (
                OPTIMIZATION_WINDOWS * WINDOW_FRAMES
                if task_type == "demonstration_optimization" else WINDOW_FRAMES
            ),
        ],
        "reference_is_unique": False,
        "review_status": "pending_human_and_ai_review",
        "include_in_training": False,
    }


def build_questions(
    dataset_directories: list[Path],
    output_directory: Path,
    samples_per_type: int = 100,
    seed: int = 20260729,
    frame_width: int = 320,
    frame_height: int = 180,
    overwrite: bool = False,
) -> dict[str, Any]:
    """生成题面、隔离的参考示范和待审核清单。"""
    if samples_per_type < 1:
        raise ValueError("samples_per_type 必须大于零")
    output = Path(output_directory)
    _prepare_output(output, overwrite)
    randomizer = random.Random(seed)
    reader = TrajectoryReader(
        dataset_directories, ["action", "image"], frame_width, frame_height,
    )
    questions: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    try:
        episodes = reader.episode_names()
        if not episodes:
            raise ValueError("action 与 image 模态没有共同 episode")
        for task_type in TASK_TYPES:
            generated = 0
            attempts = 0
            while generated < samples_per_type and attempts < samples_per_type * 500:
                attempts += 1
                episode = randomizer.choice(episodes)
                first = max(HISTORY_OFFSETS) if task_type == "history_to_future_action" else 0
                needed = (
                    OPTIMIZATION_WINDOWS * WINDOW_FRAMES
                    if task_type == "demonstration_optimization" else WINDOW_FRAMES
                )
                last = reader.episode_length(episode) - needed
                if last < first:
                    continue
                start = randomizer.randint(first, last)
                reference = _action_blocks(reader, episode, start, 1)
                if not is_informative_action(
                    reader.readers["action"].read_frames(episode, start, WINDOW_FRAMES),
                ):
                    continue
                raw_actions = None
                answer_actions = reference
                if task_type == "demonstration_optimization":
                    raw_actions = _action_blocks(reader, episode, start, OPTIMIZATION_WINDOWS)
                    answer_actions = raw_actions
                sample_id = f"{task_type}_{generated:06d}"
                frames = source_frames(task_type, start)
                images = _save_images(reader, output, sample_id, episode, frames)
                questions.append(build_question_record(
                    sample_id, task_type, episode, start, images, raw_actions,
                ))
                answers.append({
                    "id": sample_id,
                    "task_type": task_type,
                    "reference_action_sequence": answer_actions,
                    "reference_kind": "recorded_human_demonstration",
                    "reference_is_unique": False,
                    "source": {"episode": episode, "action_start_frame": start},
                })
                generated += 1
            if generated < samples_per_type:
                raise RuntimeError(f"{task_type} 在采样上限内只生成 {generated} 题")
    finally:
        reader.close()
    _write_jsonl(output / "questions.jsonl", questions)
    _write_jsonl(output / "answer_key.jsonl", answers)
    write_dataset_readme(output, questions, answers)
    manifest = {
        "format": "minestudio_trajectory_questions_v1",
        "task_types": list(TASK_TYPES),
        "samples_per_type": samples_per_type,
        "sample_count": len(questions),
        "seed": seed,
        "frame_size": [frame_width, frame_height],
        "questions": "questions.jsonl",
        "answer_key": "answer_key.jsonl",
        "readme": "README.md",
        "initial_review_status": "pending_human_and_ai_review",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="从 MineStudio 生成三类动作轨迹训练题")
    parser.add_argument("--dataset-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    manifest = build_questions(
        arguments.dataset_dir, arguments.output_dir, arguments.samples_per_type,
        arguments.seed, arguments.frame_width, arguments.frame_height, arguments.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
