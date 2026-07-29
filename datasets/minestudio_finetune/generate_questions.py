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
from typing import Literal, TypeAlias

TaskType: TypeAlias = Literal[
    "demonstration_optimization",
    "image_sequence_to_action",
    "history_to_future_action",
]
TASK_TYPES: tuple[TaskType, ...] = (
    "demonstration_optimization",
    "image_sequence_to_action",
    "history_to_future_action",
)
TASK_PROMPTS: dict[TaskType, str] = {
    "demonstration_optimization": (
        "The images and raw action blocks form one chronological Minecraft demonstration. "
        "Rewrite it as a cleaner action sequence while preserving visible intent and causal "
        "order. Return only a JSON array of valid action blocks."
    ),
    "image_sequence_to_action": (
        "The images are consecutive Minecraft observations in chronological order. Infer one "
        "reasonable action sequence that produced the transition. Return only a JSON array "
        "containing one valid action block."
    ),
    "history_to_future_action": (
        "The images are past Minecraft observations in chronological order. Infer one "
        "reasonable action sequence for the next 200 ms. Return only a JSON array containing "
        "one valid action block."
    ),
}
OUTPUT_CONTRACT = {
    "type": "json_array",
    "item": "variable-length named-token action block",
    "action_markers": ["<|action_start|>", "<|action_end|>"],
    "chunk_count": "variable; MineStudio references use four 50 ms ticks",
    "chunk_duration_ms": 50,
    "mouse": "Mouse dx dy moves the camera in gameplay and the cursor in GUI",
    "mixing_guidance": "Prefer standalone Mouse unless keys and mouse execute together",
}
REVIEW_DIMENSIONS = {
    "source_integrity": "Images and actions come from one episode in chronological order.",
    "visual_answerability": "The complete image sequence supports at least one reasonable answer.",
    "action_grounding": "The answer explains visible changes without unsupported actions.",
    "demonstration_quality": "The action is coherent and contains no isolated control noise.",
    "protocol_compliance": "The answer follows variable ticks and Mouse dx dy semantics.",
    "safety_and_privacy": "No account, chat, server address, or private data is visible.",
}

AI_REVIEW_PROMPT = """You are the visual quality gate for Minecraft trajectory SFT data.
Inspect every image in order, the question, and the proposed answer. Score every rubric dimension
from 1 to 5. Reject when any score is below 3, evidence is too dark or obscured, state changes do
not support the answer, GUI transitions lack required clicks, or camera motion is an obvious
outlier. In GUI, repeated held mouse states must become rising-edge click pulses. In gameplay,
continuous MouseLeft may represent holding to mine. For demonstration optimization, approve only
an independently cleaned answer, never the raw recording. Return JSON only with id, decision,
scores, reasons, suggested_revision, and reviewed_answer_sequence."""

HUMAN_REVIEW_PROMPT = """Review all images at full size and compare them with the question and
answer. Confirm chronology, visible intent, GUI/gameplay mouse semantics, privacy, and whether a
different reasonable answer remains allowed. Approve only when every dimension scores at least 3.
For demonstration optimization, edit the answer into a genuinely cleaner trajectory and mark its
reference_kind as reviewed_optimized_demonstration. Return the same JSON fields as the AI review."""

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


def automatic_quality_reasons(
    images: list[np.ndarray],
    actions: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    task_type: TaskType,
) -> list[str]:
    """在昂贵审核前过滤明显不可作答、跨界面和动作异常的窗口。"""
    reasons: list[str] = []
    arrays = [np.asarray(image, dtype=np.float32) for image in images]
    if min(float(image.mean()) for image in arrays) < 12.0:
        reasons.append("image_too_dark")
    gui_states = {bool(item.get("isGuiOpen")) for item in metadata}
    if len(gui_states) > 1:
        reasons.append("gui_state_transition_inside_context")
    camera = np.asarray(actions["camera"], dtype=np.float64)
    if camera.size and float(np.abs(camera).max()) > 350.0:
        reasons.append("camera_outlier")
    if task_type == "image_sequence_to_action":
        changes = [float(np.abs(right - left).mean()) for left, right in zip(arrays, arrays[1:])]
        if not changes or max(changes) < 0.8:
            reasons.append("insufficient_visual_change")
        in_gui = gui_states == {True}
        has_click = any(
            bool(np.asarray(actions.get(field, [])).astype(bool).any())
            for field in ("attack", "use")
        )
        if in_gui and max(changes, default=0.0) >= 2.0 and not has_click:
            reasons.append("gui_change_without_click")
    return reasons


def source_frames(task_type: TaskType, start: int) -> list[int]:
    """返回题面使用的帧号；预测题不会取目标区间的未来帧。"""
    if task_type == "image_sequence_to_action":
        return list(range(start, start + WINDOW_FRAMES + 1))
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


def _read_source_images(
    reader: TrajectoryReader, episode: str, frames: list[int],
) -> list[np.ndarray]:
    return [
        np.asarray(reader.readers["image"].read_frames(episode, frame, 1)[0], dtype=np.uint8)
        for frame in frames
    ]


def _action_blocks(
    reader: TrajectoryReader,
    episode: str,
    start: int,
    count: int,
) -> list[str]:
    total_frames = count * WINDOW_FRAMES
    actions = reader.readers["action"].read_frames(episode, start, total_frames)
    metadata = reader.readers["meta_info"].read_frames(episode, start, total_frames)
    normalized = normalize_gui_clicks(actions, metadata)
    return [
        encode_lumine_action({
            field: np.asarray(values)[
                index * WINDOW_FRAMES:(index + 1) * WINDOW_FRAMES
            ]
            for field, values in normalized.items()
        }).to_text()
        for index in range(count)
    ]


def normalize_gui_clicks(
    actions: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """把 GUI 内连续鼠标键 held 状态转换为离散按下沿脉冲。

    普通游戏画面保留原始 held 语义。GUI 中一次物理点击在采样数据里可能连续保持多个 tick；
    训练标签只在该连续段的首 tick 写鼠标键，后续 tick 留空，以显式表达释放间隔。
    """
    normalized = {field: np.array(values, copy=True) for field, values in actions.items()}
    if len(metadata) != len(np.asarray(actions["camera"])):
        raise ValueError("meta_info 与 action 帧数不一致")
    for field in ("attack", "use"):
        if field not in normalized:
            continue
        original = np.asarray(actions[field]).astype(bool)
        pulses = np.array(normalized[field], copy=True)
        previously_active = False
        for index, active in enumerate(original):
            in_gui = bool(metadata[index].get("isGuiOpen"))
            if in_gui:
                pulses[index] = int(active and not previously_active)
            else:
                pulses[index] = int(active)
            previously_active = bool(active) if in_gui else False
        normalized[field] = pulses
    return normalized


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
        dataset_directories, ["action", "image", "meta_info"], frame_width, frame_height,
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
                needed = {
                    "demonstration_optimization": OPTIMIZATION_WINDOWS * WINDOW_FRAMES,
                    "image_sequence_to_action": WINDOW_FRAMES + 1,
                    "history_to_future_action": WINDOW_FRAMES,
                }[task_type]
                last = reader.episode_length(episode) - needed
                if last < first:
                    continue
                start = randomizer.randint(first, last)
                frames = source_frames(task_type, start)
                source_images = _read_source_images(reader, episode, frames)
                source_metadata = [
                    reader.readers["meta_info"].read_frames(episode, frame, 1)[0]
                    for frame in frames
                ]
                target_actions = reader.readers["action"].read_frames(
                    episode, start, WINDOW_FRAMES,
                )
                reference = _action_blocks(reader, episode, start, 1)
                if not is_informative_action(target_actions):
                    continue
                if automatic_quality_reasons(
                    source_images, target_actions, source_metadata, task_type,
                ):
                    continue
                raw_actions = None
                answer_actions = reference
                if task_type == "demonstration_optimization":
                    raw_actions = _action_blocks(reader, episode, start, OPTIMIZATION_WINDOWS)
                    answer_actions = raw_actions
                sample_id = f"{task_type}_{generated:06d}"
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
    _write_jsonl(output / "ai_review_requests.jsonl", [
        {
            "id": question["id"],
            "system": AI_REVIEW_PROMPT,
            "rubric": REVIEW_DIMENSIONS,
            "question": question,
            "answer": next(item for item in answers if item["id"] == question["id"]),
        }
        for question in questions
    ])
    _write_jsonl(output / "human_review_templates.jsonl", [
        {
            "id": question["id"],
            "instructions": HUMAN_REVIEW_PROMPT,
            "question": question,
            "answer": next(item for item in answers if item["id"] == question["id"]),
            "decision": "revise",
            "scores": {dimension: 0 for dimension in REVIEW_DIMENSIONS},
            "reasons": [],
            "reviewed_answer_sequence": None,
        }
        for question in questions
    ])
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
        "automatic_filters": [
            "image_too_dark", "gui_state_transition_inside_context", "camera_outlier",
            "insufficient_visual_change", "gui_change_without_click",
        ],
        "gui_click_normalization": "held mouse buttons become rising-edge pulses in GUI frames",
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
