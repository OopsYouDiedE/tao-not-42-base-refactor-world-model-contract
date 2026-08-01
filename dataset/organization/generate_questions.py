"""从 MineStudio 真实轨迹机器生成四类动作训练题。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from PIL import Image

from dataset.extraction.minestudio import MineStudioDataset
from dataset.organization.sft_protocol import TASK_PROMPTS
from tao.protocols.action import decode_action_sequence, encode_action_sequence

TaskType: TypeAlias = Literal[
    "demonstration_optimization",
    "image_sequence_to_action",
    "history_to_future_action",
    "single_frame_intent_to_action",
]
TASK_TYPES: tuple[TaskType, ...] = (
    "demonstration_optimization",
    "image_sequence_to_action",
    "history_to_future_action",
    "single_frame_intent_to_action",
)
OUTPUT_CONTRACT = {
    "type": "json_array",
    "item": "variable-length named-token action block",
    "action_markers": ["<|action_start|>", "<|action_end|>"],
    "chunk_count": "variable; use one action block per task-required action interval",
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
Inspect every image in order, the question, and the proposed answer. Return only approve or reject
with one concise reason. Reject when evidence is too dark or obscured, state changes do not support
the answer, GUI transitions lack required clicks, or camera motion is an obvious outlier. In GUI,
repeated held mouse states must become rising-edge click pulses. In gameplay, continuous MouseLeft
may represent holding to mine. For demonstration optimization, approve only an independently
cleaned answer, never the raw recording. Return JSON only with id, decision, and reason."""

HUMAN_REVIEW_PROMPT = """Review all images at full size and compare them with the question and
answer. Confirm chronology, visible intent, GUI/gameplay mouse semantics, privacy, and whether a
different reasonable answer remains allowed. Return only approve or reject with one concise reason.
Demonstration optimization answers still require a separately cleaned trajectory with reference_kind
reviewed_optimized_demonstration before packing."""

WINDOW_FRAMES = 4
HISTORY_OFFSETS = (12, 8, 4, 0)
OPTIMIZATION_WINDOWS = 4


def is_informative_action(actions: dict[str, np.ndarray]) -> bool:
    """动作窗口包含按键或可见相机移动时返回真。"""
    camera = np.asarray(actions["camera"], dtype=np.float64)
    if float(np.linalg.norm(camera, axis=-1).sum()) >= 3.0:
        return True
    return any(
        bool(np.asarray(values).astype(bool).any())
        for field, values in actions.items()
        if field != "camera"
    )


def infer_action_intent(actions: dict[str, np.ndarray]) -> tuple[str, str] | None:
    """把未来动作归纳为可审核的意图文本和用于均衡采样的主类别。"""
    active = {
        field
        for field, values in actions.items()
        if field != "camera" and bool(np.asarray(values).astype(bool).any())
    }
    camera_distance = float(
        np.linalg.norm(np.asarray(actions["camera"], dtype=np.float64), axis=-1).sum()
    )
    phrases: list[str] = []
    category = "camera"
    if "attack" in active:
        phrases.append("hold the primary action to mine or attack the visible target")
        category = "attack"
    if "use" in active:
        phrases.append("use or place the selected item")
        category = "use"
    movement = active.intersection({"forward", "back", "left", "right"})
    if movement:
        directions = ", ".join(sorted(movement))
        phrases.append(f"move {directions}")
        if category == "camera":
            category = "movement"
    if "jump" in active:
        phrases.append("jump")
        if category == "camera":
            category = "jump"
    if "sprint" in active:
        phrases.append("sprint")
    if "sneak" in active:
        phrases.append("sneak")
    if "inventory" in active:
        phrases.append("open or close the inventory")
        category = "inventory"
    hotbars = sorted(field for field in active if field.startswith("hotbar."))
    if hotbars:
        phrases.append(f"select hotbar slot {hotbars[0].split('.', 1)[1]}")
        category = "hotbar"
    if "drop" in active:
        phrases.append("drop the selected item")
        category = "drop"
    if camera_distance >= 3.0:
        phrases.append("adjust the view or GUI cursor toward the target")
    if not phrases:
        return None
    return "; then ".join(phrases), category


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
        changes = [
            float(np.abs(right - left).mean())
            for left, right in zip(arrays, arrays[1:], strict=False)
        ]
        if not changes or max(changes) < 0.8:
            reasons.append("insufficient_visual_change")
        in_gui = gui_states == {True}
        has_click = any(
            bool(np.asarray(actions.get(field, [])).astype(bool).any())
            for field in ("attack", "use")
        )
        if in_gui and max(changes, default=0.0) >= 2.0 and not has_click:
            reasons.append("gui_change_without_click")
    if infer_action_intent(actions) is None:
        reasons.append("static_or_weak_action")
    return reasons


def source_frames(task_type: TaskType, start: int) -> list[int]:
    """返回题面使用的帧号；预测题不会取目标区间的未来帧。"""
    if task_type == "image_sequence_to_action":
        return list(range(start, start + WINDOW_FRAMES + 1))
    if task_type == "history_to_future_action":
        return [start - offset for offset in HISTORY_OFFSETS]
    if task_type == "single_frame_intent_to_action":
        return [start]
    return [
        start + offset for offset in range(0, OPTIMIZATION_WINDOWS * WINDOW_FRAMES, WINDOW_FRAMES)
    ]


def target_interval_for(task_type: TaskType, start: int) -> list[int]:
    """根据题面输入帧定义动作边界；预测题的终点是唯一未来关键帧。"""
    frames = source_frames(task_type, start)
    if task_type in {"demonstration_optimization", "image_sequence_to_action"}:
        return [frames[0], frames[-1]]
    return [frames[-1], frames[-1] + WINDOW_FRAMES]


def action_node_frames(
    task_type: TaskType,
    image_frames: list[int],
    target_interval: list[int],
) -> list[int]:
    """返回动作分段节点；每段 tick 数严格等于相邻节点帧差。"""
    if task_type in {"demonstration_optimization", "image_sequence_to_action"}:
        nodes = list(image_frames)
    else:
        nodes = [image_frames[-1], target_interval[1]]
    if len(nodes) < 2 or any(right <= left for left, right in zip(nodes, nodes[1:], strict=False)):
        raise ValueError("动作节点必须至少有两个，并且帧号严格递增")
    return nodes


def _prepare_output(output: Path, overwrite: bool, append: bool = False) -> None:
    if append:
        (output / "images").mkdir(parents=True, exist_ok=True)
        return
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_generated_dataset(output: Path) -> dict[str, int]:
    """重新读取题面、答案、图片与动作协议，确认生成结果完整落盘。"""
    questions = _read_jsonl(output / "questions.jsonl")
    answers = {record["id"]: record for record in _read_jsonl(output / "answer_key.jsonl")}
    if len(questions) != len(answers):
        raise ValueError("questions 与 answer_key 数量不一致")
    image_count = 0
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise ValueError(f"题目 {question['id']} 缺少答案")
        interval = question["target_interval"]
        action_nodes = action_node_frames(
            question["task_type"],
            question["source"]["image_frames"],
            interval,
        )
        answer_source = answer.get("source", {})
        if answer_source.get("action_start_frame") != interval[0]:
            raise ValueError(f"题目 {question['id']} 的参考动作起点与目标区间不一致")
        if answer_source.get("action_end_frame") != interval[1]:
            raise ValueError(f"题目 {question['id']} 的参考动作终点与目标区间不一致")
        for relative in question["images"]:
            path = output / relative
            with Image.open(path) as image:
                image.verify()
            image_count += 1
        reference_blocks = answer["reference_action_sequence"]
        if len(reference_blocks) != len(action_nodes) - 1:
            raise ValueError(f"题目 {question['id']} 的参考动作段数与图像间隔数不一致")
        action_ticks = 0
        for segment_index, block in enumerate(reference_blocks):
            ticks = decode_action_sequence(block).ticks
            if not ticks:
                raise ValueError(f"题目 {question['id']} 包含空动作块")
            expected_ticks = action_nodes[segment_index + 1] - action_nodes[segment_index]
            if len(ticks) != expected_ticks:
                raise ValueError(
                    f"题目 {question['id']} 第 {segment_index + 1} 段动作长度为 {len(ticks)}，"
                    f"对应图像间隔为 {expected_ticks} 帧"
                )
            action_ticks += len(ticks)
        if action_ticks != interval[1] - interval[0]:
            raise ValueError(
                f"题目 {question['id']} 的参考动作共 {action_ticks} tick，"
                f"但目标区间长度为 {interval[1] - interval[0]} 帧"
            )
        if question["task_type"] == "single_frame_intent_to_action":
            inputs = question.get("inputs", {})
            if len(question["images"]) != 1 or "intent" not in inputs:
                raise ValueError(f"单帧意图题 {question['id']} 输入不完整")
            if not inputs["intent"] and inputs.get("intent_status") != "pending_human_authoring":
                raise ValueError(f"单帧意图题 {question['id']} 缺少人工意图状态")
    return {"sample_count": len(questions), "image_count": image_count}


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
        "# MineStudio 轨迹训练题生成报告",
        "",
        "> 本报告由出题流程自动生成。图片与参考动作来自真实 MineStudio 轨迹。",
        "> 参考轨迹是一种人类示范，不是唯一正确答案。`answer_key.jsonl` 不应交给做题模型。",
        "",
        "## 汇总",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 候选题目 | {len(questions)} |",
        f"| 结构审核完成 | {len(review_by_id)} |",
        f"| 结构审核通过 | {sum(r.get('decision') == 'pass' for r in review_by_id.values())} |",
        "",
    ]
    for question in questions:
        sample_id = question["id"]
        answer = answer_by_id.get(sample_id, {})
        review = review_by_id.get(sample_id)
        lines.extend(
            [
                f"## {sample_id}",
                "",
                "| 字段 | 内容 |",
                "|---|---|",
                f"| 题型 | `{question['task_type']}` |",
                f"| 来源 episode | `{question['source']['episode']}` |",
                f"| 图片帧 | `{question['source']['image_frames']}` |",
                f"| 目标动作区间 | `{question['target_interval']}` |",
                f"| 初始训练准入 | `{question['include_in_training']}` |",
                f"| 结构审核 | `{review['decision'] if review else 'pending'}` |",
                "",
                "### 图片",
                "",
            ]
        )
        for index, image_path in enumerate(question["images"]):
            frame = question["source"]["image_frames"][index]
            lines.extend(
                [
                    f"**图 {index + 1}，帧 {frame}**",
                    "",
                    f"![{sample_id} frame {frame}]({image_path})",
                    "",
                ]
            )
        lines.extend(["### 问题", "", question["prompt"], ""])
        raw = question.get("inputs", {}).get("raw_action_sequence")
        if raw:
            lines.extend(["### 待优化的原始动作序列", ""])
            for index, block in enumerate(raw, 1):
                lines.extend([f"动作块 {index}：", "", "```text", block, "```", ""])
        intent = question.get("inputs", {}).get("intent")
        if intent:
            lines.extend(["### 给定意图", "", intent, ""])
        lines.extend(
            [
                "### 参考答案轨迹",
                "",
                "参考类型：真实人类演示；该轨迹不是唯一合理答案。",
                "",
            ]
        )
        for index, block in enumerate(answer.get("reference_action_sequence", []), 1):
            lines.extend([f"动作块 {index}：", "", "```text", block, "```", ""])
        if review is not None:
            lines.extend(
                [
                    "### 结构校验结果",
                    "",
                    "```json",
                    json.dumps(review, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_images(
    reader: MineStudioDataset,
    output: Path,
    sample_id: str,
    episode: str,
    frames: list[int],
) -> list[str]:
    paths: list[str] = []
    for index, frame_index in enumerate(frames):
        relative = f"images/{sample_id}_{index:02d}.jpg"
        frame = reader.read_modality("image", episode, frame_index, 1)[0]
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
            output / relative,
            quality=95,
        )
        paths.append(relative)
    return paths


def _read_source_images(
    reader: MineStudioDataset,
    episode: str,
    frames: list[int],
) -> list[np.ndarray]:
    return [
        np.asarray(reader.read_modality("image", episode, frame, 1)[0], dtype=np.uint8)
        for frame in frames
    ]


def _action_blocks_between(
    reader: MineStudioDataset,
    episode: str,
    nodes: list[int],
) -> list[str]:
    if len(nodes) < 2 or any(right <= left for left, right in zip(nodes, nodes[1:], strict=False)):
        raise ValueError("动作节点必须至少有两个，并且帧号严格递增")
    start = nodes[0]
    total_frames = nodes[-1] - start
    actions = reader.read_modality("action", episode, start, total_frames)
    metadata = reader.read_modality("meta_info", episode, start, total_frames)
    normalized = normalize_gui_clicks(actions, metadata)
    return [
        encode_action_sequence(
            {
                field: np.asarray(values)[left - start : right - start]
                for field, values in normalized.items()
            }
        ).to_text()
        for left, right in zip(nodes, nodes[1:], strict=False)
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
    intent: str | None = None,
) -> dict[str, Any]:
    """构造公开题面。来源定位保留给审核，目标动作只进入独立答案文件。"""
    frames = source_frames(task_type, start)
    interval = target_interval_for(task_type, start)
    nodes = action_node_frames(task_type, frames, interval)
    inputs: dict[str, Any] = {"images_chronological": True}
    inputs["action_block_ticks"] = [
        right - left for left, right in zip(nodes, nodes[1:], strict=False)
    ]
    if raw_actions is not None:
        inputs["raw_action_sequence"] = raw_actions
    if intent is not None:
        inputs["intent"] = intent
        if task_type == "single_frame_intent_to_action" and not intent:
            inputs["intent_status"] = "pending_human_authoring"
    return {
        "id": sample_id,
        "task_type": task_type,
        "prompt": TASK_PROMPTS[task_type],
        "images": images,
        "inputs": inputs,
        "output_contract": OUTPUT_CONTRACT,
        "source": {"episode": episode, "image_frames": frames},
        "target_interval": interval,
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
    append: bool = False,
) -> dict[str, Any]:
    """生成题面、隔离的参考示范和待审核清单。"""
    if samples_per_type < 1:
        raise ValueError("samples_per_type 必须大于零")
    output = Path(output_directory)
    if overwrite and append:
        raise ValueError("--overwrite 与 --append 不能同时使用")
    _prepare_output(output, overwrite, append)
    randomizer = random.Random(seed)
    reader = MineStudioDataset(
        dataset_directories[0], ["action", "image", "meta_info"]
    ).updata_index()
    questions: list[dict[str, Any]] = _read_jsonl(output / "questions.jsonl") if append else []
    answers: list[dict[str, Any]] = _read_jsonl(output / "answer_key.jsonl") if append else []
    initial_count = len(questions)
    generated_per_type: dict[str, int] = {}
    try:
        episodes = reader.keys
        if not episodes:
            raise ValueError("action 与 image 模态没有共同 episode")
        for task_type in TASK_TYPES:
            existing_type_count = sum(item["task_type"] == task_type for item in questions)
            generated = 0
            attempts = 0
            intent_counts: dict[str, int] = {}
            while generated < samples_per_type and attempts < samples_per_type * 500:
                attempts += 1
                episode = randomizer.choice(episodes)
                first = max(HISTORY_OFFSETS) if task_type == "history_to_future_action" else 0
                needed = {
                    "demonstration_optimization": ((OPTIMIZATION_WINDOWS - 1) * WINDOW_FRAMES + 1),
                    "image_sequence_to_action": WINDOW_FRAMES + 1,
                    "history_to_future_action": WINDOW_FRAMES,
                    "single_frame_intent_to_action": WINDOW_FRAMES,
                }[task_type]
                last = reader.lengths[episode] - needed
                if last < first:
                    continue
                start = randomizer.randint(first, last)
                frames = source_frames(task_type, start)
                source_images = _read_source_images(reader, episode, frames)
                target_interval = target_interval_for(task_type, start)
                action_frame_count = target_interval[1] - target_interval[0]
                action_nodes = action_node_frames(task_type, frames, target_interval)
                target_actions = reader.read_modality(
                    "action",
                    episode,
                    start,
                    action_frame_count,
                )
                target_metadata = reader.read_modality(
                    "meta_info",
                    episode,
                    start,
                    action_frame_count,
                )
                reference = _action_blocks_between(reader, episode, action_nodes)
                if not is_informative_action(target_actions):
                    continue
                if automatic_quality_reasons(
                    source_images,
                    target_actions,
                    target_metadata,
                    task_type,
                ):
                    continue
                inferred_intent = infer_action_intent(target_actions)
                intent = "" if task_type == "single_frame_intent_to_action" else None
                if task_type == "single_frame_intent_to_action":
                    if inferred_intent is None:
                        continue
                    category = inferred_intent[1]
                    category_limit = max(2, int(samples_per_type * 0.35))
                    if intent_counts.get(category, 0) >= category_limit:
                        continue
                raw_actions = None
                answer_actions = reference
                if task_type == "demonstration_optimization":
                    raw_actions = reference
                    answer_actions = raw_actions
                sample_id = f"{task_type}_{existing_type_count + generated:06d}"
                images = _save_images(reader, output, sample_id, episode, frames)
                questions.append(
                    build_question_record(
                        sample_id,
                        task_type,
                        episode,
                        start,
                        images,
                        raw_actions,
                        intent,
                    )
                )
                answers.append(
                    {
                        "id": sample_id,
                        "task_type": task_type,
                        "reference_action_sequence": answer_actions,
                        "reference_kind": "recorded_human_demonstration",
                        "reference_is_unique": False,
                        "source": {
                            "episode": episode,
                            "action_start_frame": start,
                            "action_end_frame": start + action_frame_count,
                        },
                    }
                )
                generated += 1
                if task_type == "single_frame_intent_to_action" and inferred_intent is not None:
                    category = inferred_intent[1]
                    intent_counts[category] = intent_counts.get(category, 0) + 1
            generated_per_type[task_type] = generated
    finally:
        reader.close()
    _write_jsonl(output / "questions.jsonl", questions)
    _write_jsonl(output / "answer_key.jsonl", answers)
    _write_jsonl(
        output / "ai_review_requests.jsonl",
        [
            {
                "id": question["id"],
                "system": AI_REVIEW_PROMPT,
                "rubric": REVIEW_DIMENSIONS,
                "question": question,
                "answer": next(item for item in answers if item["id"] == question["id"]),
            }
            for question in questions
        ],
    )
    _write_jsonl(
        output / "human_review_templates.jsonl",
        [
            {
                "id": question["id"],
                "instructions": HUMAN_REVIEW_PROMPT,
                "question": question,
                "answer": next(item for item in answers if item["id"] == question["id"]),
                "decision": "pending",
                "reason": "",
            }
            for question in questions
        ],
    )
    write_dataset_readme(output, questions, answers)
    manifest = {
        "format": "minestudio_trajectory_questions_v1",
        "task_types": list(TASK_TYPES),
        "samples_per_type": samples_per_type,
        "sample_count": len(questions),
        "new_sample_count": len(questions) - initial_count,
        "new_samples_per_type": generated_per_type,
        "shortfall_per_type": {
            task_type: samples_per_type - count
            for task_type, count in generated_per_type.items()
            if count < samples_per_type
        },
        "seed": seed,
        "frame_size": [frame_width, frame_height],
        "questions": "questions.jsonl",
        "answer_key": "answer_key.jsonl",
        "readme": "README.md",
        "initial_review_status": "pending_human_and_ai_review",
        "automatic_filters": [
            "image_too_dark",
            "gui_state_transition_inside_context",
            "camera_outlier",
            "insufficient_visual_change",
            "gui_change_without_click",
            "static_or_weak_action",
        ],
        "gui_click_normalization": "held mouse buttons become rising-edge pulses in GUI frames",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["validation"] = validate_generated_dataset(output)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="从 MineStudio 生成四类动作轨迹训练题")
    parser.add_argument("--dataset-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    arguments = parser.parse_args()
    manifest = build_questions(
        arguments.dataset_dir,
        arguments.output_dir,
        arguments.samples_per_type,
        arguments.seed,
        arguments.frame_width,
        arguments.frame_height,
        arguments.overwrite,
        arguments.append,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
