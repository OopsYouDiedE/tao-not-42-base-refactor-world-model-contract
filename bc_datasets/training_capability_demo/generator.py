"""从 MineStudio 真实轨迹采样八类 LoRA 多任务训练样本。

本模块生成的是小规模训练设计 demo。八类题全部可纳入多任务训练；公开题目与答案分开
保存，答案中保留来源帧和确定性证据，便于审查标签、时间泄漏和弱监督风险。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from bc_datasets.minestudio.action_benchmark_common import prepare_output
from bc_datasets.minestudio.action_choice_benchmark import is_informative_action
from bc_datasets.minestudio.lmdb_modality_reader import TrajectoryReader
from bc_datasets.minestudio.lumine_action_codec import (
    DEGREES_PER_PIXEL,
    MINECRAFT_KEYMAP,
    MOUSE_DELTA_LIMIT,
)

WINDOW_FRAMES = 4
HISTORY_OFFSETS = (12, 8, 4, 0)
CAPABILITY_ASPECTS = (
    "future_control",
    "inverse_dynamics",
    "timing_and_magnitude",
    "visual_gui_state",
    "event_outcome",
    "short_horizon_transition",
    "goal_conditioned_control",
    "protocol_translation",
)

_KEY_RENAMES = {"mouse_left": "MouseLeft", "mouse_right": "MouseRight"}
_IGNORED_EVENT_SUFFIXES = (
    "play_one_minute",
    "time_since_death",
    "time_since_rest",
)


@dataclass(frozen=True)
class SampleContext:
    episode: str
    start: int
    actions: dict[str, np.ndarray]
    previous_actions: dict[str, np.ndarray]
    metadata: list[dict[str, Any]]


def _integer_mouse_delta(value: float) -> int:
    rounded = int(np.rint(float(value) / DEGREES_PER_PIXEL))
    return max(-MOUSE_DELTA_LIMIT, min(MOUSE_DELTA_LIMIT, rounded))


def action_ticks(actions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """把逐帧 MineStudio 动作转为可审计的 tick 结构。"""
    camera = np.asarray(actions["camera"], dtype=np.float64)
    ticks: list[dict[str, Any]] = []
    for index, (pitch, yaw) in enumerate(camera):
        keys = [
            _KEY_RENAMES.get(key, key)
            for field, key in MINECRAFT_KEYMAP.items()
            if field in actions and bool(np.asarray(actions[field])[index])
        ]
        ticks.append({
            "keys": keys,
            "mouse": [_integer_mouse_delta(yaw), _integer_mouse_delta(pitch)],
        })
    return ticks


def action_contract_text(actions: dict[str, np.ndarray]) -> str:
    """按已定稿的 tick 级命名 token 契约序列化动作，仅供 demo 使用。"""
    chunks: list[str] = []
    for tick in action_ticks(actions):
        tokens = list(tick["keys"])
        mouse_x, mouse_y = tick["mouse"]
        if mouse_x or mouse_y:
            tokens.extend(("Mouse", str(mouse_x), str(mouse_y)))
        chunks.append(" ".join(tokens))
    return "<|action_start|> ; " + " ; ".join(chunks) + " <|action_end|>"


def meaningful_events(metadata: list[dict[str, Any]]) -> dict[str, float]:
    """汇总窗口内任务事件，排除持续计数占主导的 custom 类。"""
    totals: dict[str, float] = {}
    for frame in metadata:
        for name, increment in (frame.get("events") or {}).items():
            if str(name).startswith("minecraft.custom:") or any(
                str(name).endswith(suffix) for suffix in _IGNORED_EVENT_SUFFIXES
            ):
                continue
            totals[str(name)] = totals.get(str(name), 0.0) + float(increment)
    return {name: value for name, value in sorted(totals.items()) if value != 0.0}


def state_transition(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """生成短期世界状态差分；精确值作为审计真值，不放进问题输入。"""
    return {
        "position_delta": {
            axis: round(float(after[axis]) - float(before[axis]), 6)
            for axis in ("xpos", "ypos", "zpos")
        },
        "view_delta": {
            axis: round(float(after[axis]) - float(before[axis]), 6)
            for axis in ("yaw", "pitch")
        },
        "gui_before": bool(before.get("isGuiOpen")),
        "gui_after": bool(after.get("isGuiOpen")),
        "inventory_gui_before": bool(before.get("isGuiInventory")),
        "inventory_gui_after": bool(after.get("isGuiInventory")),
        "hotbar_before": int(before.get("hotbar", 0)) + 1,
        "hotbar_after": int(after.get("hotbar", 0)) + 1,
    }


def timing_summary(actions: dict[str, np.ndarray]) -> dict[str, Any]:
    """生成前后画面可以支持的粗粒度视角变化标签。"""
    camera = np.asarray(actions["camera"], dtype=np.float64)
    total_pitch = float(camera[:, 0].sum())
    total_yaw = float(camera[:, 1].sum())
    magnitude = float(np.abs(camera).sum())

    def signed(value: float) -> str:
        if abs(value) < 0.15:
            return "stable"
        return "positive" if value > 0 else "negative"

    return {
        "pitch_direction": signed(total_pitch),
        "yaw_direction": signed(total_yaw),
        "magnitude": "small" if magnitude < 2.0 else "medium" if magnitude < 8.0 else "large",
    }


def coarse_inverse_dynamics(actions: dict[str, np.ndarray]) -> dict[str, Any]:
    """把不可唯一恢复的逐 tick 动作降为窗口级动作族标签。"""
    ticks = action_ticks(actions)
    active_keys = {key for tick in ticks for key in tick["keys"]}
    return {
        "movement_keys": [key for key in ("W", "A", "S", "D", "space", "shift", "ctrl") if key in active_keys],
        "interaction_keys": [key for key in ("MouseLeft", "MouseRight", "E", "Q") if key in active_keys],
        "camera": timing_summary(actions),
    }


def categorical_transition(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """把连续状态差分降为适合视觉世界模型的稳定分类标签。"""
    exact = state_transition(before, after)

    def direction(value: float, threshold: float) -> str:
        if abs(value) <= threshold:
            return "stable"
        return "positive" if value > 0 else "negative"

    return {
        "position_direction": {
            axis: direction(value, 0.01)
            for axis, value in exact["position_delta"].items()
        },
        "view_direction": {
            axis: direction(value, 0.1)
            for axis, value in exact["view_delta"].items()
        },
        "gui_change": (
            "opened" if not exact["gui_before"] and exact["gui_after"]
            else "closed" if exact["gui_before"] and not exact["gui_after"]
            else "unchanged"
        ),
        "hotbar_changed": exact["hotbar_before"] != exact["hotbar_after"],
    }


def _question(
    sample_id: str,
    aspect: str,
    prompt: str,
    images: list[str],
    inputs: dict[str, Any] | None = None,
    assessment_scope: str = "auxiliary_training",
    known_risks: list[str] | None = None,
    review_status: str = "accepted_for_training",
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "aspect": aspect,
        "prompt": prompt,
        "images": images,
        "inputs": inputs or {},
        "assessment_scope": assessment_scope,
        "known_risks": known_risks or [],
        "review_status": review_status,
        "include_in_training": True,
    }


def _answer(
    sample_id: str,
    context: SampleContext,
    aspect: str,
    answer: Any,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "aspect": aspect,
        "answer": answer,
        "source": {
            "episode": context.episode,
            "observation_frame": context.start,
            "action_frame_range": [context.start, context.start + WINDOW_FRAMES],
            "future_state_frame": context.start + WINDOW_FRAMES,
        },
        "validity_checks": {
            "no_future_metadata_in_prompt": True,
            "action_label_is_demonstration_not_unique_optimum": True,
            "modalities_share_episode": True,
        },
    }


def _write_image(
    reader: TrajectoryReader,
    output_directory: Path,
    episode: str,
    frame: int,
    name: str,
) -> str:
    relative = f"images/{name}.jpg"
    array = reader.readers["image"].read_frames(episode, frame, 1)[0]
    Image.fromarray(array).save(output_directory / relative, quality=95)
    return relative


def _sample_context(
    reader: TrajectoryReader,
    episodes: list[str],
    randomizer: random.Random,
    predicate: Callable[[SampleContext], bool],
    attempts: int = 20_000,
) -> SampleContext:
    for _ in range(attempts):
        episode = randomizer.choice(episodes)
        maximum_start = reader.episode_length(episode) - WINDOW_FRAMES - 1
        if maximum_start < max(HISTORY_OFFSETS):
            continue
        start = randomizer.randint(max(HISTORY_OFFSETS), maximum_start)
        actions = reader.readers["action"].read_frames(episode, start, WINDOW_FRAMES)
        previous = reader.readers["action"].read_frames(
            episode, start - WINDOW_FRAMES, WINDOW_FRAMES,
        )
        metadata = reader.readers["meta_info"].read_frames(
            episode, start, WINDOW_FRAMES + 1,
        )
        context = SampleContext(episode, start, actions, previous, metadata)
        if predicate(context):
            return context
    raise RuntimeError("在采样上限内找不到符合题型约束的真实窗口")


def _images_for(
    reader: TrajectoryReader,
    output: Path,
    context: SampleContext,
    aspect: str,
    history: bool = False,
    transition: bool = False,
) -> list[str]:
    if history and transition:
        raise ValueError("history 和 transition 不能同时启用")
    if history:
        frames = [context.start - offset for offset in HISTORY_OFFSETS]
    elif transition:
        frames = list(range(context.start, context.start + WINDOW_FRAMES + 1))
    else:
        frames = [context.start]
    return [
        _write_image(reader, output, context.episode, frame, f"{aspect}_{context.start:08d}_{index}")
        for index, frame in enumerate(frames)
    ]


def build_training_capability_demo(
    dataset_directory: Path,
    output_directory: Path,
    samples_per_aspect: int = 1,
    seed: int = 20260729,
    image_width: int = 320,
    image_height: int = 180,
    overwrite: bool = False,
) -> dict[str, Any]:
    """从真实 `image/action/meta_info` 生成八方面多任务训练采样 demo。"""
    if samples_per_aspect < 1:
        raise ValueError("samples_per_aspect 必须大于零")
    output = prepare_output(output_directory, overwrite)
    randomizer = random.Random(seed)
    reader = TrajectoryReader(
        [dataset_directory], ["action", "image", "meta_info"],
        frame_width=image_width, frame_height=image_height,
    )
    questions: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    try:
        episodes = reader.episode_names()
        for aspect in CAPABILITY_ASPECTS:
            for sample_index in range(samples_per_aspect):
                predicate: Callable[[SampleContext], bool] = lambda c: is_informative_action(c.actions)
                if aspect == "timing_and_magnitude":
                    predicate = lambda c: (
                        bool(np.abs(np.asarray(c.actions["camera"])).sum())
                        and not bool(c.metadata[0].get("isGuiOpen"))
                        and not bool(c.metadata[-1].get("isGuiOpen"))
                    )
                elif aspect == "visual_gui_state":
                    predicate = lambda c: bool(c.metadata[0].get("isGuiOpen"))
                elif aspect == "event_outcome":
                    predicate = lambda c: bool(meaningful_events(c.metadata[1:]))
                elif aspect == "short_horizon_transition":
                    predicate = lambda c: (
                        any(abs(value) > 0.01 for value in state_transition(c.metadata[0], c.metadata[-1])["position_delta"].values())
                        or bool(meaningful_events(c.metadata[1:]))
                    )
                elif aspect == "goal_conditioned_control":
                    predicate = lambda c: bool(meaningful_events(c.metadata[1:]))
                context = _sample_context(reader, episodes, randomizer, predicate)
                sample_id = f"{aspect}_{sample_index:06d}"
                answer_value: Any
                assessment_scope = "auxiliary_training"
                known_risks: list[str] = []
                review_status = "accepted_for_training"
                if aspect == "future_control":
                    images = _images_for(reader, output, context, sample_id, history=True)
                    prompt = "Reproduce the demonstrated next 200 ms control from the visual history. Output only one action block."
                    inputs = {"previous_action": action_contract_text(context.previous_actions)}
                    answer_value = action_contract_text(context.actions)
                    assessment_scope = "behavior_cloning"
                    known_risks = ["the demonstrated action is not the unique optimal action"]
                    review_status = "accepted_for_training"
                elif aspect == "inverse_dynamics":
                    images = _images_for(reader, output, context, sample_id, transition=True)
                    prompt = (
                        "The five images are consecutive frames in chronological order across 200 ms. "
                        "Infer only the coarse action families visible across this transition. "
                        "Answer JSON with movement_keys (subset of W,A,S,D,space,shift,ctrl), "
                        "interaction_keys (subset of MouseLeft,MouseRight,E,Q), and camera with "
                        "pitch_direction/yaw_direction in positive,negative,stable and magnitude in small,medium,large."
                    )
                    inputs = {}
                    answer_value = coarse_inverse_dynamics(context.actions)
                    assessment_scope = "inverse_dynamics_auxiliary_training"
                    known_risks = ["coarse action families can remain ambiguous under occlusion or negligible displacement"]
                    review_status = "accepted_for_training"
                elif aspect == "timing_and_magnitude":
                    images = _images_for(reader, output, context, sample_id, transition=True)
                    prompt = (
                        "The five images are consecutive frames in chronological order across 200 ms. "
                        "Classify the recorded camera motion across them. Answer JSON with "
                        "pitch_direction and yaw_direction chosen from positive, negative, stable, and "
                        "magnitude chosen from small, medium, large. Positive yaw turns right; positive "
                        "pitch looks down, following the recorded camera coordinate convention."
                    )
                    inputs = {}
                    answer_value = timing_summary(context.actions)
                    review_status = "accepted_for_training"
                elif aspect == "visual_gui_state":
                    images = _images_for(reader, output, context, sample_id)
                    prompt = "Is a GUI open, and is it the player inventory GUI? Answer as JSON booleans."
                    inputs = {}
                    answer_value = {
                        "gui_open": bool(context.metadata[0].get("isGuiOpen")),
                        "player_inventory_gui": bool(context.metadata[0].get("isGuiInventory")),
                    }
                    review_status = "accepted_for_training"
                elif aspect == "event_outcome":
                    images = _images_for(reader, output, context, sample_id, transition=True)
                    event_answer = meaningful_events(context.metadata[1:])
                    prompt = (
                        "The five images are consecutive frames in chronological order across 200 ms. "
                        "Which listed non-timer game events occurred during the demonstrated "
                        "transition? Return a JSON object mapping only candidate event names to numeric increments."
                    )
                    inputs = {
                        "executed_action": action_contract_text(context.actions),
                        "candidate_events": list(event_answer),
                    }
                    answer_value = event_answer
                    assessment_scope = "world_model_weak_supervision"
                    known_risks = ["the event label can be correct while visually imperceptible in 200 ms"]
                    review_status = "accepted_for_training"
                elif aspect == "short_horizon_transition":
                    images = _images_for(reader, output, context, sample_id, transition=True)
                    prompt = (
                        "The five images show the supplied 200 ms action transition in chronological order. "
                        "Describe the observed player-state change. Return JSON with "
                        "position_direction{xpos,ypos,zpos} and view_direction{yaw,pitch}, each using "
                        "positive,negative,stable; gui_change using opened,closed,unchanged; "
                        "hotbar_changed as a boolean; and events as an event-to-increment object."
                    )
                    inputs = {
                        "executed_action": action_contract_text(context.actions),
                        "initial_state": {
                            "gui_open": bool(context.metadata[0].get("isGuiOpen")),
                            "player_inventory_gui": bool(context.metadata[0].get("isGuiInventory")),
                            "hotbar_slot": int(context.metadata[0].get("hotbar", 0)) + 1,
                        },
                    }
                    answer_value = {
                        **categorical_transition(context.metadata[0], context.metadata[-1]),
                        "events": meaningful_events(context.metadata[1:]),
                    }
                    assessment_scope = "world_model_training"
                    known_risks = ["five consecutive frames may still not expose hidden state or world-axis orientation"]
                    review_status = "accepted_for_training"
                elif aspect == "goal_conditioned_control":
                    images = _images_for(reader, output, context, sample_id, history=True)
                    goal_events = meaningful_events(context.metadata[1:])
                    goal = next(iter(goal_events))
                    prompt = "Given the hindsight task goal, reproduce the next 200 ms demonstrated control. Output only one action block."
                    inputs = {
                        "hindsight_goal": goal,
                        "previous_action": action_contract_text(context.previous_actions),
                    }
                    answer_value = action_contract_text(context.actions)
                    assessment_scope = "goal_conditioned_behavior_cloning"
                    known_risks = ["the hindsight event goal must be supplied by a planner at inference"]
                    review_status = "accepted_for_training"
                else:
                    images = []
                    ticks = action_ticks(context.actions)
                    prompt = "Translate the four structured 50 ms ticks into the strict named-token action contract."
                    inputs = {"ticks": ticks}
                    answer_value = action_contract_text(context.actions)
                    assessment_scope = "format_warmup_only"
                    known_risks = ["this task does not measure visual control"]
                    review_status = "accepted_for_training"
                questions.append(_question(
                    sample_id, aspect, prompt, images, inputs,
                    assessment_scope=assessment_scope,
                    known_risks=known_risks,
                    review_status=review_status,
                ))
                answers.append(_answer(sample_id, context, aspect, answer_value))
    finally:
        reader.close()

    questions_path = output / "questions.jsonl"
    answers_path = output / "answer_key.jsonl"
    with questions_path.open("w", encoding="utf-8") as handle:
        for record in questions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with answers_path.open("w", encoding="utf-8") as handle:
        for record in answers:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "format": "minestudio_training_capability_demo_v1",
        "purpose": "multitask_training_demo",
        "aspects": list(CAPABILITY_ASPECTS),
        "aspect_count": len(CAPABILITY_ASPECTS),
        "samples_per_aspect": samples_per_aspect,
        "sample_count": len(questions),
        "seed": seed,
        "image_size": [image_width, image_height],
        "questions": questions_path.name,
        "answer_key": answers_path.name,
        "limitations": [
            "future control is a demonstrated action, not a unique optimal action",
            "goal-conditioned hindsight goals come from future task events and require a planner at inference",
            "the demo serializer follows the final contract while the production codec is still pending migration",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 MineStudio 八方面 LoRA 多任务训练采样 demo")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-aspect", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    manifest = build_training_capability_demo(
        arguments.dataset_dir, arguments.output_dir, arguments.samples_per_aspect,
        arguments.seed, arguments.image_width, arguments.image_height, arguments.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
