"""从 MineStudio 轨迹构造“两帧选四段动作”的四选一测试集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from bc_datasets.minestudio.action_benchmark_common import (
    CHOICE_LABELS,
    ActionLocation,
    copy_actions as _copy_actions,
    load_episode_subset as _load_episode_subset,
    prepare_output as _prepare_output,
    random_location as _random_location,
    read_action as _read_action,
    serialize_action,
    serialized_action_signature,
    shuffled_choices,
)
from bc_datasets.minestudio.lmdb_modality_reader import TrajectoryReader
from bc_datasets.minestudio.lumine_action_codec import MINECRAFT_KEYMAP

ACTION_TYPES = ("camera_only", "movement", "interaction", "mixed")
DETAILED_ACTION_TYPES = (
    "camera_only",
    "locomotion",
    "jump",
    "attack",
    "use",
    "inventory_hotbar",
    "move_attack",
    "move_use",
    "complex",
)
MOVEMENT_FIELDS = frozenset({"forward", "back", "left", "right", "jump", "sneak", "sprint"})
INTERACTION_FIELDS = frozenset(
    {"attack", "use", "drop", "inventory", *(f"hotbar.{index}" for index in range(1, 10))},
)


def action_signature(actions: dict[str, np.ndarray]) -> str:
    """返回动作的稳定签名，用于排除四个候选中完全相同的序列。"""
    digest = hashlib.sha256()
    for field in sorted(actions):
        value = np.ascontiguousarray(actions[field])
        digest.update(field.encode())
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def is_informative_action(actions: dict[str, np.ndarray]) -> bool:
    """动作窗口包含视角变化或至少一个按键时返回真。"""
    camera = np.asarray(actions["camera"], dtype=np.float64)
    if np.any(np.abs(camera) > 1e-9):
        return True
    return any(
        field in actions and np.asarray(actions[field]).astype(bool).any()
        for field in MINECRAFT_KEYMAP
    )


def classify_action_type(actions: dict[str, np.ndarray]) -> str:
    """按互斥主语义把动作分成纯视角、移动、交互和混合四类。"""
    movement = any(
        field in actions and np.asarray(actions[field]).astype(bool).any()
        for field in MOVEMENT_FIELDS
    )
    interaction = any(
        field in actions and np.asarray(actions[field]).astype(bool).any()
        for field in INTERACTION_FIELDS
    )
    if movement and interaction:
        return "mixed"
    if movement:
        return "movement"
    if interaction:
        return "interaction"
    camera = np.asarray(actions["camera"], dtype=np.float64)
    if np.any(np.abs(camera) > 1e-9):
        return "camera_only"
    return "noop"


def _has_action(actions: dict[str, np.ndarray], fields: Iterable[str]) -> bool:
    return any(
        field in actions and np.asarray(actions[field]).astype(bool).any()
        for field in fields
    )


def classify_detailed_action_type(actions: dict[str, np.ndarray]) -> str:
    """把长窗口动作划为九个互斥类别，用于全面覆盖测试。"""
    locomotion_fields = MOVEMENT_FIELDS - {"jump"}
    item_fields = INTERACTION_FIELDS - {"attack", "use"}
    locomotion = _has_action(actions, locomotion_fields)
    jump = _has_action(actions, {"jump"})
    attack = _has_action(actions, {"attack"})
    use = _has_action(actions, {"use"})
    item = _has_action(actions, item_fields)
    if locomotion and attack and not use:
        return "move_attack"
    if locomotion and use and not attack:
        return "move_use"
    if jump and not attack and not use:
        return "jump"
    if attack and not locomotion and not use:
        return "attack"
    if use and not locomotion and not attack:
        return "use"
    if item and not locomotion and not jump and not attack and not use:
        return "inventory_hotbar"
    if locomotion and not jump and not attack and not use:
        return "locomotion"
    if not any((locomotion, jump, attack, use, item)) and np.any(
        np.abs(np.asarray(actions["camera"], dtype=np.float64)) > 1e-9
    ):
        return "camera_only"
    return "complex" if is_informative_action(actions) else "noop"


def _swap_fields(
    actions: dict[str, np.ndarray],
    first: str,
    second: str,
) -> dict[str, np.ndarray]:
    candidate = _copy_actions(actions)
    template = np.zeros(np.asarray(actions["camera"]).shape[0], dtype=np.int64)
    first_value = np.asarray(actions.get(first, template))
    second_value = np.asarray(actions.get(second, template))
    candidate[first] = np.array(second_value, copy=True)
    candidate[second] = np.array(first_value, copy=True)
    return candidate


def counterfactual_candidates(
    correct_actions: dict[str, np.ndarray],
    randomizer: random.Random | None = None,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """生成只改变方向、幅度、按键或时序的受控反事实候选库。"""
    candidates: list[tuple[str, dict[str, np.ndarray]]] = []
    camera = np.asarray(correct_actions["camera"], dtype=np.float64)
    for name, columns in (
        ("reverse_yaw", (1,)),
        ("reverse_pitch", (0,)),
        ("reverse_camera", (0, 1)),
    ):
        candidate = _copy_actions(correct_actions)
        for column in columns:
            candidate["camera"][:, column] *= -1
        candidates.append((name, candidate))
    for factor in (0.5, 1.5):
        candidate = _copy_actions(correct_actions)
        candidate["camera"] = camera * factor
        candidates.append((f"scale_camera_{factor:g}", candidate))

    reversed_candidate = {
        field: np.asarray(value)[::-1].copy()
        for field, value in correct_actions.items()
    }
    candidates.append(("reverse_timeline", reversed_candidate))
    for offset in (-1, 1):
        shifted = {
            field: np.roll(np.asarray(value), offset, axis=0).copy()
            for field, value in correct_actions.items()
        }
        candidates.append((f"shift_timeline_{offset:+d}", shifted))

    for first, second in (
        ("forward", "back"),
        ("left", "right"),
        ("attack", "use"),
        ("jump", "sneak"),
    ):
        candidates.append((f"swap_{first}_{second}", _swap_fields(correct_actions, first, second)))

    num_frames = camera.shape[0]
    active_fields = [
        field
        for field in MINECRAFT_KEYMAP
        if field in correct_actions and np.asarray(correct_actions[field]).astype(bool).any()
    ]
    for field in active_fields:
        dropped = _copy_actions(correct_actions)
        dropped[field] = np.zeros_like(np.asarray(correct_actions[field]))
        candidates.append((f"drop_{field}", dropped))
        delayed = _copy_actions(correct_actions)
        delayed[field] = np.zeros_like(np.asarray(correct_actions[field]))
        delayed[field][num_frames // 2 :] = 1
        candidates.append((f"delay_{field}", delayed))

    if not np.any(np.abs(camera) > 1e-9):
        for yaw in (-0.15, 0.15):
            injected = _copy_actions(correct_actions)
            injected["camera"][:, 1] = yaw
            candidates.append((f"inject_yaw_{yaw:+g}", injected))

    if randomizer is not None:
        randomizer.shuffle(candidates)
    return candidates


def _build_counterfactual_distractors(
    correct_actions: dict[str, np.ndarray],
    randomizer: random.Random,
) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    signatures = {serialized_action_signature(correct_actions)}
    actions: list[dict[str, np.ndarray]] = []
    transforms: list[str] = []
    candidates = counterfactual_candidates(correct_actions)
    groups = [
        [item for item in candidates if item[0].startswith(("reverse_", "swap_")) and "timeline" not in item[0]],
        [item for item in candidates if item[0].startswith(("scale_", "drop_", "inject_"))],
        [item for item in candidates if item[0].startswith(("reverse_timeline", "shift_", "delay_"))],
    ]
    for group in groups:
        randomizer.shuffle(group)
    ordered = [item for group in groups for item in group]
    for group in groups:
        for transform, candidate in group:
            signature = serialized_action_signature(candidate)
            if signature in signatures:
                continue
            signatures.add(signature)
            actions.append(candidate)
            transforms.append(transform)
            break
    if len(actions) == 3:
        return actions, transforms
    for transform, candidate in ordered:
        signature = serialized_action_signature(candidate)
        if signature in signatures:
            continue
        signatures.add(signature)
        actions.append(candidate)
        transforms.append(transform)
        if len(actions) == 3:
            return actions, transforms
    raise RuntimeError("无法生成三个互异的反事实干扰动作")


def _primary_magnitude_field(
    actions: dict[str, np.ndarray],
    action_type: str,
    randomizer: random.Random,
) -> str:
    preferred: dict[str, tuple[str, ...]] = {
        "locomotion": ("forward", "back", "left", "right", "sprint", "sneak"),
        "jump": ("jump",),
        "attack": ("attack",),
        "use": ("use",),
        "inventory_hotbar": (
            "inventory", "drop", *(f"hotbar.{index}" for index in range(1, 10)),
        ),
        "move_attack": ("forward", "back", "left", "right", "attack"),
        "move_use": ("forward", "back", "left", "right", "use"),
        "complex": tuple(MINECRAFT_KEYMAP),
    }
    active = [
        field
        for field in preferred.get(action_type, ())
        if field in actions and np.asarray(actions[field]).astype(bool).any()
    ]
    if not active:
        raise ValueError(f"动作类型 {action_type!r} 没有可调整的按键字段")
    return randomizer.choice(active)


def _adjust_binary_duration(values: np.ndarray, target_count: int) -> np.ndarray:
    """保持动作大致中心位置，把二值按键持续帧数改为 target_count。"""
    source = np.asarray(values)
    num_frames = source.shape[0]
    if not 0 <= target_count <= num_frames:
        raise ValueError("目标持续帧数越界")
    active = np.flatnonzero(source.astype(bool))
    center = float(active.mean()) if active.size else (num_frames - 1) / 2
    order = sorted(range(num_frames), key=lambda index: (abs(index - center), index))
    selected = set(order[:target_count])
    result = np.zeros_like(source)
    for index in selected:
        result[index] = 1
    return result


def build_magnitude_distractors(
    correct_actions: dict[str, np.ndarray],
    action_type: str,
    correct_rank: int,
    randomizer: random.Random,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """构造四档对称程度序列，真实动作的档位由 correct_rank 指定。"""
    if correct_rank not in range(4):
        raise ValueError("correct_rank 必须为 0、1、2 或 3")
    candidates: list[dict[str, np.ndarray]] = []
    if action_type == "camera_only":
        camera = np.asarray(correct_actions["camera"], dtype=np.float64)
        totals = camera.sum(axis=0)
        axis = int(np.argmax(np.abs(totals)))
        base_total = float(totals[axis])
        direction = 1.0 if base_total >= 0 else -1.0
        base_magnitude = abs(base_total)
        levels = [base_magnitude + (rank - correct_rank) * 5.0 for rank in range(4)]
        if min(levels) < 0.0:
            raise ValueError("相机总转角不足以构造当前程度档位")
        for level in levels:
            candidate = _copy_actions(correct_actions)
            delta = direction * level - base_total
            candidate["camera"][:, axis] += delta / camera.shape[0]
            candidates.append(candidate)
        metadata = {
            "magnitude_kind": "camera_degrees",
            "target": "pitch" if axis == 0 else "yaw",
            "levels": [round(level, 6) for level in levels],
            "correct_rank": correct_rank,
            "step": 5.0,
        }
    else:
        field = _primary_magnitude_field(correct_actions, action_type, randomizer)
        values = np.asarray(correct_actions[field])
        base_count = int(values.astype(bool).sum())
        step = max(1, int(round(values.shape[0] * 0.25)))
        levels = [base_count + (rank - correct_rank) * step for rank in range(4)]
        if min(levels) < 0 or max(levels) > values.shape[0] or len(set(levels)) != 4:
            raise ValueError("按键持续帧数不足以构造当前程度档位")
        for rank, level in enumerate(levels):
            candidate = _copy_actions(correct_actions)
            if rank != correct_rank:
                candidate[field] = _adjust_binary_duration(values, level)
            candidates.append(candidate)
        metadata = {
            "magnitude_kind": "key_duration_frames",
            "target": field,
            "levels": levels,
            "correct_rank": correct_rank,
            "step": step,
        }
    correct_signature = serialized_action_signature(correct_actions)
    if serialized_action_signature(candidates[correct_rank]) != correct_signature:
        raise RuntimeError("程度候选的正确档没有保持原始动作")
    signatures = [serialized_action_signature(candidate) for candidate in candidates]
    if len(set(signatures)) != 4:
        raise ValueError("程度候选四档在公开表示中不唯一")
    return [candidate for rank, candidate in enumerate(candidates) if rank != correct_rank], metadata


def build_action_choice_benchmark(
    dataset_directory: Path,
    output_directory: Path,
    sample_count: int = 100,
    minimum_gap: int = 12,
    maximum_gap: int = 40,
    seed: int = 3407,
    image_width: int = 640,
    image_height: int = 360,
    episode_file: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """生成四选一测试集，公开题目和私有答案分别写入 JSONL。"""
    if sample_count < 1:
        raise ValueError("sample_count 必须大于零")
    if minimum_gap < 1 or maximum_gap < minimum_gap:
        raise ValueError("帧间隔必须满足 1 <= minimum_gap <= maximum_gap")
    output_directory = _prepare_output(output_directory, overwrite)
    randomizer = random.Random(seed)
    answer_schedule = [CHOICE_LABELS[index % len(CHOICE_LABELS)] for index in range(sample_count)]
    randomizer.shuffle(answer_schedule)
    action_type_schedule = [
        DETAILED_ACTION_TYPES[index % len(DETAILED_ACTION_TYPES)]
        for index in range(sample_count)
    ]
    randomizer.shuffle(action_type_schedule)
    rank_remaining = {
        rank: sample_count // 4 + (1 if rank < sample_count % 4 else 0)
        for rank in range(4)
    }
    reader = TrajectoryReader(
        [dataset_directory],
        ["action", "image"],
        frame_width=image_width,
        frame_height=image_height,
    )
    questions_path = output_directory / "questions.jsonl"
    answers_path = output_directory / "answer_key.jsonl"
    gap_counts: dict[int, int] = {}
    action_type_counts: dict[str, int] = {}
    try:
        episodes = _load_episode_subset(episode_file, reader.episode_names())
        with questions_path.open("w", encoding="utf-8") as questions_file, answers_path.open(
            "w", encoding="utf-8"
        ) as answers_file:
            for sample_index in range(sample_count):
                target_action_type = action_type_schedule[sample_index]
                for _ in range(10_000):
                    frame_gap = randomizer.randint(minimum_gap, maximum_gap)
                    correct_location = _random_location(reader, episodes, frame_gap, randomizer)
                    correct_actions = _read_action(reader, correct_location)
                    if classify_detailed_action_type(correct_actions) != target_action_type:
                        continue
                    available_ranks = [rank for rank, remaining in rank_remaining.items() if remaining]
                    randomizer.shuffle(available_ranks)
                    built = None
                    for target_correct_rank in available_ranks:
                        try:
                            built = build_magnitude_distractors(
                                correct_actions,
                                target_action_type,
                                target_correct_rank,
                                randomizer,
                            )
                        except ValueError:
                            continue
                        break
                    if built is None:
                        continue
                    distractors, magnitude_metadata = built
                    rank_remaining[target_correct_rank] -= 1
                    break
                else:
                    raise RuntimeError(f"无法抽到动作类型 {target_action_type!r} 的正确窗口")
                choices, answer = shuffled_choices(
                    correct_actions,
                    distractors,
                    randomizer,
                    correct_label=answer_schedule[sample_index],
                )
                frames = reader.readers["image"].read_frames(
                    correct_location.episode,
                    correct_location.start_frame,
                    frame_gap + 1,
                )
                sample_id = f"q{sample_index:06d}"
                before_relative = f"images/{sample_id}_before.jpg"
                after_relative = f"images/{sample_id}_after.jpg"
                Image.fromarray(frames[0]).save(output_directory / before_relative, quality=95)
                Image.fromarray(frames[-1]).save(output_directory / after_relative, quality=95)
                question = {
                    "id": sample_id,
                    "frame_gap": frame_gap,
                    "elapsed_seconds_at_20fps": round(frame_gap / 20, 3),
                    "image_before": before_relative,
                    "image_after": after_relative,
                    "choices": choices,
                }
                answer_record = {
                    "id": sample_id,
                    "answer": answer,
                    "correct_source": asdict(correct_location),
                    "correct_action_type": target_action_type,
                    "magnitude_design": magnitude_metadata,
                }
                questions_file.write(json.dumps(question, ensure_ascii=False) + "\n")
                answers_file.write(json.dumps(answer_record, ensure_ascii=False) + "\n")
                gap_counts[frame_gap] = gap_counts.get(frame_gap, 0) + 1
                action_type_counts[target_action_type] = action_type_counts.get(target_action_type, 0) + 1
    finally:
        reader.close()
    manifest = {
        "format": "minestudio_action_choice_v3_magnitude",
        "sample_count": sample_count,
        "seed": seed,
        "frame_gap_range": [minimum_gap, maximum_gap],
        "action_interval": "[before_frame, after_frame)",
        "distractor_mode": "symmetric_magnitude_levels",
        "action_type_policy": "balanced_detailed_round_robin",
        "correct_magnitude_rank_policy": "balanced_0_to_3",
        "action_type_counts": {
            key: action_type_counts.get(key, 0)
            for key in DETAILED_ACTION_TYPES
        },
        "image_size": [image_width, image_height],
        "episode_count": len(episodes),
        "gap_counts": {str(key): value for key, value in sorted(gap_counts.items())},
        "questions": questions_path.name,
        "answer_key": answers_path.name,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成两帧选四段动作的 MineStudio 测试集")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--min-gap", type=int, default=12)
    parser.add_argument("--max-gap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--episode-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    manifest = build_action_choice_benchmark(
        dataset_directory=arguments.dataset_dir,
        output_directory=arguments.output_dir,
        sample_count=arguments.samples,
        minimum_gap=arguments.min_gap,
        maximum_gap=arguments.max_gap,
        seed=arguments.seed,
        image_width=arguments.image_width,
        image_height=arguments.image_height,
        episode_file=arguments.episode_file,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
