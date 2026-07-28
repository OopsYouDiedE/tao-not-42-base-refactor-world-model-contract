"""构造 WASD 持续帧数与鼠标局部幅度的四选一程度测试。"""

from __future__ import annotations

import argparse
import json
import random
from itertools import permutations
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bc_datasets.minestudio.action_benchmark_common import (
    CHOICE_LABELS,
    copy_actions,
    load_episode_subset,
    prepare_output,
    random_location,
    read_action,
    serialized_action_signature,
    shuffled_choices,
)
from bc_datasets.minestudio.lmdb_modal_reader import TrajectoryReader

TARGET_TYPES = ("key_W", "key_A", "key_S", "key_D", "mouse_yaw", "mouse_pitch")
KEY_FIELDS = {"key_W": "forward", "key_A": "left", "key_S": "back", "key_D": "right"}
KEY_DELTAS = (-8, -4, 4, 8)
MOUSE_FACTORS = (1.25, 1.5)


def build_key_frame_distractors(
    actions: dict[str, np.ndarray],
    field: str,
    randomizer: random.Random,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """把目标按键总持续时间改变 -8/-4/+4/+8 帧中的三个合法值。"""
    if field not in actions:
        raise ValueError(f"动作缺少字段 {field!r}")
    values = np.asarray(actions[field])
    base_count = int(values.astype(bool).sum())
    inactive_count = values.shape[0] - base_count
    designs: list[tuple[tuple[int, int, int], int, tuple[int, int, int, int]]] = []
    maximum_blocks = min(values.shape[0] // 4, 12)
    for block_count in range(4, maximum_blocks + 1):
        for selected in permutations(KEY_DELTAS, 3):
            for shared_sum in range(-block_count, block_count + 1, 2):
                column_sums = (
                    shared_sum,
                    selected[0] - shared_sum,
                    selected[1] - shared_sum,
                    selected[2] - shared_sum,
                )
                if any(abs(total) > block_count or (block_count + total) % 2 for total in column_sums):
                    continue
                required_inactive = sum((block_count + total) // 2 for total in column_sums)
                required_active = 4 * block_count - required_inactive
                if required_inactive <= inactive_count and required_active <= base_count:
                    designs.append((selected, block_count, column_sums))
    if not designs:
        raise ValueError("目标按键无法构造四候选等距的 ±4/±8 扰动")
    selected_tuple, block_count, column_sums = randomizer.choice(designs)
    selected = list(selected_tuple)
    active_pool = np.flatnonzero(values.astype(bool)).tolist()
    inactive_pool = np.flatnonzero(~values.astype(bool)).tolist()
    randomizer.shuffle(active_pool)
    randomizer.shuffle(inactive_pool)
    columns: list[list[int]] = []
    for total in column_sums:
        inactive_needed = (block_count + total) // 2
        active_needed = block_count - inactive_needed
        column = [inactive_pool.pop() for _ in range(inactive_needed)]
        column.extend(active_pool.pop() for _ in range(active_needed))
        randomizer.shuffle(column)
        columns.append(column)
    masks = (
        columns[0] + columns[1],
        columns[0] + columns[2],
        columns[0] + columns[3],
    )
    distractors: list[dict[str, np.ndarray]] = []
    for mask in masks:
        candidate = copy_actions(actions)
        changed = np.array(values, copy=True)
        changed[mask] = 1 - changed[mask]
        candidate[field] = changed
        distractors.append(candidate)
    signatures = {serialized_action_signature(actions), *(serialized_action_signature(x) for x in distractors)}
    if len(signatures) != 4:
        raise ValueError("按键帧数扰动没有生成四个互异候选")
    return distractors, {
        "degree_kind": "key_total_pressed_frames",
        "target": field,
        "true_value": base_count,
        "deltas": selected,
        "candidate_values": [base_count + delta for delta in selected],
        "equal_hamming_distance": 2 * block_count,
        "equal_distance_code": True,
    }


def build_mouse_local_distractors(
    actions: dict[str, np.ndarray],
    axis: int,
    randomizer: random.Random,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """在部分非零帧上按等距符号模式乘或除同一系数。"""
    camera = np.asarray(actions["camera"], dtype=np.float64)
    active = np.flatnonzero(np.abs(camera[:, axis]) > 1e-9)
    if active.size < 8:
        raise ValueError("目标鼠标轴非零帧不足八个")
    selected_count = min((active.size // 4) * 4, 16)
    selected_positions = sorted(randomizer.sample(active.tolist(), selected_count))
    factor = randomizer.choice(MOUSE_FACTORS)
    patterns = (
        (factor, factor, 1.0 / factor, 1.0 / factor),
        (factor, 1.0 / factor, factor, 1.0 / factor),
        (factor, 1.0 / factor, 1.0 / factor, factor),
    )
    distractors: list[dict[str, np.ndarray]] = []
    multipliers_by_candidate: list[list[float]] = []
    for pattern in patterns:
        multipliers = [pattern[index % 4] for index in range(selected_count)]
        candidate = copy_actions(actions)
        candidate["camera"][selected_positions, axis] *= np.asarray(multipliers)
        distractors.append(candidate)
        multipliers_by_candidate.append(multipliers)
    signatures = {serialized_action_signature(actions), *(serialized_action_signature(x) for x in distractors)}
    if len(signatures) != 4:
        raise ValueError("鼠标幅度扰动没有生成四个互异候选")
    return distractors, {
        "degree_kind": "mouse_local_scale",
        "target": "pitch" if axis == 0 else "yaw",
        "selected_frame_offsets": selected_positions,
        "factor": factor,
        "multipliers_by_candidate": multipliers_by_candidate,
        "true_factor": 1.0,
        "equal_distance_code": True,
    }


def build_action_degree_benchmark(
    dataset_directory: Path,
    output_directory: Path,
    sample_count: int = 100,
    minimum_gap: int = 12,
    maximum_gap: int = 40,
    seed: int = 20260728,
    image_width: int = 320,
    image_height: int = 180,
    episode_file: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """生成六类近似均衡的 WASD/鼠标程度测试。"""
    if sample_count < 1:
        raise ValueError("sample_count 必须大于零")
    if minimum_gap < 1 or maximum_gap < minimum_gap:
        raise ValueError("帧间隔范围非法")
    output_directory = prepare_output(output_directory, overwrite)
    randomizer = random.Random(seed)
    target_schedule = [TARGET_TYPES[index % len(TARGET_TYPES)] for index in range(sample_count)]
    answer_schedule = [CHOICE_LABELS[index % 4] for index in range(sample_count)]
    randomizer.shuffle(target_schedule)
    randomizer.shuffle(answer_schedule)
    reader = TrajectoryReader(
        [dataset_directory], ["action", "image"],
        frame_width=image_width, frame_height=image_height,
    )
    questions_path = output_directory / "questions.jsonl"
    answers_path = output_directory / "answer_key.jsonl"
    target_counts: dict[str, int] = {}
    gap_counts: dict[int, int] = {}
    try:
        episodes = load_episode_subset(episode_file, reader.episode_names())
        with questions_path.open("w", encoding="utf-8") as question_file, answers_path.open(
            "w", encoding="utf-8"
        ) as answer_file:
            for sample_index, target_type in enumerate(target_schedule):
                for _ in range(20_000):
                    frame_gap = randomizer.randint(minimum_gap, maximum_gap)
                    location = random_location(reader, episodes, frame_gap, randomizer)
                    actions = read_action(reader, location)
                    try:
                        if target_type in KEY_FIELDS:
                            distractors, degree_design = build_key_frame_distractors(
                                actions, KEY_FIELDS[target_type], randomizer
                            )
                        else:
                            axis = 1 if target_type == "mouse_yaw" else 0
                            distractors, degree_design = build_mouse_local_distractors(
                                actions, axis, randomizer
                            )
                    except ValueError:
                        continue
                    break
                else:
                    raise RuntimeError(f"无法抽到目标类型 {target_type!r} 的合法窗口")
                choices, answer = shuffled_choices(
                    actions, distractors, randomizer,
                    correct_label=answer_schedule[sample_index],
                )
                frames = reader.readers["image"].read_frames(
                    location.episode, location.start_frame, frame_gap + 1,
                )
                sample_id = f"q{sample_index:06d}"
                before = f"images/{sample_id}_before.jpg"
                after = f"images/{sample_id}_after.jpg"
                Image.fromarray(frames[0]).save(output_directory / before, quality=95)
                Image.fromarray(frames[-1]).save(output_directory / after, quality=95)
                question_file.write(json.dumps({
                    "id": sample_id,
                    "frame_gap": frame_gap,
                    "elapsed_seconds_at_20fps": round(frame_gap / 20, 3),
                    "image_before": before,
                    "image_after": after,
                    "choices": choices,
                }, ensure_ascii=False) + "\n")
                answer_file.write(json.dumps({
                    "id": sample_id,
                    "answer": answer,
                    "correct_source": asdict(location),
                    "target_type": target_type,
                    "degree_design": degree_design,
                }, ensure_ascii=False) + "\n")
                target_counts[target_type] = target_counts.get(target_type, 0) + 1
                gap_counts[frame_gap] = gap_counts.get(frame_gap, 0) + 1
    finally:
        reader.close()
    manifest = {
        "format": "minestudio_action_choice_v4_wasd_mouse_degree",
        "sample_count": sample_count,
        "seed": seed,
        "frame_gap_range": [minimum_gap, maximum_gap],
        "action_interval": "[before_frame, after_frame)",
        "target_policy": "balanced_W_A_S_D_mouse_yaw_mouse_pitch",
        "target_counts": {name: target_counts.get(name, 0) for name in TARGET_TYPES},
        "key_frame_deltas": list(KEY_DELTAS),
        "mouse_local_factors": list(MOUSE_FACTORS),
        "gap_counts": {str(key): value for key, value in sorted(gap_counts.items())},
        "questions": questions_path.name,
        "answer_key": answers_path.name,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 WASD 持续帧数与鼠标局部幅度测试")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--min-gap", type=int, default=12)
    parser.add_argument("--max-gap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--episode-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_action_degree_benchmark(
        args.dataset_dir, args.output_dir, args.samples, args.min_gap, args.max_gap,
        args.seed, args.image_width, args.image_height, args.episode_file, args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
