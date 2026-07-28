"""MineStudio 动作考题生成器共享的数据结构与基础操作。"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bc_datasets.minestudio.lmdb_modal_reader import TrajectoryReader
from bc_datasets.minestudio.lumine_action_codec import MINECRAFT_KEYMAP, encode_lumine_action

CHOICE_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class ActionLocation:
    """一段动作在原始数据中的位置。"""

    episode: str
    start_frame: int
    frame_gap: int


def serialize_action(actions: dict[str, np.ndarray]) -> dict[str, Any]:
    """把原始逐帧动作转换成不含来源信息的候选答案。"""
    encoded = encode_lumine_action(actions, frames_per_chunk=1)
    camera = np.asarray(actions["camera"], dtype=np.float64)
    frames: list[dict[str, Any]] = []
    for frame_index, (pitch, yaw) in enumerate(camera):
        held_keys = [
            key_name
            for field, key_name in MINECRAFT_KEYMAP.items()
            if field in actions and bool(np.asarray(actions[field])[frame_index])
        ]
        frames.append(
            {
                "camera_pitch_degrees": round(float(pitch), 6),
                "camera_yaw_degrees": round(float(yaw), 6),
                "held_keys": held_keys,
            }
        )
    return {
        "lumine_text": encoded.to_text(),
        "total_mouse_delta_pixels": {
            "x": encoded.mouse_delta_x,
            "y": encoded.mouse_delta_y,
        },
        "frames": frames,
    }


def serialized_action_signature(actions: dict[str, np.ndarray]) -> str:
    """签名最终公开内容，排除浮点舍入后相同的候选项。"""
    payload = json.dumps(serialize_action(actions), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def shuffled_choices(
    correct: dict[str, np.ndarray],
    distractors: Iterable[dict[str, np.ndarray]],
    randomizer: random.Random,
    correct_label: str | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    """打乱一个正确项与三个互异干扰项，返回公开选项和答案标签。"""
    candidates = [correct, *distractors]
    if len(candidates) != 4:
        raise ValueError("四选一必须恰好包含一个正确项和三个干扰项")
    signatures = [serialized_action_signature(candidate) for candidate in candidates]
    if len(set(signatures)) != 4:
        raise ValueError("四个候选动作必须互不相同")
    if correct_label is not None and correct_label not in CHOICE_LABELS:
        raise ValueError(f"correct_label 必须是 {CHOICE_LABELS} 之一")

    distractor_items = list(enumerate(candidates[1:], start=1))
    randomizer.shuffle(distractor_items)
    selected_label = correct_label or randomizer.choice(CHOICE_LABELS)
    correct_position = CHOICE_LABELS.index(selected_label)
    indexed_candidates = list(distractor_items)
    indexed_candidates.insert(correct_position, (0, correct))
    choices = {
        label: serialize_action(candidate)
        for label, (_, candidate) in zip(CHOICE_LABELS, indexed_candidates)
    }
    return choices, selected_label


def load_episode_subset(path: Path | None, available: list[str]) -> list[str]:
    """读取可选 episode 清单，并与当前数据集取交集。"""
    if path is None:
        return available
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = payload.get("validation_episodes") if isinstance(payload, dict) else payload
    if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
        raise ValueError("episode 文件应为字符串列表，或含 validation_episodes 的 split JSON")
    selected = sorted(set(available) & set(requested))
    if not selected:
        raise ValueError("episode 文件与 image/action 共同 episode 没有交集")
    return selected


def random_location(
    reader: TrajectoryReader,
    episodes: list[str],
    frame_gap: int,
    randomizer: random.Random,
    excluded_episode: str | None = None,
) -> ActionLocation:
    """随机选择一段能够容纳指定帧间隔的轨迹位置。"""
    pool = [episode for episode in episodes if episode != excluded_episode]
    for _ in range(10_000):
        episode = randomizer.choice(pool or episodes)
        maximum_start = reader.episode_length(episode) - frame_gap - 1
        if maximum_start >= 0:
            return ActionLocation(episode, randomizer.randint(0, maximum_start), frame_gap)
    raise RuntimeError(f"找不到可容纳 {frame_gap} 帧间隔的 episode")


def read_action(
    reader: TrajectoryReader,
    location: ActionLocation,
) -> dict[str, np.ndarray]:
    """读取一个位置对应的半开区间动作序列。"""
    return reader.readers["action"].read_frames(
        location.episode,
        location.start_frame,
        location.frame_gap,
    )


def copy_actions(actions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """深复制动作数组，供干扰项变换使用。"""
    return {field: np.array(value, copy=True) for field, value in actions.items()}


def prepare_output(output_directory: Path, overwrite: bool) -> Path:
    """创建生成目录；显式允许时安全覆盖已有目录。"""
    output_directory = output_directory.resolve()
    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在：{output_directory}；使用 --overwrite 覆盖")
        if output_directory.name in {"", ".", ".."} or output_directory.parent == output_directory:
            raise ValueError(f"拒绝清除不安全目录：{output_directory}")
        shutil.rmtree(output_directory)
    (output_directory / "images").mkdir(parents=True)
    return output_directory
