"""变长动作段与图像—动作对齐校验。

动作仍以 MineStudio 的 20 Hz tick 为最小执行单位。数据集可以把连续且稳定的 tick
压缩成动作段；展开后必须与观测图像逐帧一一对应，任何长度不一致的样本都拒绝进入训练集。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionSegment:
    """一段连续执行的动作。``duration_ticks`` 必须为正。"""

    duration_ticks: int
    keys: tuple[str, ...] = ()
    mouse: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.duration_ticks < 1:
            raise ValueError("duration_ticks 必须大于零")


def action_ticks(actions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """把逐帧动作归一化为可比较的 key/mouse tick。"""
    camera = np.asarray(actions["camera"])
    ticks: list[dict[str, Any]] = []
    key_fields = ("forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use")
    names = {
        "forward": "W",
        "back": "S",
        "left": "A",
        "right": "D",
        "jump": "space",
        "sneak": "shift",
        "sprint": "ctrl",
        "attack": "MouseLeft",
        "use": "MouseRight",
    }
    for index, (pitch, yaw) in enumerate(camera):
        keys = tuple(
            names[field] for field in key_fields if field in actions and bool(actions[field][index])
        )
        ticks.append({"keys": keys, "mouse": (int(yaw), int(pitch))})
    return ticks


def compress_action_ticks(ticks: list[dict[str, Any]]) -> list[ActionSegment]:
    """合并相邻相同 tick；不规则鼠标变化会自然形成独立动作段。"""
    if not ticks:
        return []
    segments: list[ActionSegment] = []
    for tick in ticks:
        keys = tuple(tick["keys"])
        mouse = tuple(tick["mouse"])
        if segments and segments[-1].keys == keys and segments[-1].mouse == mouse:
            previous = segments[-1]
            segments[-1] = ActionSegment(previous.duration_ticks + 1, keys, mouse)
        else:
            segments.append(ActionSegment(1, keys, mouse))
    return segments


def expand_action_segments(segments: list[ActionSegment]) -> list[dict[str, Any]]:
    """展开动作段，作为写入编码器前的唯一逐帧表示。"""
    ticks: list[dict[str, Any]] = []
    for segment in segments:
        ticks.extend(
            {"keys": segment.keys, "mouse": segment.mouse} for _ in range(segment.duration_ticks)
        )
    return ticks


def validate_action_image_alignment(
    actions: dict[str, np.ndarray],
    images: np.ndarray,
    *,
    expected_frames: int | None = None,
) -> dict[str, Any]:
    """验证动作帧和图片帧数量相等，并返回可审计统计。"""
    action_length = len(np.asarray(actions["camera"]))
    image_array = np.asarray(images)
    if image_array.ndim < 1:
        raise ValueError("images 必须至少包含帧维度")
    image_length = image_array.shape[0]
    if expected_frames is not None and action_length != expected_frames:
        raise ValueError(f"动作帧数 {action_length} 不等于期望值 {expected_frames}")
    if action_length != image_length:
        raise ValueError(f"动作帧数 {action_length} 与图片帧数 {image_length} 不一致")
    if action_length < 1:
        raise ValueError("动作和图片不能为空")
    segments = compress_action_ticks(action_ticks(actions))
    return {
        "frames": action_length,
        "duration_ms": action_length * 50,
        "segments": len(segments),
        "irregular_mouse_frames": sum(
            segment.duration_ticks == 1 and segment.mouse != (0, 0) for segment in segments
        ),
    }
