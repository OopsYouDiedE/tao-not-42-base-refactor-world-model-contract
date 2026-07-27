# -*- coding: utf-8 -*-
"""界面/JSON 宏描述 → MacroCommand 实例的转换（无重依赖，可单测）。

从 trajectory_recorder_server 抽出：server 顶部 import cv2/craftground（起 env 才需要），
但"把宏 spec 转成 MacroCommand"是纯逻辑，不该被那些重依赖拖住可测性。本模块只依赖
macro_action_compiler，故可在 CPU/Windows 上直接单测。

支持的 spec：
  - {"kind": "turn", "holds": {"forward": 40, ...}, "clicks": ["jump", "attack"],
     "release": ["sneak"], "camera_mode": "delta"|"screen"|"none",
     "delta_yaw": .., "delta_pitch": .., "screen_x": .., "screen_y": ..,
     "gui_cursor": bool, "wait_ticks": int}
  - {"kind": "mc", "command": "setblock ~ ~ ~1 crafting_table"}
"""
from __future__ import annotations

from typing import Dict

from rl_training_environments.craftground.macro_action_compiler import (
    CAMERA_NONE,
    MacroCommand,
    minecraft_command,
    turn,
)


def macro_from_dict(spec: Dict) -> MacroCommand:
    """把界面/JSON 的宏描述转成 MacroCommand 实例。

    Parameters
    ----------
    spec : Dict
        宏描述，必含 "kind"。字段校验（时长/键名/相机上限/空回合）在 Turn.__post_init__。

    Returns
    -------
    MacroCommand
        Turn 或 MinecraftCommand。

    Raises
    ------
    ValueError
        未知 kind，或 Turn/MinecraftCommand 构造校验失败。
    """
    kind = spec.get("kind")
    label = spec.get("label", "")
    if kind == "turn":
        holds = {str(key): int(ticks) for key, ticks in (spec.get("holds") or {}).items()}
        return turn(
            holds=holds,
            clicks=[str(key) for key in (spec.get("clicks") or [])],
            release=[str(key) for key in (spec.get("release") or [])],
            camera_mode=spec.get("camera_mode", CAMERA_NONE),
            delta_yaw=float(spec.get("delta_yaw", 0.0)),
            delta_pitch=float(spec.get("delta_pitch", 0.0)),
            screen_x=float(spec.get("screen_x", 0.0)),
            screen_y=float(spec.get("screen_y", 0.0)),
            gui_cursor=bool(spec.get("gui_cursor", False)),
            wait_ticks=int(spec.get("wait_ticks", 0)),
            label=label,
        )
    if kind == "mc":
        return minecraft_command(spec["command"], label=label)
    raise ValueError(f"未知宏类型 kind={kind!r}")
