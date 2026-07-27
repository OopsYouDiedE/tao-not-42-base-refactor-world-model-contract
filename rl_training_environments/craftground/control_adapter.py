# -*- coding: utf-8 -*-
"""设备无关 DeviceFrame → CraftGround V2 动作 dict 的适配层（唯一知道 Minecraft 键名的地方）。

对外接口：
    ROLE_TO_V2_KEY — 语义角色 → V2 二值键的映射表。
    CURSOR_DEGREES_PER_SCREEN_WIDTH / _HEIGHT — 光标归一化位移 → 相机度数（两轴不同）。
    device_frame_to_v2_action — 单帧转换。
    device_frames_to_v2_actions — 序列转换（带光标状态跨帧延续）。

CraftGround 的 GUI 光标由 camera 字段驱动（见记忆 craftground-recorder-capability-probe），
没有绝对定位通道，因此 minecraft_mouse_keyboard profile 的 cursor_cap_per_tick 只有 0.2 屏
（而非桌面鼠标那种单 tick 直达）：编译器据此把 point 拆成逐 tick 的目标光标位置，本适配层
再把"相邻 tick 的位置差"换成相机增量。

本模块是 control_contract 的下游消费者，控制契约本身不依赖它，也不依赖 craftground 运行时。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from control_contract.device_frame import DeviceFrame
from rl_training_environments.craftground.action_contract import V2_KEYS
from rl_training_environments.craftground.segment_text_codec import (
    CURSOR_DEGREES_PER_PIXEL,
    CURSOR_SCREEN_HEIGHT_PIXELS,
    CURSOR_SCREEN_WIDTH_PIXELS,
)

# 语义角色 → CraftGround V2 二值键。profile 已把本游戏不存在的角色列入 unavailable_roles，
# 因此这里只需覆盖 minecraft_mouse_keyboard 声明为可用的角色。
ROLE_TO_V2_KEY: Dict[str, str] = {
    "primary": "attack",
    "secondary": "use",
    "interact": "use",
    "jump": "jump",
    "crouch": "sneak",
    "sprint": "sprint",
    "inventory": "inventory",
    "aux1": "drop",
}

# 光标：归一化屏幕单位 → 相机度数。实测标定（见 segment_text_codec 的常数注释）：
# 光标只走整数像素，1 px = 0.15°；两轴同为 6.667 px/度，折成屏幅后因宽高比不同而
# 不同（x 96°/屏、y 54°/屏），所以**不能共用一个常数**。
CURSOR_DEGREES_PER_SCREEN_WIDTH = (
    CURSOR_SCREEN_WIDTH_PIXELS * CURSOR_DEGREES_PER_PIXEL)      # 96.0
CURSOR_DEGREES_PER_SCREEN_HEIGHT = (
    CURSOR_SCREEN_HEIGHT_PIXELS * CURSOR_DEGREES_PER_PIXEL)     # 54.0

# 位移向量 → 方向键的判定阈值（键式位移摇杆的 profile 下向量已量化为 {-1,0,1}）。
_AXIS_THRESHOLD = 0.5

# 快捷栏槽位对应的 V2 键前缀（select_slot 取 1..9）。
_HOTBAR_PREFIX = "hotbar."


def _movement_keys(frame: DeviceFrame) -> Dict[str, bool]:
    """把量化位移向量转成 forward/back/left/right 四个键。"""
    keys = {"forward": False, "back": False, "left": False, "right": False}
    if frame.move_y > _AXIS_THRESHOLD:
        keys["forward"] = True
    elif frame.move_y < -_AXIS_THRESHOLD:
        keys["back"] = True
    if frame.move_x > _AXIS_THRESHOLD:
        keys["right"] = True
    elif frame.move_x < -_AXIS_THRESHOLD:
        keys["left"] = True
    return keys


def device_frame_to_v2_action(
    frame: DeviceFrame,
    previous_cursor: Optional[Tuple[float, float]],
) -> Tuple[Dict[str, object], Optional[Tuple[float, float]]]:
    """把单个 DeviceFrame 转成 CraftGround V2 动作 dict。

    Parameters
    ----------
    frame : DeviceFrame
        编译器产出的设备无关帧。
    previous_cursor : Optional[Tuple[float, float]]
        上一 tick 的虚拟光标位置；None 表示本 tick 之前光标未被驱动过（不产生相机增量）。

    Returns
    -------
    action : Dict[str, object]
        键集 = V2_KEYS ∪ {"camera_yaw", "camera_pitch"} 的完整动作 dict。
    next_cursor : Optional[Tuple[float, float]]
        本 tick 之后的光标位置，供下一 tick 调用时作为 previous_cursor 传入。
    """
    action: Dict[str, object] = {key: False for key in V2_KEYS}
    for role in frame.pressed_roles:
        key = ROLE_TO_V2_KEY.get(role)
        if key is not None:
            action[key] = True
    action.update(_movement_keys(frame))
    if frame.select_slot is not None:
        action[f"{_HOTBAR_PREFIX}{frame.select_slot}"] = True

    camera_yaw = frame.aim_yaw_deg
    camera_pitch = frame.aim_pitch_deg
    next_cursor = previous_cursor
    if frame.cursor_x is not None and frame.cursor_y is not None:
        if previous_cursor is not None:
            # 两轴分别换算（宽高比不同），并折成整数像素——底层只认整数像素，
            # 小数会被截断且不累积。
            pixels_x = round((frame.cursor_x - previous_cursor[0]) * CURSOR_SCREEN_WIDTH_PIXELS)
            pixels_y = round((frame.cursor_y - previous_cursor[1]) * CURSOR_SCREEN_HEIGHT_PIXELS)
            camera_yaw += pixels_x * CURSOR_DEGREES_PER_PIXEL
            camera_pitch += pixels_y * CURSOR_DEGREES_PER_PIXEL
        next_cursor = (frame.cursor_x, frame.cursor_y)
    action["camera_yaw"] = camera_yaw
    action["camera_pitch"] = camera_pitch
    return action, next_cursor


def device_frames_to_v2_actions(
    frames: Sequence[DeviceFrame],
    initial_cursor: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, object]]:
    """把一段 DeviceFrame 序列转成逐 tick 的 CraftGround V2 动作 dict 序列。

    Parameters
    ----------
    frames : Sequence[DeviceFrame]
        编译器产出的逐 tick 帧（通常是一个 Segment 的 compile_segment 结果）。
    initial_cursor : Optional[Tuple[float, float]]
        进入本序列时的虚拟光标位置；None 表示尚未建立光标基准。

    Returns
    -------
    List[Dict[str, object]]
        与 frames 等长的 V2 动作 dict 列表。
    """
    actions: List[Dict[str, object]] = []
    cursor = initial_cursor
    for frame in frames:
        action, cursor = device_frame_to_v2_action(frame, cursor)
        actions.append(action)
    return actions
