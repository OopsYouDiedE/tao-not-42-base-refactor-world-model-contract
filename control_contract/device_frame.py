# -*- coding: utf-8 -*-
"""设备无关的逐 tick 输入帧：编译器的输出、各环境 adapter 的唯一输入。

对外接口：
    DeviceFrame — 单 tick 的完整设备输入（视角增量、位移向量、光标、按下角色集、槽位、文本）。
    build_neutral_frame — 全中性帧。

设计要点：DeviceFrame 只表达**两个摇杆的本 tick 取值 + 按下的语义角色**，不含任何游戏键名。
鼠标键盘 adapter 把 ``aim_yaw_deg`` 换成鼠标计数、把 ``move_x/move_y`` 换成方向键；手柄
adapter 把它们分别换成右摇杆与左摇杆偏转。量化已在编译期按 profile 完成（键盘拿到的
``move_x/move_y`` 只会是 {-1, 0, 1}），adapter 只做最后一层物理映射。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceFrame:
    """单 tick 的设备无关输入。

    Attributes
    ----------
    aim_yaw_deg, aim_pitch_deg : float
        本 tick 施加的视角增量（度）。RATE 设备由 adapter 再除以角速度上限得到摇杆量。
    move_x, move_y : float
        位移向量，范围 [-1, 1]。move_y 正为前，move_x 正为右（角色坐标系）。
        键式位移摇杆（WASD / dpad）的 profile 下取值已被量化为 {-1, 0, 1}。
    cursor_x, cursor_y : Optional[float]
        本 tick 期望的归一化光标位置；None 表示本 tick 不驱动光标。编译器已按 profile 的
        单 tick 上限把长距离拆成多 tick，因此 adapter 只需把光标送到该位置：能绝对定位的
        后端直接置位，只能给增量的后端取与上一 tick 的差值。
    pressed_roles : frozenset[str]
        本 tick 处于按下状态的核心角色（含 latch 与本 tick 的点按）。
    select_slot : Optional[int]
        本 tick 要直达的槽位（仅 DIRECT_INDEX 设备产出；CYCLE_ONLY 已展开为 next/prev
        角色点按）。
    text : Optional[str]
        本 tick 要输入的文本。
    """

    aim_yaw_deg: float = 0.0
    aim_pitch_deg: float = 0.0
    move_x: float = 0.0
    move_y: float = 0.0
    cursor_x: Optional[float] = None
    cursor_y: Optional[float] = None
    pressed_roles: frozenset = frozenset()
    select_slot: Optional[int] = None
    text: Optional[str] = None


def build_neutral_frame() -> DeviceFrame:
    """构造一个全中性帧（无视角增量、无位移、无按下角色）。"""
    return DeviceFrame()
