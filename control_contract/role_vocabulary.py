# -*- coding: utf-8 -*-
"""游戏无关的语义控制角色词表与设备模型枚举（控制契约的最底层定义）。

对外接口：
    CORE_ROLES — 跨游戏语义按钮角色，大模型只允许使用这些角色名或 profile 声明的
        ability 别名，词表本身不含任何具体游戏的键位或术语。
    PrimitiveName — 一个决策步内允许出现的原语通道名。
    is_core_role — 角色名合法性判定。

设计要点：角色是**语义**而非物理键。"primary" 在鼠标键盘上是左键、在手柄上是右扳机，
映射写在 profile 数据里，本模块与编译器都不知道任何具体游戏。

设备统一模型：任何游戏输入 = **两个抽象摇杆（位移 + 瞄准）+ 一堆语义按钮**。手柄是两个
连续摇杆；键鼠是 8 向单档的 WASD（位移摇杆）加一个上限极大的鼠标（瞄准摇杆，上限大到
可在单 tick 内跳到目标 = "跳转位置"）。因此设备差异不是族之分，只是 ``AxisSpec``（见
``binding_profile``）里几个数值不同，编译器只有一条代码路径。
"""
from __future__ import annotations

from enum import Enum

# 跨游戏语义按钮角色。选取标准：在动作、射击、生存、驾驶、策略与菜单交互中普遍存在，
# 且在鼠标键盘与手柄上都有自然对应物。游戏专有能力不进本词表，走 profile 的 ability 别名。
CORE_ROLES: tuple[str, ...] = (
    "primary",     # 主要动作：开火 / 攻击 / 挖掘 / 左键
    "secondary",   # 次要动作：瞄准 / 格挡 / 放置 / 右键
    "interact",    # 交互：拾取 / 开门 / 对话 / 使用
    "jump",        # 跳跃 / 上升 / 加速踏板
    "crouch",      # 下蹲 / 潜行 / 下降 / 刹车
    "sprint",      # 冲刺修饰
    "inventory",   # 打开物品栏
    "map",         # 打开地图
    "menu",        # 打开菜单 / 暂停
    "confirm",     # 界面确认
    "cancel",      # 界面取消 / 返回
    "nav_up",      # 界面方向导航（方向键 / dpad）
    "nav_down",
    "nav_left",
    "nav_right",
    "next",        # 循环切换下一个（滚轮下 / 右肩键）
    "prev",        # 循环切换上一个
    "aux1",        # 通用备用角色，由 profile 的 ability 别名赋予游戏语义
    "aux2",
    "aux3",
    "aux4",
)

_CORE_ROLE_SET = frozenset(CORE_ROLES)


def is_core_role(name: str) -> bool:
    """判断名字是否为核心角色。"""
    return name in _CORE_ROLE_SET


class PrimitiveName(str, Enum):
    """一个决策步内允许出现的原语通道名（同一步内各通道并发生效）。"""

    AIM = "aim"          # 相对视角增量（度）
    MOVE = "move"        # 极坐标位移意图（方向度 + 力度 0..1）
    POINT = "point"      # 界面指向（归一化屏幕坐标 0..1）
    PRESS = "press"      # 点按角色（本步内短按后松开）
    HOLD = "hold"        # 按住角色（跨步 latch，直到 release 或 lease 到期）
    RELEASE = "release"  # 松开被 latch 的角色
    SELECT = "select"    # 选择离散槽位（1 起）
    TEXT = "text"        # 文本输入（仅有键盘的 profile 支持）
