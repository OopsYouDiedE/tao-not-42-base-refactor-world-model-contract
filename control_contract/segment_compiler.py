# -*- coding: utf-8 -*-
"""决策段 → 逐 tick 设备帧 + 守卫计划的确定性编译器（纯逻辑，无运行时依赖）。

对外接口：
    CompiledSegment — 编译产物：逐 tick 设备帧、守卫计划、步边界、结束状态。
    GuardPlan — 守卫在 tick 尺度上的求值参数（sustain 换算成 tick 数）。
    compile_segment — Segment × BindingProfile × ControlState → CompiledSegment。
    compile_tail — 段末 tail 策略 + 租约 → 推理延迟窗口的逐 tick 帧。

编译规则要点：
  - 毫秒 → tick：``ceil(ms × tick_hz / 1000)``，至少 1 tick，避免出现零长度步。
  - 视角与光标共用 ``_advance_toward``：每 tick 朝目标推进至多 ``cap_per_tick``，推不完则
    **如实记账**（不静默假装转到位）。鼠标的"跳转到位置"只是上限足够大时的一 tick 特例，
    不是另一条代码路径。
  - 位移：极坐标 → 笛卡尔，量化交给 ``AxisSpec.quantise``；无力度档位的摇杆按 sprint 阈值
    自动附加 sprint 角色作为近似。
  - 槽位：有直达按键则直接置位，否则展开为最短方向的 next/prev 点按序列。
  - latch：Hold 写入状态并在后续 tick / 后续段持续为真，直到 Release / tail / 租约。
  - 角色：本层再解析一次 profile 的能力别名，因此手写录制脚本与 BC 目标可以直接写游戏
    能力名（如 ``mine``），无需先过 codec。

宽严分工：``segment_codec`` 是面向大模型的**宽容**边界（未知角色丢弃、越界截断、永不抛错）；
本模块是面向程序调用方的**严格**边界（未知角色与不支持的原语一律报错），二者互补。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from control_contract.binding_profile import BindingProfile
from control_contract.decision_segment import (
    ControlState,
    Guard,
    Segment,
    Step,
    TailPolicy,
)
from control_contract.device_frame import DeviceFrame

# 位移类角色：TailPolicy.RELEASE_MOVE 只松开这些。
MOVEMENT_ROLES = frozenset(("sprint", "crouch"))


@dataclass(frozen=True)
class GuardPlan:
    """守卫在 tick 尺度上的求值参数。

    Attributes
    ----------
    guard : Guard
        原始守卫声明（触发后原样写进 ExecutionReport 回灌大模型）。
    sustain_ticks : int
        条件需连续成立的 tick 数（由 sustain_ms 换算，至少 1）。
    """

    guard: Guard
    sustain_ticks: int


@dataclass
class CompiledSegment:
    """编译产物：可直接逐 tick 执行的设备帧序列与配套元信息。

    Attributes
    ----------
    frames : List[DeviceFrame]
        逐 tick 设备帧。运行时按序消费，每 tick 同时求值 guard_plans。
    guard_plans : List[GuardPlan]
        每 tick 求值的守卫计划。
    step_boundary_ticks : List[int]
        各步的结束 tick 下标（升序，最后一项 == len(frames)），用于记账"完整执行完几步"。
    end_state : ControlState
        本段全部 tick 执行完后的控制状态（latch / 光标 / 槽位）。
    aim_truncation_deg : Tuple[float, float]
        因步长不足而未能施加的视角增量（yaw, pitch），单位度。非零表示大模型给的步太短，
        运行时应记账并在下一轮报告中提示，不静默丢弃。
    cursor_reached : bool
        光标是否在本段内到达目标；False 表示步长不够、还差一截（能单 tick 直达的设备恒 True）。
    """

    frames: List[DeviceFrame] = field(default_factory=list)
    guard_plans: List[GuardPlan] = field(default_factory=list)
    step_boundary_ticks: List[int] = field(default_factory=list)
    end_state: ControlState = field(default_factory=ControlState)
    aim_truncation_deg: Tuple[float, float] = (0.0, 0.0)
    cursor_reached: bool = True

    @property
    def total_ticks(self) -> int:
        return len(self.frames)


def _ticks_for_ms(milliseconds: int, profile: BindingProfile) -> int:
    """毫秒 → tick 数，向上取整且至少 1。"""
    return max(1, math.ceil(milliseconds * profile.tick_hz / 1000.0))


def _advance_toward(
    remaining: float, tick_count: int, cap_per_tick: float,
) -> Tuple[List[float], float]:
    """把一个标量增量按单 tick 上限推进到 tick_count 个 tick 上（尽量前置）。

    这是**唯一**的摇杆推进函数：视角（被控量为度）与界面光标（被控量为归一化屏幕距离）
    共用它。所谓"跳转到位置"只是 cap_per_tick 足够大时它一 tick 就推完的极限情形，
    不需要单独的设备分支。

    Parameters
    ----------
    remaining : float
        要推进的总量（带符号）。
    tick_count : int
        可用 tick 数。
    cap_per_tick : float
        单 tick 上限（正数）。

    Returns
    -------
    per_tick : List[float]
        长度 == tick_count 的每 tick 增量。
    truncated : float
        tick 数不足而未能推进的剩余量（带符号，0 表示全部推完）。
    """
    per_tick: List[float] = []
    left = remaining
    for _ in range(max(0, tick_count)):
        piece = math.copysign(min(abs(left), cap_per_tick), left) if left else 0.0
        per_tick.append(piece)
        left -= piece
    return per_tick, left


def _locomotion_vector(
    direction_deg: float, power: float, profile: BindingProfile,
) -> Tuple[float, float, frozenset]:
    """极坐标位移意图 → (move_x, move_y, 自动附加的角色集)。

    量化交给 ``profile.locomotion_axis.quantise``：连续摇杆原样通过，键盘 WASD 被规整到
    8 向单档，二者走同一条代码路径。

    Returns
    -------
    move_x, move_y : float
        角色坐标系位移向量，move_y 正为前、move_x 正为右。
    extra_roles : frozenset[str]
        摇杆无法表达力度档位时，高力度自动附加 ``sprint`` 作为近似。
    """
    axis = profile.locomotion_axis
    direction, magnitude = axis.quantise(direction_deg, power)
    if magnitude == 0.0:
        return 0.0, 0.0, frozenset()
    radians = math.radians(direction)
    unit_x = math.sin(radians)
    unit_y = math.cos(radians)
    if axis.magnitude_levels == 1 and axis.direction_count in (4, 8):
        # 键式摇杆（WASD / dpad）：分量只能取 {-1, 0, 1}，对角线由两键同按合成。
        unit_x = round(unit_x)
        unit_y = round(unit_y)
    move_x = unit_x * magnitude
    move_y = unit_y * magnitude
    extra: frozenset = frozenset()
    if (not axis.is_continuous_magnitude
            and max(0.0, min(1.0, power)) >= profile.sprint_threshold
            and "sprint" not in profile.unavailable_roles):
        extra = frozenset(("sprint",))
    return float(move_x), float(move_y), extra


def _cursor_path(
    start: Tuple[float, float],
    target: Tuple[float, float],
    tick_count: int,
    profile: BindingProfile,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float], bool]:
    """规划光标在本步内的逐 tick 位置。

    与视角推进同一套逻辑：每 tick 沿直线朝目标推进至多 ``cursor_cap_per_tick``。上限大到
    覆盖整屏时首 tick 即到达（真实鼠标的"跳转位置"），上限小则逐 tick 逼近（手柄虚拟光标、
    或受相机增量上限约束的鼠标）——同一段代码同时表达两种设备。

    Returns
    -------
    path : List[Tuple[float, float]]
        长度 == tick_count 的逐 tick 目标光标位置。
    end_position : Tuple[float, float]
        本步结束时的光标位置。
    reached : bool
        是否到达目标。
    """
    step_distance = profile.cursor_cap_per_tick
    current_x, current_y = start
    path: List[Tuple[float, float]] = []
    reached = False
    for _ in range(tick_count):
        delta_x = target[0] - current_x
        delta_y = target[1] - current_y
        distance = math.hypot(delta_x, delta_y)
        if distance <= step_distance:
            current_x, current_y = target
            reached = True
        else:
            scale = step_distance / distance
            current_x += delta_x * scale
            current_y += delta_y * scale
        path.append((current_x, current_y))
    return path, (current_x, current_y), reached


def _cycle_roles_for_slot(
    current_slot: int, target_slot: int, profile: BindingProfile,
) -> List[str]:
    """无直达槽位按键时，从 current_slot 切到 target_slot 所需的逐 tick 点按角色序列。

    取环形最短方向；current_slot 为 0（未知）时按 next 逼近 1 起的绝对位置无从计算，
    退化为不切换（返回空列表），由大模型下轮据观测重新决定。
    """
    if profile.direct_slot_buttons or profile.slot_count <= 0 \
            or current_slot <= 0 or current_slot == target_slot:
        return []
    forward_steps = (target_slot - current_slot) % profile.slot_count
    backward_steps = (current_slot - target_slot) % profile.slot_count
    if forward_steps <= backward_steps:
        return ["next"] * forward_steps
    return ["prev"] * backward_steps


def _compile_step(
    step: Step, profile: BindingProfile, state: ControlState,
) -> Tuple[List[DeviceFrame], ControlState, Tuple[float, float], bool]:
    """把单个 Step 展开为逐 tick 设备帧，并返回更新后的状态与记账信息。"""
    tick_count = _ticks_for_ms(step.duration_ms, profile)
    latched = set(state.latched_roles)
    for release in step.releases:
        latched.discard(profile.resolve_role(release.role))
    for hold in step.holds:
        latched.add(profile.resolve_role(hold.role))

    cap = profile.aim_axis.cap_per_tick
    if step.aim is not None:
        yaw_path, yaw_left = _advance_toward(step.aim.yaw_deg, tick_count, cap)
        pitch_path, pitch_left = _advance_toward(step.aim.pitch_deg, tick_count, cap)
    else:
        yaw_path = [0.0] * tick_count
        pitch_path = [0.0] * tick_count
        yaw_left = pitch_left = 0.0

    if step.move is not None:
        move_x, move_y, move_roles = _locomotion_vector(
            step.move.direction_deg, step.move.power, profile)
    else:
        move_x = move_y = 0.0
        move_roles = frozenset()

    cursor_path: List[Optional[Tuple[float, float]]] = [None] * tick_count
    cursor_end = (state.cursor_x, state.cursor_y)
    cursor_reached = True
    if step.point is not None and profile.menu_cursor:
        cursor_path, cursor_end, cursor_reached = _cursor_path(
            (state.cursor_x, state.cursor_y), (step.point.x, step.point.y), tick_count, profile)

    # 点按：repeat 次按下摊在本步的前若干 tick 上，相邻两次之间空一个 tick 以产生边沿。
    press_by_tick: Dict[int, set] = {}
    for press in step.presses:
        role = profile.resolve_role(press.role)
        for repetition in range(press.repeat):
            index = repetition * 2
            if index < tick_count:
                press_by_tick.setdefault(index, set()).add(role)

    # 槽位：DIRECT_INDEX 首 tick 直达；CYCLE_ONLY 展开为逐 tick 的 next/prev 点按。
    select_slot: Optional[int] = None
    slot_after = state.current_slot
    if step.select is not None:
        if profile.direct_slot_buttons:
            select_slot = step.select.slot
            slot_after = step.select.slot
        else:
            cycle = _cycle_roles_for_slot(state.current_slot, step.select.slot, profile)
            for offset, role in enumerate(cycle):
                index = offset * 2
                if index < tick_count:
                    press_by_tick.setdefault(index, set()).add(role)
                else:
                    break
            consumed = min(len(cycle), (tick_count + 1) // 2)
            if consumed == len(cycle):
                slot_after = step.select.slot

    frames: List[DeviceFrame] = []
    for tick in range(tick_count):
        cursor = cursor_path[tick]
        frames.append(DeviceFrame(
            aim_yaw_deg=yaw_path[tick],
            aim_pitch_deg=pitch_path[tick],
            move_x=move_x,
            move_y=move_y,
            cursor_x=cursor[0] if cursor is not None else None,
            cursor_y=cursor[1] if cursor is not None else None,
            pressed_roles=frozenset(latched | move_roles | press_by_tick.get(tick, set())),
            select_slot=select_slot if tick == 0 else None,
            text=step.text.text if (step.text is not None and tick == 0) else None,
        ))
    end_state = ControlState(
        latched_roles=frozenset(latched),
        cursor_x=cursor_end[0], cursor_y=cursor_end[1],
        current_slot=slot_after,
    )
    return frames, end_state, (yaw_left, pitch_left), cursor_reached


def compile_segment(
    segment: Segment,
    profile: BindingProfile,
    state: Optional[ControlState] = None,
) -> CompiledSegment:
    """把一个决策段确定性地编译为逐 tick 设备帧与守卫计划。

    纯函数：不持有状态、不接触环境。同一 (segment, profile, state) 三元组永远产出逐字节
    相同的结果——这是"录制回放 = 在线执行 = BC 目标编码"三侧口径一致的基础。

    Parameters
    ----------
    segment : Segment
        大模型本轮输出的决策段（已由 segment_codec 保证结构合法）。
    profile : BindingProfile
        目标游戏 / 设备的能力声明。
    state : Optional[ControlState]
        进入本段时的控制状态（latch / 光标 / 槽位）；None 视为全新会话的中性状态。

    Returns
    -------
    CompiledSegment

    Raises
    ------
    ValueError
        段内出现本 profile 不支持的原语（界面无光标却用 point、无槽位游戏用 select、
        不支持文本的 profile 用 text）。这类错误说明 prompt 里的能力声明没被遵守，
        应由调用方捕获并要求重新决策，而不是静默丢弃动作。
    """
    current = state if state is not None else ControlState()
    frames: List[DeviceFrame] = []
    boundaries: List[int] = []
    truncated_yaw = 0.0
    truncated_pitch = 0.0
    cursor_reached = True
    for index, step in enumerate(segment.steps):
        if step.point is not None and not profile.menu_cursor:
            raise ValueError(
                f"step {index}: profile {profile.profile_name!r} 界面里没有光标，不支持 point "
                "原语（改用 nav_* + confirm）")
        if step.select is not None and profile.slot_count <= 0:
            raise ValueError(f"step {index}: profile {profile.profile_name!r} 没有槽位概念")
        if step.text is not None and not profile.supports_text:
            raise ValueError(f"step {index}: profile {profile.profile_name!r} 不支持文本输入")
        step_frames, current, truncation, reached = _compile_step(step, profile, current)
        frames.extend(step_frames)
        boundaries.append(len(frames))
        truncated_yaw += truncation[0]
        truncated_pitch += truncation[1]
        cursor_reached = cursor_reached and reached
    guard_plans = [
        GuardPlan(guard=guard, sustain_ticks=_ticks_for_ms(guard.sustain_ms, profile))
        for guard in segment.guards
    ]
    return CompiledSegment(
        frames=frames,
        guard_plans=guard_plans,
        step_boundary_ticks=boundaries,
        end_state=current,
        aim_truncation_deg=(truncated_yaw, truncated_pitch),
        cursor_reached=cursor_reached,
    )


def compile_tail(
    tail: TailPolicy,
    lease_ms: int,
    state: ControlState,
    profile: BindingProfile,
    move_vector: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[List[DeviceFrame], ControlState]:
    """编译推理延迟窗口的逐 tick 帧：段末到新段生效之间运行时该做什么。

    这是纯大模型控制不卡顿的关键：段跑完后运行时不会空转，而是按 tail 语义继续执行，
    最长 lease_ms。租约耗尽后所有帧回到中性（死人开关），避免"模型挂了角色还在往前跑"。

    Parameters
    ----------
    tail : TailPolicy
        段声明的兜底行为。
    lease_ms : int
        租约时长（毫秒）；0 表示立即回到中性。
    state : ControlState
        段结束时的控制状态。
    profile : BindingProfile
        目标 profile。
    move_vector : Tuple[float, float]
        段最后一 tick 的位移向量，HOLD 策略下继续沿用。

    Returns
    -------
    frames : List[DeviceFrame]
        租约期内的逐 tick 帧（长度 = lease_ms 换算的 tick 数；lease_ms==0 时为空）。
    end_state : ControlState
        租约耗尽后的控制状态（latch 已按策略清理）。
    """
    if lease_ms < 0:
        raise ValueError(f"lease_ms 不能为负，收到 {lease_ms}")
    if tail is TailPolicy.RELEASE_ALL:
        kept = frozenset()
        vector = (0.0, 0.0)
    elif tail is TailPolicy.RELEASE_MOVE:
        kept = frozenset(state.latched_roles) - MOVEMENT_ROLES
        vector = (0.0, 0.0)
    else:
        kept = frozenset(state.latched_roles)
        vector = move_vector
    tick_count = 0 if lease_ms == 0 else _ticks_for_ms(lease_ms, profile)
    frames = [
        DeviceFrame(move_x=vector[0], move_y=vector[1], pressed_roles=kept)
        for _ in range(tick_count)
    ]
    return frames, ControlState(
        latched_roles=frozenset(),
        cursor_x=state.cursor_x, cursor_y=state.cursor_y,
        current_slot=state.current_slot,
    )
