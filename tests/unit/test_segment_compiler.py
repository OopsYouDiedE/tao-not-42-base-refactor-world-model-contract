# -*- coding: utf-8 -*-
"""决策段编译器单测：同一段在鼠标键盘与手柄 profile 上展开成不同设备形态。"""
import pytest

from control_contract.decision_segment import (
    Aim,
    ControlState,
    Guard,
    GuardComparison,
    Hold,
    Move,
    PIXEL_CHANGE_CHANNEL,
    Point,
    Press,
    Release,
    Segment,
    Select,
    Step,
    TailPolicy,
)
from control_contract.profile_registry import available_profile_names, load_named_profile
from control_contract.segment_compiler import compile_segment, compile_tail


@pytest.fixture
def mouse_keyboard():
    return load_named_profile("minecraft_mouse_keyboard")


@pytest.fixture
def gamepad():
    return load_named_profile("generic_gamepad")


def test_profiles_are_discoverable():
    """内置 profile 应可枚举，新增游戏只加 JSON。"""
    names = available_profile_names()
    assert "minecraft_mouse_keyboard" in names
    assert "generic_gamepad" in names


def test_milliseconds_map_to_ticks_by_profile(mouse_keyboard, gamepad):
    """同一毫秒时长在不同步频下展开成不同 tick 数，大模型无需感知 tick。"""
    segment = Segment(steps=[Step(duration_ms=1000)])
    assert compile_segment(segment, mouse_keyboard).total_ticks == 20   # 20 Hz
    assert compile_segment(segment, gamepad).total_ticks == 30          # 30 Hz


def test_displacement_aim_front_loads(mouse_keyboard):
    """位移式视角（鼠标）尽量前置转完，总量守恒。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=500, aim=Aim(yaw_deg=30.0))]), mouse_keyboard)
    yaws = [frame.aim_yaw_deg for frame in compiled.frames]
    assert yaws[0] == pytest.approx(18.0)   # aim_max_deg_per_tick
    assert yaws[1] == pytest.approx(12.0)
    assert sum(yaws) == pytest.approx(30.0)
    assert compiled.aim_truncation_deg == (0.0, 0.0)


def test_slower_aim_axis_needs_more_ticks(gamepad):
    """手柄瞄准摇杆上限更低，同样角度要占更多 tick，但总量同样守恒。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=500, aim=Aim(yaw_deg=60.0))]), gamepad)
    yaws = [frame.aim_yaw_deg for frame in compiled.frames]
    per_tick = 220.0 / 30.0
    assert yaws[0] == pytest.approx(per_tick)
    assert sum(yaws) == pytest.approx(60.0)
    # 60 度 ÷ 7.33 度/tick ≈ 9 个 tick 才转完，其余 tick 归零
    assert len([value for value in yaws if value > 0]) == 9


def test_aim_truncation_is_accounted(gamepad):
    """步太短转不完时按上限截断并记账，不静默假装转到位。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=100, aim=Aim(yaw_deg=180.0))]), gamepad)
    applied = sum(frame.aim_yaw_deg for frame in compiled.frames)
    assert applied == pytest.approx(220.0 * 0.1)   # 220 度/秒 × 0.1 秒
    assert compiled.aim_truncation_deg[0] == pytest.approx(180.0 - applied)


def test_digital_movement_quantises_to_octant(mouse_keyboard):
    """键盘 profile 把任意方向量化到 8 向。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=200, move=Move(direction_deg=40.0, power=0.5))]),
        mouse_keyboard)
    frame = compiled.frames[0]
    assert (frame.move_x, frame.move_y) == (1.0, 1.0)   # 45° → 前 + 右


def test_analog_movement_keeps_power(gamepad):
    """手柄 profile 保留连续力度与精确方向。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=200, move=Move(direction_deg=90.0, power=0.4))]),
        gamepad)
    frame = compiled.frames[0]
    assert frame.move_x == pytest.approx(0.4)
    assert frame.move_y == pytest.approx(0.0, abs=1e-9)


def test_dead_zone_suppresses_movement(gamepad):
    """力度低于死区不产生位移。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=200, move=Move(direction_deg=0.0, power=0.05))]),
        gamepad)
    assert (compiled.frames[0].move_x, compiled.frames[0].move_y) == (0.0, 0.0)


def test_digital_high_power_adds_sprint(mouse_keyboard):
    """键盘无法表达力度，高力度自动附加 sprint 近似。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=200, move=Move(direction_deg=0.0, power=1.0))]),
        mouse_keyboard)
    assert "sprint" in compiled.frames[0].pressed_roles


def test_direct_index_select_jumps(mouse_keyboard):
    """键盘 profile 直达槽位，一 tick 完成。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=200, select=Select(slot=7))]), mouse_keyboard)
    assert compiled.frames[0].select_slot == 7
    assert compiled.end_state.current_slot == 7


def test_cycle_only_select_expands_to_shortest_direction(gamepad):
    """手柄 profile 无直达键，展开为最短方向的 next/prev 点按。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=400, select=Select(slot=3))]),
        gamepad, ControlState(current_slot=1))
    pressed = [frame.pressed_roles for frame in compiled.frames]
    assert "next" in pressed[0] and "next" in pressed[2]
    assert all(frame.select_slot is None for frame in compiled.frames)
    assert compiled.end_state.current_slot == 3
    # 8 槽位环上 1 → 7 走 prev 更短
    backward = compile_segment(
        Segment(steps=[Step(duration_ms=400, select=Select(slot=7))]),
        gamepad, ControlState(current_slot=1))
    assert "prev" in backward.frames[0].pressed_roles


def test_latch_persists_across_steps_and_release_clears(mouse_keyboard):
    """hold 跨步延续，release 后不再按下。"""
    compiled = compile_segment(Segment(steps=[
        Step(duration_ms=100, holds=[Hold(role="primary")]),
        Step(duration_ms=100),
        Step(duration_ms=100, releases=[Release(role="primary")]),
    ]), mouse_keyboard)
    boundaries = compiled.step_boundary_ticks
    assert "primary" in compiled.frames[boundaries[0] - 1].pressed_roles
    assert "primary" in compiled.frames[boundaries[1] - 1].pressed_roles
    assert "primary" not in compiled.frames[boundaries[2] - 1].pressed_roles
    assert compiled.end_state.latched_roles == frozenset()


def test_latch_carries_into_next_segment(mouse_keyboard):
    """上一段结束时的 latch 作为下一段初始状态延续（跨推理边界）。"""
    first = compile_segment(
        Segment(steps=[Step(duration_ms=100, holds=[Hold(role="primary")])]), mouse_keyboard)
    second = compile_segment(
        Segment(steps=[Step(duration_ms=100)]), mouse_keyboard, first.end_state)
    assert "primary" in second.frames[0].pressed_roles


def test_press_produces_edge(mouse_keyboard):
    """点按只在部分 tick 为真，产生按下边沿而非持续按住。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=300, presses=[Press(role="jump", repeat=2)])]),
        mouse_keyboard)
    jump_ticks = [index for index, frame in enumerate(compiled.frames)
                  if "jump" in frame.pressed_roles]
    assert jump_ticks == [0, 2]


def test_rate_limited_cursor_approaches_gradually(gamepad):
    """上限较小的光标按速度逐 tick 逼近，长距离一步到不了并如实记账。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=100, point=Point(x=1.0, y=1.0))]),
        gamepad, ControlState(cursor_x=0.0, cursor_y=0.0))
    assert not compiled.cursor_reached
    assert compiled.end_state.cursor_x < 1.0
    positions = [frame.cursor_x for frame in compiled.frames]
    assert positions == sorted(positions)      # 单调逼近


def test_rate_limited_cursor_reaches_with_enough_time(gamepad):
    """给足时长后光标到达目标。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=2000, point=Point(x=0.8, y=0.2))]),
        gamepad, ControlState(cursor_x=0.5, cursor_y=0.5))
    assert compiled.cursor_reached
    assert compiled.end_state.cursor_x == pytest.approx(0.8)
    assert compiled.end_state.cursor_y == pytest.approx(0.2)


def test_guard_sustain_converted_to_ticks(mouse_keyboard):
    """守卫的 sustain 毫秒按步频换算为 tick 数，至少 1。"""
    compiled = compile_segment(Segment(
        steps=[Step(duration_ms=1000)],
        guards=[
            Guard(channel=PIXEL_CHANGE_CHANNEL, comparison=GuardComparison.BELOW,
                  threshold=0.01, sustain_ms=500, label="stuck"),
            Guard(channel=PIXEL_CHANGE_CHANNEL, comparison=GuardComparison.ABOVE,
                  threshold=0.3, sustain_ms=0),
        ],
    ), mouse_keyboard)
    assert compiled.guard_plans[0].sustain_ticks == 10
    assert compiled.guard_plans[1].sustain_ticks == 1


def test_tail_hold_keeps_latch_and_motion(mouse_keyboard):
    """HOLD 尾策略在推理延迟窗口内继续按住并继续位移。"""
    frames, end_state = compile_tail(
        TailPolicy.HOLD, 200, ControlState(latched_roles=frozenset({"primary", "sprint"})),
        mouse_keyboard, move_vector=(0.0, 1.0))
    assert len(frames) == 4
    assert frames[0].pressed_roles == frozenset({"primary", "sprint"})
    assert frames[0].move_y == 1.0
    assert end_state.latched_roles == frozenset()   # 租约耗尽后强制释放


def test_tail_release_move_stops_but_keeps_buttons(mouse_keyboard):
    """RELEASE_MOVE 停下位移但保留其他按住。"""
    frames, _ = compile_tail(
        TailPolicy.RELEASE_MOVE, 100,
        ControlState(latched_roles=frozenset({"primary", "sprint"})),
        mouse_keyboard, move_vector=(0.0, 1.0))
    assert frames[0].pressed_roles == frozenset({"primary"})
    assert frames[0].move_y == 0.0


def test_tail_release_all_is_neutral(mouse_keyboard):
    """RELEASE_ALL 立即回到中性态。"""
    frames, _ = compile_tail(
        TailPolicy.RELEASE_ALL, 100,
        ControlState(latched_roles=frozenset({"primary"})), mouse_keyboard)
    assert all(frame.pressed_roles == frozenset() for frame in frames)


def test_zero_lease_yields_no_tail_frames(mouse_keyboard):
    """租约为 0 时不产生尾帧（立即安全态）。"""
    frames, _ = compile_tail(
        TailPolicy.HOLD, 0, ControlState(latched_roles=frozenset({"primary"})), mouse_keyboard)
    assert frames == []


def test_unsupported_primitive_raises():
    """无界面光标的游戏上 point 原语在编译期报错，调用方据此要求重新决策。"""
    from control_contract.binding_profile import BindingProfile

    focus_profile = BindingProfile(profile_name="console_ui", menu_cursor=False)
    with pytest.raises(ValueError, match="没有光标"):
        compile_segment(
            Segment(steps=[Step(duration_ms=200, point=Point(x=0.5, y=0.5))]), focus_profile)


def test_compiler_resolves_ability_alias(mouse_keyboard):
    """手写段（未过 codec）里的游戏能力名由编译器解析为核心角色。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=100, holds=[Hold(role="mine")])]), mouse_keyboard)
    assert compiled.frames[0].pressed_roles == frozenset({"primary"})


def test_compiler_rejects_unknown_role(mouse_keyboard):
    """程序调用方写错角色名时编译期报错（与 codec 的宽容丢弃互补）。"""
    with pytest.raises(ValueError, match="未知角色"):
        compile_segment(
            Segment(steps=[Step(duration_ms=100, holds=[Hold(role="teleport")])]), mouse_keyboard)


def test_compilation_is_deterministic(mouse_keyboard):
    """同一 (segment, profile, state) 三元组的编译结果逐字段一致。"""
    segment = Segment(steps=[
        Step(duration_ms=300, aim=Aim(yaw_deg=25.0), move=Move(direction_deg=90.0, power=0.9),
             holds=[Hold(role="primary")]),
        Step(duration_ms=150, presses=[Press(role="jump")], select=Select(slot=4)),
    ])
    first = compile_segment(segment, mouse_keyboard)
    second = compile_segment(segment, mouse_keyboard)
    assert first.frames == second.frames
    assert first.end_state == second.end_state
