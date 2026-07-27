# -*- coding: utf-8 -*-
"""键鼠与手柄的统一摇杆模型单测：设备差异只是 AxisSpec 的数值差异，不是两条代码路径。"""
import pytest

from control_contract.binding_profile import (
    SCREEN_DIAGONAL,
    AxisSpec,
    BindingProfile,
    describe_capabilities,
)
from control_contract.decision_segment import Aim, ControlState, Point, Segment, Step
from control_contract.profile_registry import load_named_profile
from control_contract.segment_compiler import compile_segment


@pytest.fixture
def mouse_keyboard():
    return load_named_profile("minecraft_mouse_keyboard")


@pytest.fixture
def gamepad():
    return load_named_profile("generic_gamepad")


@pytest.fixture
def desktop():
    return load_named_profile("desktop_mouse_keyboard")


def test_continuous_axis_passes_through():
    """连续摇杆（手柄）不改动方向与力度。"""
    axis = AxisSpec(direction_count=0, magnitude_levels=0, dead_zone=0.1)
    assert axis.quantise(37.5, 0.42) == (37.5, 0.42)
    assert axis.is_continuous_direction and axis.is_continuous_magnitude


def test_key_axis_snaps_to_eight_directions_and_one_level():
    """键式摇杆（WASD）把方向吸附到 8 向、力度并成单档。"""
    axis = AxisSpec(direction_count=8, magnitude_levels=1, dead_zone=0.15)
    assert axis.quantise(40.0, 0.4) == (45.0, 1.0)
    assert axis.quantise(-10.0, 0.9) == (0.0, 1.0)
    assert not axis.is_continuous_direction and not axis.is_continuous_magnitude


def test_dead_zone_returns_neutral():
    """力度低于死区时摇杆归中。"""
    axis = AxisSpec(dead_zone=0.2)
    assert axis.quantise(90.0, 0.1) == (0.0, 0.0)


def test_intermediate_magnitude_levels_are_expressible():
    """力度档数是连续量：3 档介于开关与连续之间，无需新增设备族。"""
    axis = AxisSpec(direction_count=0, magnitude_levels=3, dead_zone=0.05)
    assert axis.quantise(0.0, 0.2)[1] == pytest.approx(1 / 3)
    assert axis.quantise(0.0, 0.5)[1] == pytest.approx(2 / 3)
    assert axis.quantise(0.0, 0.9)[1] == pytest.approx(1.0)


def test_four_direction_axis_has_no_diagonals():
    """方向档数同样连续可调：4 向 dpad 不产生对角线分量。"""
    profile = BindingProfile(
        profile_name="dpad_only",
        locomotion_axis=AxisSpec(direction_count=4, magnitude_levels=1, dead_zone=0.1))
    from control_contract.decision_segment import Move
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=100, move=Move(direction_deg=40.0, power=1.0))]), profile)
    frame = compiled.frames[0]
    assert (abs(frame.move_x) + abs(frame.move_y)) == pytest.approx(1.0)


def test_cursor_jump_is_the_large_cap_limit(desktop, gamepad):
    """"跳转到位置"= 光标上限 ≥ 屏幕对角线，与"限速逼近"共用同一段代码。"""
    assert desktop.cursor_jumps_in_one_tick
    assert not gamepad.cursor_jumps_in_one_tick
    assert desktop.cursor_cap_per_tick >= SCREEN_DIAGONAL

    step = Step(duration_ms=200, point=Point(x=1.0, y=1.0))
    start = ControlState(cursor_x=0.0, cursor_y=0.0)
    jumped = compile_segment(Segment(steps=[step]), desktop, start)
    crept = compile_segment(Segment(steps=[step]), gamepad, start)

    assert jumped.frames[0].cursor_x == 1.0          # 首 tick 即到位
    assert jumped.cursor_reached
    assert crept.frames[0].cursor_x < 1.0            # 同一段代码，只是上限小
    assert not crept.cursor_reached


def test_same_segment_compiles_on_all_profiles(mouse_keyboard, gamepad, desktop):
    """同一段大模型输出在三个 profile 上都能编译，只是展开结果不同。"""
    from control_contract.decision_segment import Move

    segment = Segment(steps=[
        Step(duration_ms=500, aim=Aim(yaw_deg=45.0), move=Move(direction_deg=30.0, power=0.6)),
    ])
    results = {
        profile.profile_name: compile_segment(segment, profile)
        for profile in (mouse_keyboard, gamepad, desktop)
    }
    assert results["minecraft_mouse_keyboard"].total_ticks == 10     # 20 Hz
    assert results["generic_gamepad"].total_ticks == 15              # 30 Hz
    assert results["desktop_mouse_keyboard"].total_ticks == 30       # 60 Hz
    # 键盘量化到 8 向单档，手柄保留连续方向与力度
    keyboard_frame = results["minecraft_mouse_keyboard"].frames[0]
    gamepad_frame = results["generic_gamepad"].frames[0]
    assert (keyboard_frame.move_x, keyboard_frame.move_y) == (1.0, 1.0)
    assert gamepad_frame.move_x == pytest.approx(0.6 * 0.5, abs=1e-9)


def test_capability_text_derived_from_numbers(mouse_keyboard, gamepad):
    """能力说明由数值推导，不含任何设备族标签。"""
    for profile in (mouse_keyboard, gamepad):
        text = describe_capabilities(profile)
        for label in ("analog", "digital", "displacement", "rate based",
                      "absolute_warp", "virtual_cursor", "focus_grid"):
            assert label not in text.lower()
    assert "8 directions" in describe_capabilities(mouse_keyboard)
    assert "any direction" in describe_capabilities(gamepad)


def test_cap_per_second_is_converted_by_tick_hz():
    """JSON 可按"每秒"标定上限，由步频换算，避免手工除法写错。"""
    from control_contract.binding_profile import parse_binding_profile

    profile = parse_binding_profile({
        "profile_name": "by_second", "tick_hz": 50.0,
        "aim_axis": {"cap_per_second": 200.0},
        "cursor_cap_per_second": 1.5,
    })
    assert profile.aim_axis.cap_per_tick == pytest.approx(4.0)
    assert profile.cursor_cap_per_tick == pytest.approx(0.03)


def test_cycling_profile_must_keep_next_prev():
    """无直达槽位按键却又禁用 next/prev 是自相矛盾的声明，应报错。"""
    with pytest.raises(ValueError, match="next/prev"):
        BindingProfile(
            profile_name="broken", slot_count=8, direct_slot_buttons=False,
            unavailable_roles=frozenset({"next"}))
