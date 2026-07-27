# -*- coding: utf-8 -*-
"""跨模块契约测试：设备无关决策段 → CraftGround V2 动作 dict 的端到端落地。"""
import pytest

from control_contract.decision_segment import (
    Aim,
    ControlState,
    Hold,
    Move,
    Point,
    Press,
    Segment,
    Select,
    Step,
)
from control_contract.profile_registry import load_named_profile
from control_contract.segment_codec import decode_segment
from control_contract.segment_compiler import compile_segment
from rl_training_environments.craftground.action_contract import CAM_MAX_DEG, V2_KEYS
from rl_training_environments.craftground.control_adapter import (
    CURSOR_DEGREES_PER_SCREEN_WIDTH,
    device_frames_to_v2_actions,
)


@pytest.fixture
def profile():
    return load_named_profile("minecraft_mouse_keyboard")


def test_action_dict_key_set_matches_v2_contract(profile):
    """适配层产物的键集必须恰好是 V2 契约的键集 + 两个相机字段。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=100, move=Move(direction_deg=0.0, power=0.5))]), profile)
    action = device_frames_to_v2_actions(compiled.frames)[0]
    assert set(action) == set(V2_KEYS) | {"camera_yaw", "camera_pitch"}
    assert all(isinstance(action[key], bool) for key in V2_KEYS)


def test_semantic_roles_land_on_minecraft_keys(profile):
    """语义角色经适配层落到正确的 Minecraft 键，模型端从未写过键名。"""
    compiled = compile_segment(Segment(steps=[
        Step(duration_ms=100, move=Move(direction_deg=0.0, power=0.4),
             holds=[Hold(role="mine")], presses=[Press(role="jump")]),
    ]), profile)
    action = device_frames_to_v2_actions(compiled.frames)[0]
    assert action["attack"] is True      # mine → primary → attack
    assert action["jump"] is True
    assert action["forward"] is True
    assert action["back"] is False and action["left"] is False and action["right"] is False


def test_movement_never_emits_opposing_keys(profile):
    """极坐标位移在结构上不可能产生前后同按 / 左右同按（取代旧互斥组消解）。"""
    for direction in range(0, 360, 7):
        compiled = compile_segment(
            Segment(steps=[Step(duration_ms=100,
                                move=Move(direction_deg=float(direction), power=1.0))]),
            profile)
        action = device_frames_to_v2_actions(compiled.frames)[0]
        assert not (action["forward"] and action["back"])
        assert not (action["left"] and action["right"])


def test_hotbar_selection_activates_single_slot(profile):
    """槽位选择只激活一个 hotbar 键。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=100, select=Select(slot=5))]), profile)
    action = device_frames_to_v2_actions(compiled.frames)[0]
    active = [key for key in V2_KEYS if key.startswith("hotbar.") and action[key]]
    assert active == ["hotbar.5"]


def test_camera_increment_respects_per_tick_limit(profile):
    """任意大转身经编译后，单 tick 相机增量都不超过契约上限，不会被 deg_to_bins 静默截断。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=2000, aim=Aim(yaw_deg=300.0, pitch_deg=-90.0))]), profile)
    for action in device_frames_to_v2_actions(compiled.frames):
        assert abs(action["camera_yaw"]) <= CAM_MAX_DEG + 1e-6
        assert abs(action["camera_pitch"]) <= CAM_MAX_DEG + 1e-6


def test_cursor_motion_becomes_camera_delta(profile):
    """CraftGround 无绝对光标通道，point 经适配层转成相机增量。"""
    compiled = compile_segment(
        Segment(steps=[Step(duration_ms=300, point=Point(x=0.7, y=0.5))]),
        profile, ControlState(cursor_x=0.5, cursor_y=0.5))
    actions = device_frames_to_v2_actions(compiled.frames, initial_cursor=(0.5, 0.5))
    total_yaw = sum(action["camera_yaw"] for action in actions)
    # 0.7 - 0.5 = 0.2 屏宽，只走 x 轴，所以用屏宽口径（96°/屏）而非屏高口径。
    assert total_yaw == pytest.approx(0.2 * CURSOR_DEGREES_PER_SCREEN_WIDTH, rel=1e-6)


def test_dirty_model_output_still_produces_executable_actions(profile):
    """脏文本 → 解码 → 编译 → V2 动作的整链永不失败。"""
    text = "I think I should mine that block.\n{\"steps\": [{\"ms\": 400, \"hold\": [\"mine\"]}]}"
    segment = decode_segment(text, profile)
    compiled = compile_segment(segment, profile)
    actions = device_frames_to_v2_actions(compiled.frames)
    assert len(actions) == 8      # 400 ms @ 20 Hz
    assert all(action["attack"] is True for action in actions)


def test_garbage_output_yields_safe_idle_actions(profile):
    """完全无法解析时整链产出安全静止动作，而不是抛错或非法动作。"""
    segment = decode_segment("no idea", profile)
    actions = device_frames_to_v2_actions(compile_segment(segment, profile).frames)
    assert actions
    for action in actions:
        assert not any(action[key] for key in V2_KEYS)
        assert action["camera_yaw"] == 0.0 and action["camera_pitch"] == 0.0
