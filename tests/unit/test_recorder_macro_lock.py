"""验证录制器界面入口 macro_from_dict：turn spec 构造正确、校验直通到 Turn。

纯逻辑测试，不起 CraftGround env（recorder_macros 只依赖 macro_action_compiler）。
"""

import pytest

from rl_training_environments.craftground.macro_action_compiler import CAM_TICKS, TAP_TICKS
from rl_training_environments.craftground.recorder_macros import macro_from_dict


def test_turn_clicks_are_two_ticks():
    """turn 的 clicks 键固定 TAP_TICKS。"""
    macro = macro_from_dict({"kind": "turn", "clicks": ["jump", "attack"]})
    assert macro.kind == "turn"
    frames, _ = macro.expand({})
    assert len(frames) == TAP_TICKS


def test_turn_camera_delta_two_ticks():
    """turn 的相机固定 CAM_TICKS。"""
    macro = macro_from_dict({"kind": "turn", "camera_mode": "delta", "delta_yaw": 30.0})
    frames, _ = macro.expand({})
    assert len(frames) == CAM_TICKS
    assert macro.delta_yaw == 30.0


def test_turn_hold_retains_custom_duration():
    """turn 的长按时长由作者设定，原样保留。"""
    macro = macro_from_dict({"kind": "turn", "holds": {"forward": 40}})
    assert macro.holds["forward"] == 40


def test_turn_camera_over_36_raises():
    """相机单回合 > 36° 报错（直通 Turn 校验）。"""
    with pytest.raises(ValueError):
        macro_from_dict({"kind": "turn", "camera_mode": "delta", "delta_yaw": 40.0})


def test_turn_empty_raises():
    """空回合报错。"""
    with pytest.raises(ValueError):
        macro_from_dict({"kind": "turn"})


def test_turn_wait_ticks_pass_through():
    """纯等待回合直通。"""
    macro = macro_from_dict({"kind": "turn", "wait_ticks": 8})
    assert macro.wait_ticks == 8


def test_mc_pass_through():
    """mc 正常构造。"""
    assert macro_from_dict({"kind": "mc", "command": "give @p diamond 1"}).command == "give @p diamond 1"


def test_unknown_kind_raises():
    """未知宏类型报错。"""
    with pytest.raises(ValueError):
        macro_from_dict({"kind": "teleport"})
