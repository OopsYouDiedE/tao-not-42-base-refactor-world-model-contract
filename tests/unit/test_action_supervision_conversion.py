"""验证 VPT 动作 → 结构化动作转换与关键动作匹配指标。"""

import torch

from data_pipelines.minestudio.dataset import VPT_KEYS
from net.action_token_codec import CAMERA_NEUTRAL_BIN, StructuredAction
from train.minecraft.action_supervision import (
    vpt_action_to_structured,
    vpt_actions_to_structured,
)
from train.minecraft.evaluation import (
    key_action_agreement_rate,
    key_action_match,
    structured_to_v2_action,
)
from rl_training_environments.craftground.action_contract import V2_KEYS


def _vpt_action(**pressed) -> torch.Tensor:
    """构造 [22] VPT 动作：前两维相机，其余按名字置键。"""
    action = torch.zeros(2 + len(VPT_KEYS))
    action[0] = pressed.pop("yaw", 0.0)
    action[1] = pressed.pop("pitch", 0.0)
    for key, value in pressed.items():
        action[2 + VPT_KEYS.index(key)] = float(value)
    return action


def test_vpt_forward_and_attack_map_to_v2():
    """前进 + 攻击正确映射到 V2 键。"""
    action = vpt_action_to_structured(_vpt_action(key_w=1.0, key_attack=1.0))
    assert action.keys["forward"] and action.keys["attack"]
    assert not action.keys["back"]


def test_zero_camera_maps_to_neutral_bin():
    """零相机增量映射到中性 bin。"""
    action = vpt_action_to_structured(_vpt_action())
    assert action.camera_yaw_bin == CAMERA_NEUTRAL_BIN
    assert action.camera_pitch_bin == CAMERA_NEUTRAL_BIN


def test_positive_camera_above_neutral():
    """正相机值分箱高于中性。"""
    action = vpt_action_to_structured(_vpt_action(yaw=1.0, pitch=-1.0))
    assert action.camera_yaw_bin > CAMERA_NEUTRAL_BIN
    assert action.camera_pitch_bin < CAMERA_NEUTRAL_BIN


def test_sequence_conversion_length():
    """序列转换保持帧数。"""
    actions = torch.zeros(4, 2 + len(VPT_KEYS))
    assert len(vpt_actions_to_structured(actions)) == 4


def test_key_action_match_ignores_jitter():
    """相机死区内的小偏移不改变关键动作签名。"""
    base = StructuredAction(camera_yaw_bin=CAMERA_NEUTRAL_BIN)
    jittered = StructuredAction(camera_yaw_bin=CAMERA_NEUTRAL_BIN + 1)
    assert key_action_match(base, jittered)


def test_key_action_match_detects_direction_change():
    """相机粗方向不同则关键动作不一致。"""
    left = StructuredAction(camera_yaw_bin=0)
    right = StructuredAction(camera_yaw_bin=2 * CAMERA_NEUTRAL_BIN)
    assert not key_action_match(left, right)


def test_agreement_rate_full_and_zero():
    """完全一致率为 1，全不一致为 0。"""
    forward = [StructuredAction(keys={"forward": True}) for _ in range(3)]
    back = [StructuredAction(keys={"back": True}) for _ in range(3)]
    assert key_action_agreement_rate(forward, forward) == 1.0
    assert key_action_agreement_rate(forward, back) == 0.0


def test_structured_to_v2_action_has_all_keys_and_camera():
    """转换出的 V2 动作含全部键与相机角度字段。"""
    action = structured_to_v2_action(StructuredAction(keys={"forward": True}))
    for key in V2_KEYS:
        assert key in action
    assert "camera_yaw" in action and "camera_pitch" in action
    assert action["forward"] is True
