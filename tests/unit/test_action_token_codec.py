"""验证动作 token 编解码往返一致、互斥约束与脏文本鲁棒解码。"""

import random

import pytest

from net.action_token_codec import (
    ACTION_KEY_GROUPS,
    CAMERA_NEUTRAL_BIN,
    ActionTokenFormat,
    StructuredAction,
    decode_actions,
    describe_format,
    encode_actions,
)
from net.action_token_codec import V2_KEYS


def _random_action(generator: random.Random) -> StructuredAction:
    keys = {key: False for key in V2_KEYS}
    if generator.random() < 0.6:
        keys[generator.choice(["forward", "back"])] = True
    if generator.random() < 0.3:
        keys[generator.choice(["left", "right"])] = True
    if generator.random() < 0.3:
        keys[generator.choice(["sneak", "sprint"])] = True
    for key in ("jump", "attack", "use", "drop", "inventory"):
        if generator.random() < 0.2:
            keys[key] = True
    if generator.random() < 0.2:
        keys[f"hotbar.{generator.randint(1, 9)}"] = True
    return StructuredAction(
        camera_yaw_bin=generator.randint(0, 10),
        camera_pitch_bin=generator.randint(0, 10),
        keys=keys,
    )


@pytest.mark.parametrize("action_format", list(ActionTokenFormat))
def test_encode_decode_round_trip(action_format):
    """三种格式在随机合法动作上逐帧往返恒等。"""
    generator = random.Random(0)
    for _ in range(200):
        sequence = [_random_action(generator) for _ in range(5)]
        decoded = decode_actions(encode_actions(sequence, action_format), 5)
        for original, restored in zip(sequence, decoded):
            assert original.keys == restored.keys
            assert original.camera_yaw_bin == restored.camera_yaw_bin
            assert original.camera_pitch_bin == restored.camera_pitch_bin


def test_construction_enforces_mutual_exclusivity():
    """构造器对每个互斥组只保留组内优先级最高的激活键。"""
    conflicting = {key: True for key in V2_KEYS}
    action = StructuredAction(keys=conflicting)
    for group in ACTION_KEY_GROUPS:
        assert sum(int(action.keys[key]) for key in group) <= 1
    # 组内优先级：保留第一个。
    assert action.keys["forward"] and not action.keys["back"]
    assert action.keys["left"] and not action.keys["right"]
    assert action.keys["hotbar.1"] and not action.keys["hotbar.2"]


def test_camera_bins_clamped_into_range():
    """越界相机 bin 被截断到合法范围。"""
    action = StructuredAction(camera_yaw_bin=999, camera_pitch_bin=-999)
    assert 0 <= action.camera_yaw_bin <= 2 * CAMERA_NEUTRAL_BIN
    assert 0 <= action.camera_pitch_bin <= 2 * CAMERA_NEUTRAL_BIN


def test_decode_pads_and_truncates_to_horizon():
    """无法识别的行被跳过，识别不足补 noop，超出截断。"""
    text = "garbage line\nt0: F attack cam=+3,-2\nmore noise"
    decoded = decode_actions(text, 4)
    assert len(decoded) == 4
    assert decoded[0].keys["forward"] and decoded[0].keys["attack"]
    # 补齐帧为 noop（中性相机、无键）。
    assert decoded[3].active_keys() == []
    assert decoded[3].camera_yaw_bin == CAMERA_NEUTRAL_BIN


def test_decode_rejects_illegal_combinations_from_noisy_text():
    """脏文本里同时出现互斥键时解码仍产生结构合法动作。"""
    decoded = decode_actions("t0: forward back left right sneak sprint", 1)[0]
    for group in ACTION_KEY_GROUPS:
        assert sum(int(decoded.keys[key]) for key in group) <= 1


@pytest.mark.parametrize("action_format", list(ActionTokenFormat))
def test_describe_format_is_nonempty(action_format):
    """每种格式都提供非空的自然语言说明。"""
    assert describe_format(action_format).strip()
