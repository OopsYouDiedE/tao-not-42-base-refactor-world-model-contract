# -*- coding: utf-8 -*-
"""决策段编解码的鲁棒性与设备无关性单测。"""
import pytest

from control_contract.decision_segment import (
    Aim,
    GuardComparison,
    Hold,
    Move,
    PIXEL_CHANGE_CHANNEL,
    Press,
    Segment,
    Step,
    TailPolicy,
)
from control_contract.profile_registry import load_named_profile
from control_contract.segment_codec import (
    decode_segment,
    describe_segment_format,
    encode_segment,
)


@pytest.fixture
def mouse_keyboard():
    return load_named_profile("minecraft_mouse_keyboard")


@pytest.fixture
def gamepad():
    return load_named_profile("generic_gamepad")


def test_encode_decode_round_trip(mouse_keyboard):
    """编码后解码应还原语义等价的段。"""
    segment = Segment(
        steps=[
            Step(duration_ms=400, aim=Aim(yaw_deg=30.0, pitch_deg=-5.0),
                 move=Move(direction_deg=0.0, power=1.0), holds=[Hold(role="primary")]),
            Step(duration_ms=200, presses=[Press(role="jump")]),
        ],
        tail=TailPolicy.HOLD,
        lease_ms=1500,
        intent="walk forward while mining",
    )
    decoded = decode_segment(encode_segment(segment), mouse_keyboard)
    assert len(decoded.steps) == 2
    assert decoded.steps[0].aim == Aim(yaw_deg=30.0, pitch_deg=-5.0)
    assert decoded.steps[0].holds == [Hold(role="primary")]
    assert decoded.steps[1].presses == [Press(role="jump")]
    assert decoded.tail is TailPolicy.HOLD
    assert decoded.lease_ms == 1500


def test_ability_alias_resolves_to_core_role(mouse_keyboard):
    """profile 声明的游戏能力名应解析为核心角色，模型端无需知道键位。"""
    decoded = decode_segment(
        '{"steps": [{"ms": 300, "hold": ["mine"], "press": ["place"]}]}', mouse_keyboard)
    assert decoded.steps[0].holds == [Hold(role="primary")]
    assert decoded.steps[0].presses == [Press(role="secondary")]


def test_unknown_role_dropped_not_raised(mouse_keyboard):
    """未知角色被丢弃，其余原语仍然生效，解码端永不抛错。"""
    decoded = decode_segment(
        '{"steps": [{"ms": 300, "press": ["teleport", "jump"]}]}', mouse_keyboard)
    assert decoded.steps[0].presses == [Press(role="jump")]


def test_role_unavailable_on_profile_dropped(mouse_keyboard):
    """profile 声明不存在的角色（Minecraft 无 nav_up）被丢弃。"""
    decoded = decode_segment(
        '{"steps": [{"ms": 200, "press": ["nav_up", "jump"]}]}', mouse_keyboard)
    assert decoded.steps[0].presses == [Press(role="jump")]


def test_prose_and_code_fence_tolerated(mouse_keyboard):
    """输出裹散文与代码围栏仍能解出段。"""
    text = (
        "Sure, here is my plan.\n```json\n"
        '{"intent": "go", "steps": [{"ms": 500, "move": {"dir": 0, "power": 1}}], '
        '"tail": "hold", "lease_ms": 800}\n```\nHope that helps!'
    )
    decoded = decode_segment(text, mouse_keyboard)
    assert decoded.steps[0].move == Move(direction_deg=0.0, power=1.0)
    assert decoded.tail is TailPolicy.HOLD


def test_garbage_falls_back_to_safe_segment(mouse_keyboard):
    """完全无法识别时返回安全兜底段而非抛错。"""
    decoded = decode_segment("I am not sure what to do here.", mouse_keyboard)
    assert len(decoded.steps) == 1
    assert decoded.tail is TailPolicy.RELEASE_ALL
    assert decoded.lease_ms == 0


def test_out_of_range_values_clamped(mouse_keyboard):
    """越界数值被截断到合法区间，不产生非法段。"""
    decoded = decode_segment(
        '{"steps": [{"ms": 999999, "move": {"dir": 400, "power": 7}}], "lease_ms": -50}',
        mouse_keyboard)
    assert decoded.steps[0].duration_ms == 20_000
    assert decoded.steps[0].move.power == 1.0
    assert decoded.lease_ms == 0


def test_hold_wins_over_release_in_same_step(mouse_keyboard):
    """同步内 hold 与 release 同一角色时保留 hold，段永远结构合法。"""
    decoded = decode_segment(
        '{"steps": [{"ms": 200, "hold": ["primary"], "release": ["primary"]}]}', mouse_keyboard)
    assert decoded.steps[0].holds == [Hold(role="primary")]
    assert decoded.steps[0].releases == []


def test_unknown_guard_channel_dropped(mouse_keyboard):
    """未声明的守卫通道被丢弃，已知通道保留。"""
    decoded = decode_segment(
        '{"steps": [{"ms": 500}], "guards": ['
        '{"channel": "boss_rage", "when": "above", "threshold": 0.5},'
        '{"channel": "pixel.change", "when": "above", "threshold": 0.2, "label": "something moved"}'
        ']}', mouse_keyboard)
    assert len(decoded.guards) == 1
    assert decoded.guards[0].channel == PIXEL_CHANGE_CHANNEL
    assert decoded.guards[0].comparison is GuardComparison.ABOVE


def test_declared_signal_channel_accepted(mouse_keyboard):
    """环境声明的信号通道被接受。"""
    decoded = decode_segment(
        '{"steps": [{"ms": 500}], "guards": ['
        '{"channel": "health", "when": "below", "threshold": 0.4}]}',
        mouse_keyboard, signal_channels=frozenset({"health"}))
    assert decoded.guards[0].channel == "health"


def test_select_rejected_when_out_of_range(mouse_keyboard):
    """越界槽位被丢弃（Minecraft 只有 1..9）。"""
    decoded = decode_segment('{"steps": [{"ms": 200, "select": 12}]}', mouse_keyboard)
    assert decoded.steps[0].select is None


def test_text_dropped_when_profile_lacks_keyboard(gamepad):
    """不支持文本的 profile 上 text 原语被丢弃。"""
    decoded = decode_segment('{"steps": [{"ms": 200, "text": "hello"}]}', gamepad)
    assert decoded.steps[0].text is None


def test_format_description_mentions_latency_and_guards(mouse_keyboard):
    """格式说明须按实测延迟参数化，并说明守卫与租约。"""
    description = describe_segment_format(mouse_keyboard, inference_latency_ms=700)
    assert "700 ms" in description
    assert "pixel.change" in description
    assert "lease_ms" in description
    assert "minecraft_mouse_keyboard" in description


def test_format_description_states_device_costs_as_numbers(gamepad):
    """能力说明按实际数值描述两个摇杆的代价，而不是设备族标签。"""
    description = describe_segment_format(gamepad, inference_latency_ms=500)
    assert "220 degrees per second" in description        # 瞄准摇杆上限
    assert "any direction" in description                 # 位移摇杆连续
    assert "true magnitude" in description
    assert "cycling" in description                       # 无直达槽位按键
    assert "text" not in description.split("Available primitives:")[1].split("\n")[0]
