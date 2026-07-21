"""Qwen3VL 动作策略的关键动作匹配指标与闭环置信区间。

对外接口：
    KEY_ACTION_FIELDS — 参与"关键动作"比较的语义字段。
    key_action_signature — 从 StructuredAction 抽取关键动作签名。
    key_action_match — 判定两帧动作的关键动作是否一致。
    key_action_agreement_rate — 序列级关键动作一致率。
    structured_to_v2_action — StructuredAction → 完整 CraftGround V2 动作字典。
    wilson_interval — 二项成功率 Wilson 95% 区间。

"关键动作"只看对任务推进有决定意义的粗粒度控制：移动方向、转向、姿态、攻击 / 使用、
快捷栏切换、相机的粗方向。抖动级差异（相机小幅偏移、drop/inventory 等偶发键）不计入，
以便判断大模型是否抓住了关键动作而非逐位复刻（用户验收口径）。
"""

from __future__ import annotations

import math

from net.action_token_codec import CAMERA_NEUTRAL_BIN, StructuredAction
from rl_training_environments.craftground.action_contract import (
    CAM_MAX_DEG,
    CAM_MU,
    V2_KEYS,
)

# 相机方向的死区：偏移绝对值 <= 该 bin 数视为"基本不动"，避免抖动误判。
_CAMERA_DIRECTION_DEADZONE = 1

KEY_ACTION_FIELDS = ("move", "turn", "stance", "attack", "use", "hotbar", "yaw_dir", "pitch_dir")


def _direction(bin_value: int) -> int:
    """把相机 bin 转为粗方向：-1 / 0 / +1，死区内记 0。"""
    offset = bin_value - CAMERA_NEUTRAL_BIN
    if offset > _CAMERA_DIRECTION_DEADZONE:
        return 1
    if offset < -_CAMERA_DIRECTION_DEADZONE:
        return -1
    return 0


def key_action_signature(action: StructuredAction) -> dict[str, object]:
    """抽取参与关键动作比较的语义签名。"""
    move = "forward" if action.keys["forward"] else "back" if action.keys["back"] else "none"
    turn = "left" if action.keys["left"] else "right" if action.keys["right"] else "none"
    stance = "sneak" if action.keys["sneak"] else "sprint" if action.keys["sprint"] else "none"
    hotbar = next(
        (index for index in range(1, 10) if action.keys[f"hotbar.{index}"]), 0,
    )
    return {
        "move": move,
        "turn": turn,
        "stance": stance,
        "attack": bool(action.keys["attack"]),
        "use": bool(action.keys["use"]),
        "hotbar": hotbar,
        "yaw_dir": _direction(action.camera_yaw_bin),
        "pitch_dir": _direction(action.camera_pitch_bin),
    }


def key_action_match(first: StructuredAction, second: StructuredAction) -> bool:
    """两帧动作的全部关键字段是否一致。"""
    return key_action_signature(first) == key_action_signature(second)


def key_action_field_matches(
    first: StructuredAction,
    second: StructuredAction,
) -> dict[str, bool]:
    """逐关键字段返回是否一致，用于定位差异来源。"""
    left = key_action_signature(first)
    right = key_action_signature(second)
    return {name: left[name] == right[name] for name in KEY_ACTION_FIELDS}


def key_action_agreement_rate(
    predicted: list[StructuredAction],
    reference: list[StructuredAction],
) -> float:
    """两个等长序列逐帧关键动作一致的帧比例。"""
    if len(predicted) != len(reference):
        raise ValueError("predicted 与 reference 长度必须一致")
    if not predicted:
        raise ValueError("动作序列不能为空")
    matched = sum(
        1 for prediction, target in zip(predicted, reference)
        if key_action_match(prediction, target)
    )
    return matched / len(predicted)


def structured_to_v2_action(action: StructuredAction) -> dict[str, object]:
    """把结构化动作转换为完整 CraftGround V2 动作字典（含相机角度）。"""

    def bin_to_degrees(bin_value: int) -> float:
        compressed = (bin_value - CAMERA_NEUTRAL_BIN) / CAMERA_NEUTRAL_BIN
        magnitude = (math.pow(1.0 + CAM_MU, abs(compressed)) - 1.0) / CAM_MU
        return math.copysign(magnitude, compressed) * CAM_MAX_DEG

    result = {key: bool(action.keys[key]) for key in V2_KEYS}
    result["camera_yaw"] = bin_to_degrees(action.camera_yaw_bin)
    result["camera_pitch"] = bin_to_degrees(action.camera_pitch_bin)
    return result


def wilson_interval(successes: int, episodes: int, z_score: float = 1.96) -> tuple[float, float]:
    """计算二项成功率 Wilson 95% 置信区间。"""
    if episodes < 1:
        raise ValueError("episodes 必须大于零")
    probability = successes / episodes
    denominator = 1.0 + z_score**2 / episodes
    center = (probability + z_score**2 / (2.0 * episodes)) / denominator
    radius = (
        z_score
        * math.sqrt(
            probability * (1.0 - probability) / episodes
            + z_score**2 / (4.0 * episodes**2),
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)
