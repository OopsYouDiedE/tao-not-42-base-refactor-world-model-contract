"""把 MineStudio 的 VPT 原始动作转换为 Gemma4 策略的结构化动作与目标 token 文本。

对外接口：
    CAMERA_SCALE、DEGREES_PER_MOUSE_PIXEL — 相机归一化标度（与部署端同口径）。
    vpt_action_to_structured — 单帧 22 维 VPT 动作 → StructuredAction。
    vpt_actions_to_structured — 动作序列 → StructuredAction 列表。
    structured_target_text — 动作序列 → 目标动作 token 文本（SFT 标签）。

相机沿用 ``rl_training_environments.craftground.action_contract`` 的 mu-law 分箱，键序由
VPT 转 CraftGround V2。旧的快塔结构化损失随快塔一起删除。
"""

from __future__ import annotations

import torch

from data_pipelines.minestudio.dataset import VPT_KEYS
from net.action_token_codec import (
    ActionTokenFormat,
    StructuredAction,
    encode_actions,
)
from rl_training_environments.craftground.action_contract import (
    CAM_BINS,
    CAM_MAX_DEG,
    CAM_MU,
    V2_KEYS,
)

DEGREES_PER_MOUSE_PIXEL = 0.15
CAMERA_SCALE = CAM_MAX_DEG / DEGREES_PER_MOUSE_PIXEL

_V2_OF_VPT = {
    "key_w": "forward",
    "key_s": "back",
    "key_a": "left",
    "key_d": "right",
    "key_space": "jump",
    "key_sneak": "sneak",
    "key_sprint": "sprint",
    "key_attack": "attack",
    "key_use": "use",
    "key_drop": "drop",
    "key_inventory": "inventory",
    **{f"key_hotbar.{index}": f"hotbar.{index}" for index in range(1, 10)},
}
# 每个 V2 键在 22 维 VPT 动作里对应的键位下标（前两维是相机，键位从下标 2 开始）。
_V2_TO_VPT_KEY_INDEX = {
    v2_key: VPT_KEYS.index(vpt_key)
    for vpt_key, v2_key in _V2_OF_VPT.items()
}
if set(_V2_TO_VPT_KEY_INDEX) != set(V2_KEYS):
    raise RuntimeError("VPT 与 CraftGround 动作键序不完整")

import math


def _camera_value_to_bin(value: float) -> int:
    """把归一相机值 ``[-1,1]`` 经 mu-law 压缩转为分类索引。"""
    value = max(-1.0, min(1.0, float(value)))
    compressed = (
        math.copysign(1.0, value)
        * math.log1p(CAM_MU * abs(value))
        / math.log1p(CAM_MU)
    )
    return int(round((compressed + 1.0) / 2.0 * (CAM_BINS - 1)))


def vpt_action_to_structured(action: torch.Tensor) -> StructuredAction:
    """把单帧 ``[22]`` VPT 动作转换为结构化动作。

    Parameters
    ----------
    action : torch.Tensor
        Shape ``[22]``，前两维为归一相机 ``[yaw, pitch] in [-1,1]``，其余为二值键。

    Returns
    -------
    StructuredAction
        相机 bin + V2 键；互斥约束由 StructuredAction 构造器强制。
    """
    if action.shape[-1] != 2 + len(VPT_KEYS):
        raise ValueError(f"VPT 动作维度必须为 {2 + len(VPT_KEYS)}")
    values = action.detach().float().tolist()
    keys = {
        v2_key: bool(values[2 + index] >= 0.5)
        for v2_key, index in _V2_TO_VPT_KEY_INDEX.items()
    }
    return StructuredAction(
        camera_yaw_bin=_camera_value_to_bin(values[0]),
        camera_pitch_bin=_camera_value_to_bin(values[1]),
        keys=keys,
    )


def vpt_actions_to_structured(actions: torch.Tensor) -> list[StructuredAction]:
    """把 ``[T,22]`` VPT 动作序列转换为结构化动作列表。"""
    if actions.ndim != 2:
        raise ValueError("actions 必须为 [T,22]")
    return [vpt_action_to_structured(actions[index]) for index in range(actions.shape[0])]


def structured_target_text(
    actions: torch.Tensor,
    action_format: ActionTokenFormat,
) -> str:
    """把 ``[T,22]`` VPT 动作序列编码为该格式的目标动作 token 文本（SFT 标签）。"""
    return encode_actions(vpt_actions_to_structured(actions), action_format)
