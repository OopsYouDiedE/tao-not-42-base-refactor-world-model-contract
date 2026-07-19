"""将 VPT 原始动作转换为 Minecraft 快塔的结构化监督目标。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from datasets.vpt.video_dataset import VPT_KEYS
from net.spatiotemporal_fast_tower import StructuredActionOutput
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
VPT_TO_V2 = [
    VPT_KEYS.index(vpt_key)
    for v2_key in V2_KEYS
    for vpt_key, target in _V2_OF_VPT.items()
    if target == v2_key
]
if len(VPT_TO_V2) != len(V2_KEYS):
    raise RuntimeError("VPT 与 CraftGround 动作键序不完整")


def camera_to_bins(value: torch.Tensor) -> torch.Tensor:
    """归一相机值 ``[-1,1]`` 转为 mu-law 分类索引。"""
    value = value.float().clamp(-1.0, 1.0)
    compressed = (
        torch.sign(value)
        * torch.log1p(CAM_MU * value.abs())
        / math.log1p(CAM_MU)
    )
    return torch.round((compressed + 1.0) / 2.0 * (CAM_BINS - 1)).long()


def bin_centers(index: torch.Tensor) -> torch.Tensor:
    """mu-law 分类索引转为归一相机 bin 中心。"""
    compressed = index.float() / (CAM_BINS - 1) * 2.0 - 1.0
    return (
        torch.sign(compressed)
        * (torch.pow(1 + CAM_MU, compressed.abs()) - 1.0)
        / CAM_MU
    )


def encode_targets(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 ``[...,22]`` VPT 动作转换为相机、V2 键位与规范动作。"""
    camera_bins = camera_to_bins(action[..., :2])
    keys = action[..., 2:][..., VPT_TO_V2].float()
    canonical_action = torch.cat([bin_centers(camera_bins), keys], dim=-1)
    return camera_bins, keys, canonical_action


def structured_action_loss(
    output: StructuredActionOutput,
    camera_bins: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """计算有互斥掩码的动作块监督损失，危险归约使用 fp32。"""
    camera = F.cross_entropy(
        output.camera_logits.float().flatten(0, -2), camera_bins.flatten(),
    )

    def exclusive_loss(
        logits: torch.Tensor,
        negative: torch.Tensor,
        positive: torch.Tensor,
        negative_class: int,
        neutral_class: int,
        positive_class: int,
    ) -> torch.Tensor:
        valid = ~(negative.bool() & positive.bool())
        target = torch.full_like(negative, neutral_class, dtype=torch.long)
        target = torch.where(negative.bool(), negative_class, target)
        target = torch.where(positive.bool(), positive_class, target)
        losses = F.cross_entropy(
            logits.float().flatten(0, -2), target.flatten(), reduction="none",
        ).reshape_as(target)
        denominator = valid.float().sum().clamp(min=1e-4)
        return (losses * valid.float()).sum() / denominator

    forward_backward = exclusive_loss(
        output.move_fb_logits, keys[..., 1], keys[..., 0], 0, 1, 2,
    )
    left_right = exclusive_loss(
        output.move_lr_logits, keys[..., 2], keys[..., 3], 0, 1, 2,
    )
    stance = exclusive_loss(
        output.stance_logits, keys[..., 5], keys[..., 6], 1, 0, 2,
    )
    hotbar_keys = keys[..., 11:20]
    hotbar_valid = hotbar_keys.sum(dim=-1) <= 1
    hotbar_target = torch.where(
        hotbar_keys.any(dim=-1), hotbar_keys.argmax(dim=-1) + 1,
        torch.zeros_like(hotbar_keys[..., 0], dtype=torch.long),
    )
    hotbar_losses = F.cross_entropy(
        output.hotbar_logits.float().flatten(0, -2), hotbar_target.flatten(),
        reduction="none",
    ).reshape_as(hotbar_target)
    hotbar = (
        (hotbar_losses * hotbar_valid.float()).sum()
        / hotbar_valid.float().sum().clamp(min=1e-4)
    )
    button_target = keys[..., [4, 7, 8, 9, 10]]
    buttons = F.binary_cross_entropy_with_logits(
        output.button_logits.float(), button_target.float(),
    )
    return camera + forward_backward + left_right + stance + hotbar + buttons
