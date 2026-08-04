"""Framework-neutral and PyTorch objectives for relative-advantage training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def masked_mean(losses: Sequence[float], mask: Sequence[bool | float]) -> float:
    values = np.asarray(losses, dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("losses and mask must be one-dimensional with equal shapes")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("losses and mask must contain finite values and non-negative weights")
    total_weight = float(weights.sum())
    if total_weight == 0:
        raise ValueError("mask must select at least one value")
    return float(np.dot(values, weights) / total_weight)


def grouped_relative_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if len(rewards) != len(group_ids) or epsilon <= 0:
        raise ValueError("rewards/group_ids must have equal lengths and epsilon must be positive")
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rewards must be a finite one-dimensional sequence")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("each relative-advantage group needs at least two samples")
    result = np.empty_like(values)
    for indices in groups.values():
        group = values[indices]
        centered = group - group.mean()
        deviation = group.std()
        result[indices] = (
            centered
            if not normalize
            else (
                np.zeros_like(centered) if deviation < epsilon else centered / (deviation + epsilon)
            )
        )
    return result


@dataclass(frozen=True)
class JointObjectiveWeights:
    relative_advantage: float = 1.0
    behavior_cloning: float = 1.0

    def __post_init__(self) -> None:
        if min(self.relative_advantage, self.behavior_cloning) < 0:
            raise ValueError("objective weights cannot be negative")
        if self.relative_advantage == self.behavior_cloning == 0:
            raise ValueError("at least one objective weight must be positive")


@dataclass(frozen=True)
class JointResult:
    total: float
    relative_advantage: float
    behavior_cloning: float


class JointObjective:
    def __init__(self, weights: JointObjectiveWeights | None = None) -> None:
        self.weights = weights or JointObjectiveWeights()

    def __call__(self, relative_advantage_loss: float, behavior_cloning_loss: float) -> JointResult:
        advantage, cloning = float(relative_advantage_loss), float(behavior_cloning_loss)
        return JointResult(
            self.weights.relative_advantage * advantage + self.weights.behavior_cloning * cloning,
            advantage,
            cloning,
        )


@dataclass(frozen=True)
class TorchJointResult:
    total: Any
    relative_advantage: Any
    behavior_cloning: Any
    approximate_kl: Any | None = None
    clip_fraction: Any | None = None


def torch_joint_objective(
    sequence_nll: Any,
    advantages: Any,
    policy_mask: Any,
    reference_mask: Any,
    *,
    advantage_weight: float = 1.0,
    behavior_cloning_weight: float = 1.0,
) -> TorchJointResult:
    import torch

    tensors = (sequence_nll, advantages, policy_mask, reference_mask)
    if (
        any(tensor.ndim != 1 for tensor in tensors)
        or len({tensor.numel() for tensor in tensors}) != 1
    ):
        raise ValueError("all inputs must be one-dimensional tensors of equal length")
    policy, reference = policy_mask.bool(), reference_mask.bool()
    if torch.any(policy & reference) or (int(policy.sum()), int(reference.sum())) != (6, 2):
        raise ValueError("objective requires disjoint 6 policy + 2 reference masks")
    relative = (sequence_nll[policy] * advantages[policy].detach()).mean()
    cloning = sequence_nll[reference].mean()
    return TorchJointResult(
        advantage_weight * relative + behavior_cloning_weight * cloning, relative, cloning
    )


def clipped_token_joint_objective(
    new_logprobs: Any,
    old_logprobs: Any,
    response_mask: Any,
    advantages: Any,
    policy_mask: Any,
    reference_mask: Any,
    *,
    clip_epsilon: float = 0.2,
    advantage_weight: float = 1.0,
    behavior_cloning_weight: float = 1.0,
) -> TorchJointResult:
    import torch

    if (
        new_logprobs.ndim != 2
        or old_logprobs.shape != new_logprobs.shape
        or response_mask.shape != new_logprobs.shape
    ):
        raise ValueError("logprobs and response_mask must have matching two-dimensional shapes")
    batch_size = new_logprobs.shape[0]
    if any(
        value.ndim != 1 or value.numel() != batch_size
        for value in (advantages, policy_mask, reference_mask)
    ):
        raise ValueError("advantages and source masks must match the batch")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be between zero and one")
    policy, reference = policy_mask.bool(), reference_mask.bool()
    if (
        torch.any(policy & reference)
        or int((policy | reference).sum()) != 4
        or int(policy.sum()) == 0
    ):
        raise ValueError("selection requires four disjoint samples including policy samples")
    mask = response_mask.bool()
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each sample needs at least one response token")
    policy_tokens, reference_tokens = mask & policy[:, None], mask & reference[:, None]
    if not torch.isfinite(old_logprobs[policy_tokens]).all():
        raise ValueError("policy old_logprobs must be finite")
    log_ratio = new_logprobs[policy_tokens] - old_logprobs[policy_tokens].detach()
    ratio = torch.exp(log_ratio)
    token_advantages = advantages[:, None].expand_as(new_logprobs)[policy_tokens].detach()
    relative = -torch.minimum(
        ratio * token_advantages, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * token_advantages
    ).mean()
    cloning = (
        -new_logprobs[reference_tokens].mean()
        if torch.any(reference_tokens)
        else new_logprobs.sum() * 0.0
    )
    return TorchJointResult(
        advantage_weight * relative + behavior_cloning_weight * cloning,
        relative,
        cloning,
        ((ratio - 1) - log_ratio).mean(),
        ((ratio - 1).abs() > clip_epsilon).float().mean(),
    )


@dataclass(frozen=True)
class VisionGeometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale: float
    raw_patches: int
    soft_tokens: int


def plan_gemma4_geometry(width: int, height: int, scale: float = 1.0) -> VisionGeometry:
    if width <= 0 or height <= 0 or scale <= 0:
        raise ValueError("dimensions and scale must be positive")
    resized_width = max(48, round(width * scale / 48) * 48)
    resized_height = max(48, round(height * scale / 48) * 48)
    raw, soft = (
        (resized_width // 16) * (resized_height // 16),
        (resized_width // 48) * (resized_height // 48),
    )
    if raw > 2520 or soft > 280:
        raise ValueError(f"Gemma 4 visual budget exceeded: raw_patches={raw}, soft_tokens={soft}")
    return VisionGeometry(width, height, resized_width, resized_height, scale, raw, soft)


def camera_degrees_after_resize(pitch_degrees: float, yaw_degrees: float) -> tuple[float, float]:
    return float(pitch_degrees), float(yaw_degrees)
