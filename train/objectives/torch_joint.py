"""可反向传播的 2+6 联合目标。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchJointResult:
    total: torch.Tensor
    relative_advantage: torch.Tensor
    behavior_cloning: torch.Tensor
    approximate_kl: torch.Tensor | None = None
    clip_fraction: torch.Tensor | None = None


def torch_joint_objective(
    sequence_nll: torch.Tensor,
    advantages: torch.Tensor,
    policy_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    advantage_weight: float = 1.0,
    behavior_cloning_weight: float = 1.0,
) -> TorchJointResult:
    """用序列 NLL 做直接优势加权；这不是 PPO/GRPO 概率比目标。"""
    tensors = (sequence_nll, advantages, policy_mask, reference_mask)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("所有输入必须是一维张量")
    if len({tensor.numel() for tensor in tensors}) != 1:
        raise ValueError("所有输入长度必须相同")
    policy = policy_mask.bool()
    reference = reference_mask.bool()
    if torch.any(policy & reference):
        raise ValueError("policy 与 reference mask 不能重叠")
    if int(policy.sum()) != 6 or int(reference.sum()) != 2:
        raise ValueError("联合目标要求严格的 2 reference + 6 policy")
    relative = (sequence_nll[policy] * advantages[policy].detach()).mean()
    cloning = sequence_nll[reference].mean()
    total = advantage_weight * relative + behavior_cloning_weight * cloning
    return TorchJointResult(total=total, relative_advantage=relative, behavior_cloning=cloning)


def clipped_token_joint_objective(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    policy_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    advantage_weight: float = 1.0,
    behavior_cloning_weight: float = 1.0,
) -> TorchJointResult:
    """对 2+6 组中优势最高的四条执行 clipped policy 与 reference BC。"""
    if new_logprobs.ndim != 2 or old_logprobs.shape != new_logprobs.shape:
        raise ValueError("new_logprobs 与 old_logprobs 必须是形状相同的二维张量")
    if response_mask.shape != new_logprobs.shape:
        raise ValueError("response_mask 必须与 logprobs 形状相同")
    batch_size = new_logprobs.shape[0]
    vectors = (advantages, policy_mask, reference_mask)
    if any(value.ndim != 1 or value.numel() != batch_size for value in vectors):
        raise ValueError("优势和来源 mask 必须是一维 batch 张量")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon 必须位于 (0, 1)")

    policy = policy_mask.bool()
    reference = reference_mask.bool()
    if torch.any(policy & reference):
        raise ValueError("policy 与 reference mask 不能重叠")
    if int((policy | reference).sum()) != 4:
        raise ValueError("联合目标要求从 2+6 组中正好选择 4 条轨迹")
    if int(policy.sum()) == 0:
        raise ValueError("入选轨迹必须至少包含一条 policy")

    mask = response_mask.bool()
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("每条轨迹必须至少包含一个响应 token")
    policy_tokens = mask & policy[:, None]
    reference_tokens = mask & reference[:, None]
    if not torch.isfinite(old_logprobs[policy_tokens]).all():
        raise ValueError("policy old_logprobs 必须是有限值")

    log_ratio = new_logprobs[policy_tokens] - old_logprobs[policy_tokens].detach()
    ratio = torch.exp(log_ratio)
    token_advantages = advantages[:, None].expand_as(new_logprobs)[policy_tokens].detach()
    unclipped = ratio * token_advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * token_advantages
    relative = -torch.minimum(unclipped, clipped).mean()
    cloning = (
        -new_logprobs[reference_tokens].mean()
        if torch.any(reference_tokens)
        else new_logprobs.sum() * 0.0
    )
    total = advantage_weight * relative + behavior_cloning_weight * cloning
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
    return TorchJointResult(
        total=total,
        relative_advantage=relative,
        behavior_cloning=cloning,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )
