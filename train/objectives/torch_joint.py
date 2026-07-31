"""可反向传播的 2+6 联合目标。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchJointResult:
    total: torch.Tensor
    relative_advantage: torch.Tensor
    behavior_cloning: torch.Tensor


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
