from pathlib import Path

import pytest
import torch

from train.objectives import torch_joint_objective
from train.rlhf.rollout_contract import load_execution_group, masks, require_on_policy_logprobs
from train.vision_geometry import camera_degrees_after_resize, plan_gemma4_geometry

RUN = Path("runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/execution.json")


def test_real_execution_loads_as_strict_2_plus_6() -> None:
    samples = load_execution_group(RUN)
    role_masks = masks(samples)
    assert sum(role_masks["reference_bc"]) == 2
    assert sum(role_masks["policy"]) == 6
    assert {sample.original_width for sample in samples} == {640}
    assert {sample.original_height for sample in samples} == {360}


def test_real_execution_has_aligned_on_policy_logprobs() -> None:
    samples = load_execution_group(RUN)

    require_on_policy_logprobs(samples)

    policies = [sample for sample in samples if sample.policy_eligible]
    assert all(len(sample.old_logprobs) == len(sample.response_token_ids) for sample in policies)


def test_torch_joint_objective_backpropagates_through_separate_roles() -> None:
    parameter = torch.nn.Parameter(torch.arange(1.0, 9.0))
    policy = torch.tensor([False, False, True, True, True, True, True, True])
    reference = ~policy
    advantages = torch.tensor([99.0, -99.0, -1.0, -0.5, 0.25, 0.5, 1.0, 2.0])
    result = torch_joint_objective(parameter.square(), advantages, policy, reference)
    result.total.backward()
    assert parameter.grad is not None
    assert torch.allclose(parameter.grad[:2], parameter[:2])
    assert torch.allclose(parameter.grad[2:], 2 * parameter[2:] * advantages[2:] / 6)


def test_gemma4_geometry_and_camera_contract() -> None:
    geometry = plan_gemma4_geometry(640, 360, 1.65)
    assert (geometry.resized_width, geometry.resized_height) == (1056, 576)
    assert (geometry.raw_patches, geometry.soft_tokens) == (2376, 264)
    assert camera_degrees_after_resize(-4.5, 9.0) == (-4.5, 9.0)
    with pytest.raises(ValueError, match="预算超限"):
        plan_gemma4_geometry(1920, 1080)
