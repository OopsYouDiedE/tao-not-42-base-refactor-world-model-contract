import numpy as np
import pytest
import torch

from train.gemma_vision_rlhf import top_half_training_mask
from train.objectives import (
    JointObjective,
    JointObjectiveWeights,
    clipped_token_joint_objective,
    grouped_relative_advantages,
    masked_mean,
)


def test_grouped_relative_advantages_are_independent() -> None:
    result = grouped_relative_advantages([1, 2, 3, 10, 10], ["a", "a", "a", "b", "b"])
    np.testing.assert_allclose(result[:3], [-1.22474337, 0, 1.22474337], rtol=1e-5)
    np.testing.assert_array_equal(result[3:], [0, 0])


def test_relative_advantage_can_only_center_rewards() -> None:
    np.testing.assert_array_equal(
        grouped_relative_advantages([2, 6], [0, 0], normalize=False), [-2, 2]
    )


def test_relative_advantage_rejects_single_candidate_group() -> None:
    with pytest.raises(ValueError, match="至少需要两个样本"):
        grouped_relative_advantages([1], ["only"])


def test_masked_behavior_cloning_mean() -> None:
    assert masked_mean([1, 100, 3], [1, 0, 1]) == 2


@pytest.mark.parametrize("weights", [JointObjectiveWeights(1, 0), JointObjectiveWeights(0, 1)])
def test_joint_objective_supports_single_training_mode(weights: JointObjectiveWeights) -> None:
    result = JointObjective(weights)(2, 5)
    assert result.total == weights.relative_advantage * 2 + weights.behavior_cloning * 5


def test_joint_objective_combines_and_reports_components() -> None:
    result = JointObjective(JointObjectiveWeights(0.25, 2))(4, 3)
    assert result.total == 7
    assert result.relative_advantage == 4
    assert result.behavior_cloning == 3


@pytest.mark.parametrize("values", [(-1, 1), (1, -1), (0, 0)])
def test_joint_objective_rejects_invalid_weights(values: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        JointObjectiveWeights(*values)


def test_clipped_token_joint_objective_uses_policy_ratio_and_reference_bc() -> None:
    new = torch.nn.Parameter(torch.full((8, 3), -1.0))
    old = torch.full((8, 3), -1.0)
    mask = torch.tensor([[True, True, False]] * 8)
    policy = torch.tensor([False, False, True, True, False, False, False, False])
    reference = torch.tensor([True, True, False, False, False, False, False, False])
    advantages = torch.tensor([0.0, 0.0, -1.0, -0.5, 0.25, 0.5, 1.0, 2.0])

    result = clipped_token_joint_objective(
        new,
        old,
        mask,
        advantages,
        policy,
        reference,
    )
    result.total.backward()

    assert result.approximate_kl is not None
    assert result.clip_fraction is not None
    assert torch.isclose(result.approximate_kl, torch.tensor(0.0))
    assert torch.isclose(result.clip_fraction, torch.tensor(0.0))
    assert new.grad is not None
    assert torch.all(new.grad[:2, :2] < 0)
    assert torch.all(new.grad[:2, 2] == 0)
    assert torch.all(new.grad[:, 2] == 0)
    assert torch.all(new.grad[4:] == 0)


def test_clipped_token_joint_objective_clips_large_policy_update() -> None:
    new = torch.full((8, 1), -1.0)
    old = new.clone()
    new[2:] += torch.log(torch.tensor(2.0))
    policy = torch.tensor([False, False, True, True, True, True, False, False])
    result = clipped_token_joint_objective(
        new,
        old,
        torch.ones_like(new, dtype=torch.bool),
        torch.ones(8),
        policy,
        torch.zeros(8, dtype=torch.bool),
    )
    assert result.clip_fraction is not None
    assert torch.isclose(result.clip_fraction, torch.tensor(1.0))


def test_top_half_training_mask_selects_four_highest_rewards() -> None:
    from train.rollout_contract import RolloutSample

    samples = [
        RolloutSample("g", f"C{i}", "policy_sample", "a", reward, 0.0, (), 1, 1)
        for i, reward in enumerate([2.0, 8.0, 1.0, 7.0, 6.0, 3.0, 5.0, 4.0])
    ]
    assert top_half_training_mask(samples).tolist() == [
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
    ]
