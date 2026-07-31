import numpy as np
import pytest

from train.objectives import (
    JointObjective,
    JointObjectiveWeights,
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
