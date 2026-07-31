"""与模型实现解耦的训练目标。"""

from train.objectives.behavior_cloning import masked_mean
from train.objectives.joint import JointObjective, JointObjectiveWeights, JointResult
from train.objectives.relative_advantage import grouped_relative_advantages
from train.objectives.torch_joint import TorchJointResult, torch_joint_objective

__all__ = [
    "JointObjective",
    "JointObjectiveWeights",
    "JointResult",
    "grouped_relative_advantages",
    "masked_mean",
    "TorchJointResult",
    "torch_joint_objective",
]
