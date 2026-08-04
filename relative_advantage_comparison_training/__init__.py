"""同策略多分支相对优势比较样本。"""

from .comparison import ComparisonSample, build_comparison_group
from .objectives import (
    JointObjective,
    JointObjectiveWeights,
    JointResult,
    TorchJointResult,
    VisionGeometry,
    camera_degrees_after_resize,
    clipped_token_joint_objective,
    grouped_relative_advantages,
    masked_mean,
    plan_gemma4_geometry,
    torch_joint_objective,
)
from .policy_rollout import PolicyGeneration, generate_policy_rollouts, policy_prompt
from .rollouts import (
    RolloutSample,
    load_execution_group,
    masks,
    require_on_policy_logprobs,
)

__all__ = [
    "ComparisonSample",
    "JointObjective",
    "JointObjectiveWeights",
    "JointResult",
    "PolicyGeneration",
    "RolloutSample",
    "TorchJointResult",
    "VisionGeometry",
    "build_comparison_group",
    "camera_degrees_after_resize",
    "clipped_token_joint_objective",
    "generate_policy_rollouts",
    "grouped_relative_advantages",
    "load_execution_group",
    "masked_mean",
    "masks",
    "plan_gemma4_geometry",
    "policy_prompt",
    "require_on_policy_logprobs",
    "torch_joint_objective",
]
