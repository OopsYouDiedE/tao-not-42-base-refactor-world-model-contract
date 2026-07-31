"""相对优势与行为克隆的联合训练目标。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JointObjectiveWeights:
    relative_advantage: float = 1.0
    behavior_cloning: float = 1.0

    def __post_init__(self) -> None:
        if self.relative_advantage < 0 or self.behavior_cloning < 0:
            raise ValueError("联合目标权重不能为负数")
        if self.relative_advantage == 0 and self.behavior_cloning == 0:
            raise ValueError("至少一个联合目标权重必须大于零")


@dataclass(frozen=True)
class JointResult:
    total: float
    relative_advantage: float
    behavior_cloning: float


class JointObjective:
    """组合两个标量损失并保留日志所需分量。"""

    def __init__(self, weights: JointObjectiveWeights | None = None) -> None:
        self.weights = weights or JointObjectiveWeights()

    def __call__(self, relative_advantage_loss: float, behavior_cloning_loss: float) -> JointResult:
        advantage = float(relative_advantage_loss)
        cloning = float(behavior_cloning_loss)
        total = (
            self.weights.relative_advantage * advantage + self.weights.behavior_cloning * cloning
        )
        return JointResult(total=total, relative_advantage=advantage, behavior_cloning=cloning)
