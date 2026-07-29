"""MineStudio 八方面 LoRA 多任务训练数据生成。"""

from bc_datasets.training_capability_demo.generator import (
    CAPABILITY_ASPECTS,
    action_contract_text,
    action_ticks,
    build_training_capability_demo,
    categorical_transition,
    coarse_inverse_dynamics,
    meaningful_events,
    state_transition,
)

__all__ = [
    "CAPABILITY_ASPECTS",
    "action_contract_text",
    "action_ticks",
    "build_training_capability_demo",
    "categorical_transition",
    "coarse_inverse_dynamics",
    "meaningful_events",
    "state_transition",
]
