"""四类多模态训练任务的可审计静态案例。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TaskType = Literal[
    "action_optimization",
    "inverse_action_generation",
    "future_action_choice",
    "macro_intent_classification",
]


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    task_type: TaskType
    images: tuple[str, ...]
    inputs: dict[str, Any]
    answer: Any
    answer_aware_assessment: str
    blind_answer: Any
    blind_assessment: str


GOAL_IMAGES = tuple(
    f"images/goal_control_{index}.jpg"
    for index in range(4)
)
FUTURE_IMAGES = tuple(
    f"images/future_control_{index}.jpg"
    for index in range(4)
)

OPTIMIZED_MINING_ACTION = (
    "<|action_start|> ; D MouseLeft ; D MouseLeft ; "
    "D MouseLeft Mouse 2 7 ; D MouseLeft Mouse 5 23 <|action_end|>"
)

DEMO_CASES = (
    DemoCase(
        case_id="action_optimization_001",
        task_type="action_optimization",
        images=GOAL_IMAGES,
        inputs={
            "objective": "持续采掘准星附近的石块，并保持向右贴近目标",
            "original_action": (
                "<|action_start|> ; D MouseLeft ; D ; "
                "D MouseLeft Mouse 2 7 ; D MouseLeft Mouse 5 23 <|action_end|>"
            ),
            "optimization_rule": (
                "保持四个 tick、移动方向和有效视角修正；消除会中断宏观意图的偶发漏按"
            ),
        },
        answer=OPTIMIZED_MINING_ACTION,
        answer_aware_assessment=(
            "合适。标准答案只补齐第二个 tick 的 MouseLeft，保持 D、时长和两个视角修正；"
            "它消除了采掘中断，没有引入图片无法支持的新动作。"
        ),
        blind_answer=OPTIMIZED_MINING_ACTION,
        blind_assessment="盲答与标准答案完全一致。",
    ),
    DemoCase(
        case_id="inverse_action_generation_001",
        task_type="inverse_action_generation",
        images=GOAL_IMAGES,
        inputs={
            "objective": "持续采掘准星附近的石块，并保持向右贴近目标",
            "optimization_rule": (
                "生成四个连续 tick；保持宏观动作连续，只保留图像支持的移动、交互和视角修正"
            ),
        },
        answer=OPTIMIZED_MINING_ACTION,
        answer_aware_assessment=(
            "可作为弱监督答案。连续图片与采掘目标支持 D 和 MouseLeft；精确鼠标数值来自"
            "示范轨迹，因此它是可复现示范，不应声明为唯一最优控制。"
        ),
        blind_answer=(
            "<|action_start|> ; D MouseLeft ; D MouseLeft ; "
            "D MouseLeft Mouse 2 7 ; D MouseLeft Mouse 5 23 <|action_end|>"
        ),
        blind_assessment="盲答命中标准示范；训练输入中没有 original_action 字段。",
    ),
    DemoCase(
        case_id="future_action_choice_001",
        task_type="future_action_choice",
        images=FUTURE_IMAGES,
        inputs={
            "choices": {
                "A": "四个 tick 保持静止",
                "B": "四个 tick 持续 W+space",
                "C": "四个 tick 持续 S",
                "D": "四个 tick 持续 MouseLeft",
            },
            "prediction_horizon_ms": 200,
        },
        answer="B",
        answer_aware_assessment=(
            "合适。标准答案对应轨迹中的后续示范动作。题面只提供过去图片，候选互异，"
            "没有暴露未来图片或动作。该标签表示示范选择，不表示唯一最优选择。"
        ),
        blind_answer="B",
        blind_assessment="盲答选择继续向前跳跃，与标准答案一致。",
    ),
    DemoCase(
        case_id="macro_intent_classification_001",
        task_type="macro_intent_classification",
        images=GOAL_IMAGES,
        inputs={
            "choices": {
                "A": "打开背包整理物品",
                "B": "脱离目标并向后撤退",
                "C": "持续采掘准星附近的石块",
                "D": "原地等待环境变化",
            },
        },
        answer="C",
        answer_aware_assessment=(
            "合适。连续画面围绕同一近距离方块目标，标准标签也有 mine_block:stone 事件证据。"
            "宏观标签不要求恢复逐 tick 动作，因此比精确逆动力学标签稳定。"
        ),
        blind_answer="C",
        blind_assessment="盲答识别为持续采掘石块，与标准答案及事件证据一致。",
    ),
)
