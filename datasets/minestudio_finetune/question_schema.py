"""三类 MineStudio 轨迹训练题的稳定字段与审核规则。"""

from __future__ import annotations

from typing import Literal, TypeAlias

TaskType: TypeAlias = Literal[
    "demonstration_optimization",
    "image_to_action",
    "history_to_future_action",
]

TASK_TYPES: tuple[TaskType, ...] = (
    "demonstration_optimization",
    "image_to_action",
    "history_to_future_action",
)

TASK_PROMPTS: dict[TaskType, str] = {
    "demonstration_optimization": (
        "The images and raw action blocks form one chronological Minecraft demonstration. "
        "Rewrite the action sequence into a cleaner demonstration while preserving the visible "
        "intent and causal order. Remove isolated control noise, keep necessary movement and "
        "interaction, and return only a JSON array of valid action blocks."
    ),
    "image_to_action": (
        "Given only the current Minecraft image, propose one reasonable action sequence for the "
        "next 200 ms. The action does not need to be uniquely optimal. Return only a JSON array "
        "containing one valid action block."
    ),
    "history_to_future_action": (
        "The images are past observations in chronological order and contain no action labels. "
        "Infer one reasonable action sequence for the next 200 ms. Return only a JSON array "
        "containing one valid action block."
    ),
}

OUTPUT_CONTRACT = {
    "type": "json_array",
    "item": "variable-length named-token action block",
    "action_markers": ["<|action_start|>", "<|action_end|>"],
    "chunk_count": "variable; MineStudio references use four 50 ms ticks",
    "chunk_duration_ms": 50,
    "mouse": (
        "Mouse dx dy inside the tick; it moves the camera in gameplay and the cursor in GUI"
    ),
    "mixing_guidance": "Prefer standalone Mouse unless keys and mouse execute together",
}

REVIEW_DIMENSIONS: dict[str, str] = {
    "source_integrity": "图片与动作来自同一 episode，帧号合法，时间顺序严格递增。",
    "no_temporal_leakage": "预测题不包含目标动作区间内的图片、动作或未来元数据。",
    "visual_answerability": (
        "题面图像足以支持一种合理动作；GUI 题可用可见光标状态、Mouse 相对移动与鼠标键表达。"
    ),
    "demonstration_quality": "参考轨迹意图连贯，没有孤立误触、异常视角跳变或明显无效片段。",
    "prompt_contract_match": "题目输入、提示词和 JSON 动作输出契约一致。",
    "ambiguity_disclosed": "合理动作存在多解；参考动作被标为人类示范，而非唯一正确答案。",
    "safety_and_privacy": "图片不含账号、聊天、服务器地址或其他不应进入训练的数据。",
}

HARD_REJECTION_REASONS = frozenset({
    "missing_image",
    "cross_episode_mismatch",
    "non_monotonic_frames",
    "future_leakage",
    "invalid_action_contract",
    "corrupt_image",
})
