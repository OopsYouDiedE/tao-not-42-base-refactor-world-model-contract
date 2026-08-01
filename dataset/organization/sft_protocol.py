"""MineStudio 轨迹 SFT 的公开提示词与 action-first 回答协议。"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from tao.protocols.action import decode_action_sequence

FUTURE_TASKS = {"history_to_future_action", "single_frame_intent_to_action"}
GENERIC_REVIEW_REASON = "非 GUI 鼠标微动按动作块中点分区"

TASK_PROMPTS: dict[str, str] = {
    "demonstration_optimization": (
        "The images and raw action blocks form one chronological Minecraft demonstration. "
        "Rewrite it as a cleaner action sequence while preserving visible intent and causal "
        "order. Return one block per adjacent image pair and exactly match the supplied tick count "
        "for every block. One semicolon is one 50 ms tick. Do not shorten duration-sensitive held "
        "actions such as mining, attacking, moving, drawing a bow, eating, or continuous use. "
        "Remove only visually unsupported camera jitter and preserve GUI click order."
    ),
    "image_sequence_to_action": (
        "The images are consecutive Minecraft observations in chronological order. Infer one "
        "reasonable action sequence that produced every adjacent transition. Return one valid "
        "action block for each adjacent image pair, with each block exactly matching its supplied "
        "tick count. One semicolon is one 50 ms tick. Keep movement, mining, attacking, drawing, "
        "eating, and continuous use held for the required duration. Use visible camera "
        "displacement "
        "to infer meaningful mouse direction, omit unsupported 1-2 pixel jitter, and preserve GUI "
        "click order."
    ),
    "history_to_future_action": (
        "The images are past Minecraft observations in chronological order. Infer one reasonable "
        "future action block. Choose a suitable number of 50 ms ticks from the visible action type "
        "and required duration instead of waiting for a supplied target length. Keep brief actions "
        "short; sustained movement, mining, attacking, drawing, eating, or continuous use may last "
        "up to 60 ticks. Omit unsupported 1-2 pixel camera jitter and do not invent GUI clicks or "
        "auxiliary keys without visual evidence."
    ),
    "single_frame_intent_to_action": (
        "The image is the current Minecraft observation and the intent is supplied as text. Infer "
        "one reasonable future action block that advances this intent. Choose a suitable number of "
        "50 ms ticks from the action type and required duration instead of waiting for a supplied "
        "target length. Keep brief actions short; sustained movement, mining, bow drawing, eating, "
        "or continuous use may last up to 60 ticks. Omit unsupported 1-2 pixel camera jitter and "
        "preserve GUI click order."
    ),
}


def sanitize_intent(intent: str) -> str:
    """移除意图中的参考答案 tick 泄漏，同时保留方向等语义。"""
    text = re.sub(r"[（(]\s*\d+\s*ticks?\s*[，,]?\s*", "（", intent, flags=re.IGNORECASE)
    text = re.sub(r"[，,]\s*\d+\s*ticks?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\d+\s*ticks?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("（）", "").replace("()", "")
    return text.strip()


def normalize_question(question: dict[str, Any]) -> dict[str, Any]:
    """把历史题面升级为当前 action-first 协议。"""
    normalized = copy.deepcopy(question)
    task_type = normalized.get("task_type")
    if task_type in TASK_PROMPTS:
        normalized["prompt"] = TASK_PROMPTS[task_type]
    inputs = normalized.setdefault("inputs", {})
    if task_type == "single_frame_intent_to_action" and inputs.get("intent"):
        inputs["intent"] = sanitize_intent(str(inputs["intent"]))
    normalized.setdefault("output_contract", {}).update(
        {
            "type": "action_first_text",
            "action_payload": "JSON array of action-block strings",
            "action_boundary": "the closing bracket of the leading JSON array",
            "trailing_field": "newline followed by Reason: and a concise explanation",
            "trailing_field_may_be_truncated": True,
        }
    )
    return normalized


def format_question_prompt(question: dict[str, Any]) -> str:
    """构造训练和推理共用的公开提示词。"""
    question = normalize_question(question)
    prompt = question["prompt"]
    inputs = question.get("inputs", {})
    ticks = inputs.get("action_block_ticks")
    if ticks and question.get("task_type") not in FUTURE_TASKS:
        prompt += "\nRequired action-block tick counts: " + json.dumps(ticks)
    prompt += (
        "\nAction format example for a 3-tick block: "
        '"<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>". '
        "Each JSON array item must be one string action block; do not return nested tick arrays."
        "\nOutput the complete executable JSON action array first. Then start a new line with "
        '"Reason:" and briefly explain the visual evidence, intent, and duration choice. The '
        "action array must remain independently parseable because generation may stop after it."
    )
    raw = inputs.get("raw_action_sequence")
    if raw:
        prompt += "\nRaw action sequence:\n" + json.dumps(raw, ensure_ascii=False)
    intent = inputs.get("intent")
    if intent:
        prompt += f"\nIntent: {intent}"
    return prompt


def _action_summary(sequence: list[str]) -> tuple[str, int]:
    keys: set[str] = set()
    mouse = False
    total_ticks = 0
    for block in sequence:
        ticks = decode_action_sequence(block).ticks
        total_ticks += len(ticks)
        for tick in ticks:
            keys.update(tick.keys)
            mouse = mouse or tick.mouse != (0, 0)
    labels = {
        "W": "前进",
        "S": "后退",
        "A": "向左移动",
        "D": "向右移动",
        "space": "跳跃",
        "ctrl": "疾跑",
        "shift": "潜行",
        "MouseLeft": "持续主要操作",
        "MouseRight": "使用或放置物品",
    }
    actions = [labels[key] for key in labels if key in keys]
    if mouse:
        actions.append("调整视角或光标")
    return "、".join(actions) or "保持当前状态", total_ticks


def training_reason(question: dict[str, Any], answer: dict[str, Any]) -> str:
    """返回行为理由；过滤只描述审核算法的理由。"""
    existing = str(answer.get("answer_reason", "")).strip()
    if existing and GENERIC_REVIEW_REASON not in existing:
        return existing
    sequence = answer["reference_action_sequence"]
    summary, total_ticks = _action_summary(sequence)
    task_type = question.get("task_type")
    intent = question.get("inputs", {}).get("intent", "")
    if task_type == "single_frame_intent_to_action" and intent:
        return (
            f"当前动作推进“{sanitize_intent(str(intent))}”：执行{summary}。"
            f"该动作类型需要连续输入，因此选择 {total_ticks} 个 50 ms tick。"
        )
    if task_type == "history_to_future_action":
        return (
            f"历史画面支持延续已经建立的操作，接下来执行{summary}。"
            f"根据持续动作的需要选择 {total_ticks} 个 50 ms tick，并在之后重新观察。"
        )
    return f"相邻画面的可见变化与{summary}一致；动作持续时间与各图像区间保持一致。"


def format_assistant_response(question: dict[str, Any], answer: dict[str, Any]) -> str:
    """动作数组位于最前方；其后的理由允许在运行时截断。"""
    actions = json.dumps(answer["reference_action_sequence"], ensure_ascii=False)
    return f"{actions}\nReason: {training_reason(question, answer)}"
