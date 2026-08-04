"""MineStudio v110 系列数据集的行为克隆课程转换合同。"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .utils import SplitResult, build_grouped_split

HoldoutLevel = Literal["prefix", "episode"]
_EPISODE_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<session>[0-9a-f]{12})-(?P<date>\d{8})-(?P<time>\d{6})$"
)


@dataclass(frozen=True)
class EpisodeIdentity:
    """MineStudio v110 episode 名称中编码的身份信息。"""

    episode: str
    prefix: str
    session: str
    date: str
    time: str


def parse_episode_identity(episode: str) -> EpisodeIdentity:
    """解析 v110 的 ``prefix-session-date-time`` episode 名称。"""
    matched = _EPISODE_PATTERN.match(episode)
    if matched is None:
        raise ValueError(f"invalid MineStudio v110 episode identity: {episode!r}")
    return EpisodeIdentity(
        episode=episode,
        prefix=matched["prefix"],
        session=matched["session"],
        date=matched["date"],
        time=matched["time"],
    )


def build_split(
    *,
    episode_frames: dict[str, int],
    holdout_level: HoldoutLevel = "prefix",
    validation_ratio: float = 0.1,
    seed: int = 3407,
    output_path: Path | None = None,
) -> SplitResult:
    """按 MineStudio v110 的前缀或 episode 合同划分数据集。"""
    identities = {episode: parse_episode_identity(episode) for episode in sorted(episode_frames)}
    result = build_grouped_split(
        episode_frames=episode_frames,
        episode_groups={episode: identity.prefix for episode, identity in identities.items()},
        holdout_level="group" if holdout_level == "prefix" else "episode",
        validation_ratio=validation_ratio,
        seed=seed,
        result_holdout_level=holdout_level,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        payload["validation_prefixes"] = payload.pop("validation_groups")
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_split(path: Path) -> SplitResult:
    """读取采用旧版 ``validation_prefixes`` 字段的 v110 划分结果。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["validation_groups"] = payload.pop("validation_prefixes", [])
    return SplitResult(**payload)


FUTURE_TASKS = {"history_to_future_action", "single_frame_intent_to_action"}
TASK_PROMPTS = {
    "demonstration_optimization": "Rewrite the chronological Minecraft demonstration as a cleaner executable action sequence while preserving visible intent, causal order, and supplied tick counts.",
    "image_sequence_to_action": "Infer one executable action block for each adjacent Minecraft image transition and exactly match supplied tick counts.",
    "history_to_future_action": "Infer one reasonable future Minecraft action block from the chronological observation history.",
    "single_frame_intent_to_action": "Infer one reasonable future Minecraft action block that advances the supplied intent.",
}


def sanitize_intent(intent: str) -> str:
    """清理 v110 人工意图文本中残留的 tick 标记和乱码。"""
    text = re.sub(r"[（(]\s*\d+\s*ticks?\s*[，,]?\s*", "（", intent, flags=re.IGNORECASE)
    text = re.sub(r"[，,]?\s*\d+\s*ticks?\s*", "", text, flags=re.IGNORECASE)
    return text.replace("（）", "").replace("()", "").strip()


def normalize_question(question: dict[str, Any]) -> dict[str, Any]:
    """将 v110 课程题面规范化为标准输入动作协议 v1 合同。"""
    normalized = copy.deepcopy(question)
    task = normalized.get("task_type")
    if task in TASK_PROMPTS:
        normalized["prompt"] = TASK_PROMPTS[task]
    inputs = normalized.setdefault("inputs", {})
    if task == "single_frame_intent_to_action" and inputs.get("intent"):
        inputs["intent"] = sanitize_intent(str(inputs["intent"]))
    normalized.setdefault("output_contract", {}).update(
        {
            "type": "standard_input_action_v1",
            "protocol": "standard-input-action/v1",
            "action_payload": "JSON array of complete Device/Tick/<action> sequences",
            "trailing_field": "newline followed by Reason:",
            "trailing_field_may_be_truncated": True,
        }
    )
    return normalized


def format_question_prompt(question: dict[str, Any]) -> str:
    """把 v110 结构化题面格式化为行为克隆 user 文本。"""
    question = normalize_question(question)
    prompt, inputs = question["prompt"], question.get("inputs", {})
    ticks = inputs.get("action_block_ticks")
    if ticks and question.get("task_type") not in FUTURE_TASKS:
        prompt += "\nRequired action-block tick counts: " + json.dumps(ticks)
    prompt += (
        "\nOutput a JSON array whose entries are complete standard-input-action/v1 "
        "Device/Tick/<action> sequences, then a new line beginning with Reason:."
    )
    if inputs.get("raw_action_sequence"):
        prompt += "\nRaw action sequence:\n" + json.dumps(
            inputs["raw_action_sequence"], ensure_ascii=False
        )
    if inputs.get("intent"):
        prompt += "\nIntent: " + str(inputs["intent"])
    return prompt


def training_reason(question: dict[str, Any], answer: dict[str, Any]) -> str:
    """返回 v110 监督答案已有的理由或稳定的缺省理由。"""
    del question
    existing = str(answer.get("answer_reason", "")).strip()
    return existing or (
        "The action sequence follows the visible transition, intent, and required duration."
    )


def format_assistant_response(question: dict[str, Any], answer: dict[str, Any]) -> str:
    """把 v110 参考动作和训练理由格式化为 assistant 文本。"""
    actions = json.dumps(answer["reference_action_sequence"], ensure_ascii=False)
    return f"{actions}\nReason: {training_reason(question, answer)}"
