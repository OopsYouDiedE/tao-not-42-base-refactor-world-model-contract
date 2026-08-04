"""Stable episode splitting and action-first SFT conversion contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

HoldoutLevel = Literal["prefix", "episode"]
_EPISODE_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<session>[0-9a-f]{12})-(?P<date>\d{8})-(?P<time>\d{6})$"
)


@dataclass(frozen=True)
class EpisodeIdentity:
    episode: str
    prefix: str
    session: str
    date: str
    time: str


@dataclass
class SplitResult:
    holdout_level: str
    train_episodes: list[str] = field(default_factory=list)
    validation_episodes: list[str] = field(default_factory=list)
    train_frames: int = 0
    validation_frames: int = 0
    validation_prefixes: list[str] = field(default_factory=list)
    achieved_validation_ratio: float = 0.0
    target_validation_ratio: float = 0.0


def parse_episode_identity(episode: str) -> EpisodeIdentity:
    matched = _EPISODE_PATTERN.match(episode)
    if matched is None:
        raise ValueError(f"invalid episode identity: {episode!r}")
    return EpisodeIdentity(
        episode, matched["prefix"], matched["session"], matched["date"], matched["time"]
    )


def _stable_order(names: list[str], seed: int) -> list[str]:
    return sorted(names, key=lambda name: hashlib.md5(f"{seed}:{name}".encode()).hexdigest())


def _select_groups_by_frames(group_frames: dict[str, int], target_ratio: float) -> list[str]:
    names, total = sorted(group_frames), sum(group_frames.values())
    if total == 0:
        raise ValueError("total frame count is zero")
    target = total * target_ratio
    if len(names) <= 20:
        best: tuple[float, tuple[str, ...]] = (float("inf"), ())
        for size in range(1, len(names)):
            for candidate in combinations(names, size):
                best = min(
                    best, (abs(sum(group_frames[name] for name in candidate) - target), candidate)
                )
        return sorted(best[1])
    selected, accumulated = [], 0
    for name in sorted(names, key=lambda value: -group_frames[value]):
        if accumulated + group_frames[name] <= target or not selected:
            selected.append(name)
            accumulated += group_frames[name]
    return sorted(selected)


def build_split(
    *,
    episode_frames: dict[str, int],
    holdout_level: HoldoutLevel = "prefix",
    validation_ratio: float = 0.1,
    seed: int = 3407,
    output_path: Path | None = None,
) -> SplitResult:
    if not 0 < validation_ratio < 1 or not episode_frames:
        raise ValueError("validation_ratio must be in (0, 1) and episodes cannot be empty")
    frames, episodes = dict(episode_frames), sorted(episode_frames)
    identities = {name: parse_episode_identity(name) for name in episodes}
    if holdout_level == "prefix":
        group_frames: Counter[str] = Counter()
        for name in episodes:
            group_frames[identities[name].prefix] += frames[name]
        held_out = set(_select_groups_by_frames(dict(group_frames), validation_ratio))
        validation = [name for name in episodes if identities[name].prefix in held_out]
    elif holdout_level == "episode":
        validation, accumulated, target = [], 0, sum(frames.values()) * validation_ratio
        for name in _stable_order(episodes, seed):
            if accumulated >= target:
                break
            validation.append(name)
            accumulated += frames[name]
    else:
        raise ValueError(f"unknown holdout level: {holdout_level!r}")
    validation_set = set(validation)
    train = [name for name in episodes if name not in validation_set]
    if not train or not validation:
        raise ValueError("split produced an empty subset")
    validation_frames, total = sum(frames[name] for name in validation), sum(frames.values())
    result = SplitResult(
        holdout_level,
        sorted(train),
        sorted(validation),
        total - validation_frames,
        validation_frames,
        sorted({identities[name].prefix for name in validation}),
        validation_frames / total,
        validation_ratio,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def load_split(path: Path) -> SplitResult:
    return SplitResult(**json.loads(Path(path).read_text(encoding="utf-8")))


FUTURE_TASKS = {"history_to_future_action", "single_frame_intent_to_action"}
TASK_PROMPTS = {
    "demonstration_optimization": "Rewrite the chronological Minecraft demonstration as a cleaner executable action sequence while preserving visible intent, causal order, and supplied tick counts.",
    "image_sequence_to_action": "Infer one executable action block for each adjacent Minecraft image transition and exactly match supplied tick counts.",
    "history_to_future_action": "Infer one reasonable future Minecraft action block from the chronological observation history.",
    "single_frame_intent_to_action": "Infer one reasonable future Minecraft action block that advances the supplied intent.",
}


def sanitize_intent(intent: str) -> str:
    text = re.sub(r"[（(]\s*\d+\s*ticks?\s*[，,]?\s*", "（", intent, flags=re.IGNORECASE)
    text = re.sub(r"[，,]?\s*\d+\s*ticks?\s*", "", text, flags=re.IGNORECASE)
    return text.replace("（）", "").replace("()", "").strip()


def normalize_question(question: dict[str, Any]) -> dict[str, Any]:
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
    existing = str(answer.get("answer_reason", "")).strip()
    return (
        existing
        or "The action sequence follows the visible transition, intent, and required duration."
    )


def format_assistant_response(question: dict[str, Any], answer: dict[str, Any]) -> str:
    actions = json.dumps(answer["reference_action_sequence"], ensure_ascii=False)
    return f"{actions}\nReason: {training_reason(question, answer)}"
