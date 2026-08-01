"""Codex 教师动作与评分的严格本地合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tao.protocols.action import (
    MINECRAFT_KEYMAP,
    PROTOCOL_VERSION,
    ActionSegment,
    ActionSequence,
    ActionTick,
)
from tao.protocols.action.codec import MOUSE_DELTA_LIMIT, SCROLL_LIMIT

ALLOWED_KEYS = frozenset(MINECRAFT_KEYMAP.values())
SCORE_DIMENSIONS: dict[str, int] = {
    "task_progress": 35,
    "safety": 25,
    "visual_causal_consistency": 15,
    "temporal_correctness": 15,
    "action_efficiency": 10,
}


@dataclass(frozen=True)
class TeacherCandidate:
    candidate_id: str
    summary: str
    segments: tuple[ActionSegment, ...]
    ticks: tuple[ActionTick, ...]
    generation_audit: dict[str, Any]
    source_role: str = "codex_teacher"

    @property
    def description(self) -> str:
        return self.summary

    @property
    def action_text(self) -> str:
        return ActionSequence(self.ticks).to_text()


@dataclass(frozen=True)
class TeacherScore:
    candidate_id: str
    anonymous_id: str
    dimensions: dict[str, int]
    total: float
    rationale: str
    safety_flags: tuple[str, ...]


def generation_schema(horizon_ticks: int) -> dict[str, Any]:
    if horizon_ticks < 1:
        raise ValueError("horizon_ticks 必须大于零")
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "required": ["protocol", "horizon_ticks", "segments", "summary"],
        "properties": {
            "protocol": {"type": "string", "enum": [PROTOCOL_VERSION]},
            "horizon_ticks": {"type": "integer", "enum": [horizon_ticks]},
            "segments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["duration_ticks", "keys", "mouse", "scroll"],
                    "properties": {
                        "duration_ticks": {"type": "integer", "minimum": 1},
                        "keys": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": sorted(ALLOWED_KEYS)},
                        },
                        "mouse": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {
                                "type": "integer",
                                "minimum": -MOUSE_DELTA_LIMIT,
                                "maximum": MOUSE_DELTA_LIMIT,
                            },
                        },
                        "scroll": {
                            "type": "integer",
                            "minimum": -SCROLL_LIMIT,
                            "maximum": SCROLL_LIMIT,
                        },
                    },
                },
            },
            "summary": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    }


def scoring_schema(anonymous_ids: tuple[str, ...]) -> dict[str, Any]:
    if not anonymous_ids or len(set(anonymous_ids)) != len(anonymous_ids):
        raise ValueError("anonymous_ids 必须非空且唯一")
    dimension_properties = {
        name: {"type": "integer", "minimum": 0, "maximum": 5}
        for name in SCORE_DIMENSIONS
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "required": ["scores"],
        "properties": {
            "scores": {
                "type": "array",
                "minItems": len(anonymous_ids),
                "maxItems": len(anonymous_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["anonymous_id", "dimensions", "rationale", "safety_flags"],
                    "properties": {
                        "anonymous_id": {"type": "string", "enum": list(anonymous_ids)},
                        "dimensions": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(SCORE_DIMENSIONS),
                            "properties": dimension_properties,
                        },
                        "rationale": {"type": "string", "minLength": 1},
                        "safety_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            }
        },
    }


def compile_teacher_action(
    value: dict[str, Any],
    *,
    candidate_id: str,
    expected_horizon_ticks: int,
    generation_audit: dict[str, Any],
) -> TeacherCandidate:
    if set(value) != {"protocol", "horizon_ticks", "segments", "summary"}:
        raise ValueError("教师动作输出字段不符合 TAP 生成合同")
    if value["protocol"] != PROTOCOL_VERSION:
        raise ValueError(f"协议必须是 {PROTOCOL_VERSION}")
    if type(value["horizon_ticks"]) is not int or value["horizon_ticks"] != expected_horizon_ticks:
        raise ValueError("教师声明的 horizon_ticks 与请求不一致")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise ValueError("教师轨迹 summary 不能为空")
    raw_segments = value["segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("教师动作 segments 必须是非空数组")

    segments: list[ActionSegment] = []
    ticks: list[ActionTick] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict) or set(raw) != {
            "duration_ticks",
            "keys",
            "mouse",
            "scroll",
        }:
            raise ValueError(f"segment {index} 字段不符合合同")
        duration = raw["duration_ticks"]
        keys = raw["keys"]
        mouse = raw["mouse"]
        scroll = raw["scroll"]
        if type(duration) is not int or duration < 1:
            raise ValueError(f"segment {index} duration_ticks 必须是正整数")
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError(f"segment {index} keys 必须是字符串数组")
        if len(keys) != len(set(keys)) or not set(keys).issubset(ALLOWED_KEYS):
            raise ValueError(f"segment {index} 包含重复或未知按键")
        if (
            not isinstance(mouse, list)
            or len(mouse) != 2
            or any(type(item) is not int for item in mouse)
            or any(abs(item) > MOUSE_DELTA_LIMIT for item in mouse)
        ):
            raise ValueError(f"segment {index} mouse 超出 TAP 范围")
        if type(scroll) is not int or abs(scroll) > SCROLL_LIMIT:
            raise ValueError(f"segment {index} scroll 超出 TAP 范围")
        if scroll and any(key.isdigit() for key in keys):
            raise ValueError(f"segment {index} 不能同时滚轮和选择快捷栏")
        segment = ActionSegment(duration, tuple(keys), tuple(mouse), scroll)
        segments.append(segment)
        tick = ActionTick(keys=tuple(keys), mouse=tuple(mouse), scroll=scroll)
        ticks.extend(tick for _ in range(duration))
    if len(ticks) != expected_horizon_ticks:
        raise ValueError(
            f"教师动作要求恰好 {expected_horizon_ticks} tick，实际为 {len(ticks)}；不补齐也不截断"
        )
    return TeacherCandidate(
        candidate_id=candidate_id,
        summary=value["summary"].strip(),
        segments=tuple(segments),
        ticks=tuple(ticks),
        generation_audit=generation_audit,
    )


def parse_teacher_scores(
    value: dict[str, Any],
    *,
    anonymous_to_candidate: dict[str, str],
) -> tuple[TeacherScore, ...]:
    if set(value) != {"scores"} or not isinstance(value["scores"], list):
        raise ValueError("教师评分输出字段不符合合同")
    rows = value["scores"]
    if len(rows) != len(anonymous_to_candidate):
        raise ValueError("教师评分数量与候选数量不一致")
    seen: set[str] = set()
    parsed: list[TeacherScore] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "anonymous_id",
            "dimensions",
            "rationale",
            "safety_flags",
        }:
            raise ValueError("教师评分行字段不符合合同")
        anonymous_id = row["anonymous_id"]
        if anonymous_id not in anonymous_to_candidate or anonymous_id in seen:
            raise ValueError("教师评分包含未知或重复匿名候选")
        seen.add(anonymous_id)
        dimensions = row["dimensions"]
        if not isinstance(dimensions, dict) or set(dimensions) != set(SCORE_DIMENSIONS):
            raise ValueError("教师评分维度不完整")
        normalized: dict[str, int] = {}
        for name in SCORE_DIMENSIONS:
            score = dimensions[name]
            if type(score) is not int or not 0 <= score <= 5:
                raise ValueError(f"评分维度 {name} 必须是 0 到 5 的整数")
            normalized[name] = score
        rationale = row["rationale"]
        flags = row["safety_flags"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("评分理由不能为空")
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            raise ValueError("safety_flags 必须是字符串数组")
        total = sum(normalized[name] * weight / 5 for name, weight in SCORE_DIMENSIONS.items())
        parsed.append(
            TeacherScore(
                candidate_id=anonymous_to_candidate[anonymous_id],
                anonymous_id=anonymous_id,
                dimensions=normalized,
                total=round(total, 3),
                rationale=rationale.strip(),
                safety_flags=tuple(flags),
            )
        )
    return tuple(parsed)
