"""轨迹答案审核 RLHF 的候选、输出和奖励合同。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

DECISIONS = {"approve", "revise", "reject"}
SCORE_FIELDS = (
    "visual_answerability",
    "action_validity",
    "duration_consistency",
    "causal_consistency",
    "gui_order",
)


@dataclass(frozen=True)
class ReviewCandidate:
    answer: dict[str, Any]
    expected_decision: Literal["approve", "reject"]
    candidate_origin: Literal["dual_review_approved", "synthetic_mutation"]
    mutation_type: str | None = None


def make_review_candidate(answer: dict[str, Any], *, mutate: bool) -> ReviewCandidate:
    """从双审通过答案构造正候选或带来源标记的合成错误候选。"""
    copied = json.loads(json.dumps(answer, ensure_ascii=False))
    if not mutate:
        return ReviewCandidate(copied, "approve", "dual_review_approved")
    sequence = copied.get("reference_action_sequence")
    if not isinstance(sequence, list) or not sequence or not isinstance(sequence[0], str):
        raise ValueError("审核答案缺少 reference_action_sequence")
    marker = "<|action_end|>"
    if marker not in sequence[0]:
        raise ValueError("动作块缺少结束标记")
    sequence[0] = sequence[0].replace(marker, "; Drop <|action_end|>", 1)
    return ReviewCandidate(copied, "reject", "synthetic_mutation", "unsupported_key")


def reference_review(candidate: ReviewCandidate, *, wording: int = 0) -> str:
    """生成两种等价专家审核输出。"""
    valid = candidate.expected_decision == "approve"
    scores = {field: int(valid) for field in SCORE_FIELDS}
    if valid:
        reason = (
            "候选动作与双审通过答案一致，动作格式、持续时长和因果顺序均满足协议。"
            if wording == 0
            else "画面、题面与候选动作相互一致，未发现动作协议或时序错误。"
        )
    else:
        reason = (
            "候选由 synthetic_mutation 生成，包含画面和题面均不支持的 Drop 按键。"
            if wording == 0
            else "检测到 unsupported_key：候选额外加入无视觉依据的 Drop 动作。"
        )
    return json.dumps(
        {"decision": candidate.expected_decision, "scores": scores, "reasons": [reason]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_review(text: str) -> dict[str, Any] | None:
    """解析严格审核 JSON；允许模型在 JSON 后产生少量文本。"""
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("decision") not in DECISIONS:
        return None
    scores = value.get("scores")
    reasons = value.get("reasons")
    if not isinstance(scores, dict) or set(scores) != set(SCORE_FIELDS):
        return None
    if any(
        not isinstance(scores[field], int) or scores[field] not in (0, 1) for field in SCORE_FIELDS
    ):
        return None
    if not isinstance(reasons, list) or not reasons or not all(isinstance(x, str) for x in reasons):
        return None
    return value


def score_review(text: str, candidate: ReviewCandidate) -> tuple[float, dict[str, Any]]:
    """按确定性合同评分，并对错误批准合成坏答案施加强惩罚。"""
    parsed = parse_review(text)
    if parsed is None:
        return -40.0, {"json_valid": False, "decision_correct": False}
    decision = parsed["decision"]
    correct = decision == candidate.expected_decision
    reward = 5.0 + (100.0 if correct else -60.0)
    false_approve = candidate.expected_decision == "reject" and decision == "approve"
    false_reject = candidate.expected_decision == "approve" and decision in {"reject", "revise"}
    if false_approve:
        reward -= 40.0
    reasons = " ".join(parsed["reasons"]).lower()
    mutation_found = bool(
        candidate.mutation_type
        and (candidate.mutation_type.lower() in reasons or "drop" in reasons or "无依据" in reasons)
    )
    if candidate.mutation_type and mutation_found:
        reward += 10.0
    expected_score = int(candidate.expected_decision == "approve")
    score_matches = sum(parsed["scores"][field] == expected_score for field in SCORE_FIELDS)
    reward += float(score_matches * 2)
    return reward, {
        "json_valid": True,
        "decision_correct": correct,
        "false_approve": false_approve,
        "false_reject": false_reject,
        "mutation_found": mutation_found,
        "score_matches": score_matches,
    }


def relative_advantages(rewards: list[float]) -> list[float]:
    """组内中心化奖励；保留奖励量纲以匹配现有 clipped 目标。"""
    if len(rewards) != 8:
        raise ValueError("审核组必须正好包含 8 条候选")
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]
