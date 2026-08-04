"""Deterministic model-judgment review candidates and reward contract."""

from __future__ import annotations

import copy
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
    copied = copy.deepcopy(answer)
    if not mutate:
        return ReviewCandidate(copied, "approve", "dual_review_approved")
    sequence = copied.get("reference_action_sequence")
    if not isinstance(sequence, list) or not sequence or not isinstance(sequence[0], str):
        raise ValueError("answer lacks reference_action_sequence")
    marker = "</action>"
    if marker not in sequence[0]:
        raise ValueError("standard action sequence lacks end marker")
    sequence[0] = sequence[0].replace(marker, " ; Q</action>", 1)
    return ReviewCandidate(copied, "reject", "synthetic_mutation", "unsupported_key")


def reference_review(candidate: ReviewCandidate, *, wording: int = 0) -> str:
    valid = candidate.expected_decision == "approve"
    reason = (
        (
            "candidate matches the approved action and timing contract"
            if wording == 0
            else "visual evidence, prompt, and action sequence are consistent"
        )
        if valid
        else (
            "synthetic mutation added an unsupported Q action"
            if wording == 0
            else "unsupported_key: Q has no visual basis"
        )
    )
    return json.dumps(
        {
            "decision": candidate.expected_decision,
            "scores": {field: int(valid) for field in SCORE_FIELDS},
            "reasons": [reason],
        },
        separators=(",", ":"),
    )


def parse_review(text: str) -> dict[str, Any] | None:
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("decision") not in DECISIONS:
        return None
    scores, reasons = value.get("scores"), value.get("reasons")
    if not isinstance(scores, dict) or set(scores) != set(SCORE_FIELDS):
        return None
    normalized = {}
    for field in SCORE_FIELDS:
        score = scores[field]
        if isinstance(score, str) and score in {"0", "1"}:
            score = int(score)
        if not isinstance(score, int) or score not in (0, 1):
            return None
        normalized[field] = score
    if (
        not isinstance(reasons, list)
        or not reasons
        or not all(isinstance(reason, str) for reason in reasons)
    ):
        return None
    return {**value, "scores": normalized}


def score_review(text: str, candidate: ReviewCandidate) -> tuple[float, dict[str, Any]]:
    parsed = parse_review(text)
    if parsed is None:
        return -40.0, {"json_valid": False, "decision_correct": False}
    decision, expected = parsed["decision"], candidate.expected_decision
    correct = decision == expected
    reward = 5.0 + (100.0 if correct else -60.0)
    false_approve = expected == "reject" and decision == "approve"
    false_reject = expected == "approve" and decision in {"reject", "revise"}
    if false_approve:
        reward -= 40.0
    reasons = " ".join(parsed["reasons"]).lower()
    mutation_found = bool(
        candidate.mutation_type
        and (candidate.mutation_type.lower() in reasons or " q " in f" {reasons} ")
    )
    if mutation_found:
        reward += 10.0
    score_matches = sum(
        parsed["scores"][field] == int(expected == "approve") for field in SCORE_FIELDS
    )
    return reward + score_matches * 2.0, {
        "json_valid": True,
        "decision_correct": correct,
        "false_approve": false_approve,
        "false_reject": false_reject,
        "mutation_found": mutation_found,
        "score_matches": score_matches,
    }


def relative_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) != 8:
        raise ValueError("review group must contain exactly eight candidates")
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]
