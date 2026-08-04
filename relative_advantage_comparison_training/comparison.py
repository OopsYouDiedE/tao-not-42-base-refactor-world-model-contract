"""从同一策略、同一起点的轨迹审核结果构造比较组。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from interaction_trajectory_review_agents import TrajectoryReview


@dataclass(frozen=True)
class ComparisonSample:
    trajectory_id: str
    score: float
    relative_advantage: float
    rank: int
    selected: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_comparison_group(
    reviews: Iterable[TrajectoryReview],
) -> tuple[ComparisonSample, ...]:
    """以组内均值为基线计算相对优势，并稳定处理并列分数。"""
    values = tuple(reviews)
    if len(values) < 2:
        raise ValueError("相对优势比较组至少需要两条轨迹")
    mean_score = sum(item.quality_score for item in values) / len(values)
    ordered_scores = sorted({item.quality_score for item in values}, reverse=True)
    best_score = ordered_scores[0]
    ranks = {score: index + 1 for index, score in enumerate(ordered_scores)}
    return tuple(
        ComparisonSample(
            trajectory_id=item.trajectory_id,
            score=item.quality_score,
            relative_advantage=round(item.quality_score - mean_score, 6),
            rank=ranks[item.quality_score],
            selected=item.quality_score == best_score,
        )
        for item in values
    )
