"""复核相对优势比较中的均值、排序和选择结论。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from relative_advantage_comparison_training import ComparisonSample


@dataclass(frozen=True)
class ComparisonReview:
    valid: bool
    selected_trajectory_ids: tuple[str, ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_trajectory_ids"] = list(self.selected_trajectory_ids)
        value["issues"] = list(self.issues)
        return value


def review_comparison(samples: Iterable[ComparisonSample]) -> ComparisonReview:
    values = tuple(samples)
    if not values:
        raise ValueError("比较结果不能为空")
    issues: list[str] = []
    if abs(sum(item.relative_advantage for item in values)) > 1e-5:
        issues.append("relative_advantage_not_centered")
    best_score = max(item.score for item in values)
    expected = {item.trajectory_id for item in values if item.score == best_score}
    selected = {item.trajectory_id for item in values if item.selected}
    if selected != expected:
        issues.append("selection_mismatch")
    if any(item.rank < 1 for item in values):
        issues.append("invalid_rank")
    return ComparisonReview(
        valid=not issues,
        selected_trajectory_ids=tuple(sorted(selected)),
        issues=tuple(issues),
    )
