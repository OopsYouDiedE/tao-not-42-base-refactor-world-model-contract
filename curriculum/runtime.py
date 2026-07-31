"""与任务名称无关的课程续跑和快照准入合同。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CourseStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    PROGRESSING = "PROGRESSING"
    UNKNOWN = "UNKNOWN"
    PREPARATION_REQUIRED = "PREPARATION_REQUIRED"
    INFEASIBLE = "INFEASIBLE"


@dataclass(frozen=True)
class SnapshotCapabilities:
    player: bool = True
    static_blocks: bool = True
    entities: bool = False
    scheduled_ticks: bool = False
    cross_dimension: bool = False


@dataclass(frozen=True)
class CourseRequirements:
    entities: bool = False
    scheduled_ticks: bool = False
    cross_dimension: bool = False


def assert_snapshot_eligible(
    requirements: CourseRequirements,
    capabilities: SnapshotCapabilities,
) -> None:
    missing = [
        name
        for name in ("entities", "scheduled_ticks", "cross_dimension")
        if getattr(requirements, name) and not getattr(capabilities, name)
    ]
    if missing:
        raise ValueError("当前快照合同不能公平复位课程能力：" + ", ".join(missing))


@dataclass(frozen=True)
class ProgressWindow:
    objective_complete: bool
    physically_impossible: bool
    missing_prerequisites: bool
    metric_start: float
    metric_end: float
    larger_is_better: bool
    stable_checkpoint: bool
    ticks_since_progress: int
    stagnation_limit: int = 32

    @property
    def net_progress(self) -> float:
        delta = self.metric_end - self.metric_start
        return delta if self.larger_is_better else -delta


@dataclass(frozen=True)
class ContinuationDecision:
    status: CourseStatus
    extend_budget: bool
    save_checkpoint: bool


def decide_continuation(window: ProgressWindow) -> ContinuationDecision:
    """只凭任务提供的度量和物理证据决策，不包含伐木或采矿模板。"""
    if window.objective_complete:
        return ContinuationDecision(CourseStatus.FEASIBLE, False, window.stable_checkpoint)
    if window.physically_impossible:
        return ContinuationDecision(CourseStatus.INFEASIBLE, False, False)
    if window.missing_prerequisites:
        return ContinuationDecision(CourseStatus.PREPARATION_REQUIRED, False, False)
    progressing = window.net_progress > 0 and window.ticks_since_progress < window.stagnation_limit
    if progressing:
        return ContinuationDecision(
            CourseStatus.PROGRESSING,
            True,
            window.stable_checkpoint,
        )
    return ContinuationDecision(CourseStatus.UNKNOWN, False, False)
