"""Curriculum snapshot admission and continuation contracts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

DIMENSIONS = (
    "stage",
    "inventory_loadout",
    "biome_terrain",
    "hazard",
    "health_hunger",
    "route",
    "strategy",
    "version_mechanic",
)


class CourseStatus(StrEnum):
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
    requirements: CourseRequirements, capabilities: SnapshotCapabilities
) -> None:
    missing = [
        name
        for name in ("entities", "scheduled_ticks", "cross_dimension")
        if getattr(requirements, name) and not getattr(capabilities, name)
    ]
    if missing:
        raise ValueError("snapshot cannot fairly reset required state: " + ", ".join(missing))


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
    if window.objective_complete:
        return ContinuationDecision(CourseStatus.FEASIBLE, False, window.stable_checkpoint)
    if window.physically_impossible:
        return ContinuationDecision(CourseStatus.INFEASIBLE, False, False)
    if window.missing_prerequisites:
        return ContinuationDecision(CourseStatus.PREPARATION_REQUIRED, False, False)
    if window.net_progress > 0 and window.ticks_since_progress < window.stagnation_limit:
        return ContinuationDecision(CourseStatus.PROGRESSING, True, window.stable_checkpoint)
    return ContinuationDecision(CourseStatus.UNKNOWN, False, False)


@dataclass(frozen=True)
class Capability:
    capability_id: str
    title: str
    stage: str
    prerequisites: tuple[str, ...]
    dimensions: dict[str, str]
    version_mechanic: str
    task: str


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    provenance: str
    feasible: bool
    feasibility_evidence: tuple[str, ...]
    dimensions: dict[str, str]
    source_run: str | None = None
    parent_snapshot_id: str | None = None
    failure_count: int = 0


def validate_snapshot(record: SnapshotRecord) -> list[str]:
    problems = []
    if not record.snapshot_id:
        problems.append("missing snapshot id")
    if not record.provenance:
        problems.append("missing provenance")
    if not record.feasible:
        problems.append("snapshot is not feasibility-validated")
    if not record.feasibility_evidence:
        problems.append("missing observation or successful-end-state evidence")
    missing = [dimension for dimension in DIMENSIONS if dimension not in record.dimensions]
    if missing:
        problems.append("missing dimensions: " + ", ".join(missing))
    return problems


def capability_eligible(snapshot: SnapshotRecord, capability: Capability) -> bool:
    if validate_snapshot(snapshot):
        return False
    return all(
        snapshot.dimensions.get(key) == expected
        or expected in {"surface_or_cave", "end_combat_variants"}
        for key, expected in capability.dimensions.items()
    )


def stratified_sample(records: Iterable[SnapshotRecord], *, limit: int) -> list[SnapshotRecord]:
    remaining, selected = [record for record in records if not validate_snapshot(record)], []
    seen: dict[str, Counter[str]] = defaultdict(Counter)
    while remaining and len(selected) < limit:
        chosen = max(
            remaining,
            key=lambda record: (
                sum(seen[key][value] == 0 for key, value in record.dimensions.items()),
                record.failure_count,
                record.snapshot_id,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        for dimension, value in chosen.dimensions.items():
            seen[dimension][value] += 1
    return selected


def coverage_matrix(records: Iterable[SnapshotRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if not validate_snapshot(record):
            for dimension, value in record.dimensions.items():
                result[dimension][value] += 1
    return {dimension: dict(sorted(values.items())) for dimension, values in sorted(result.items())}
