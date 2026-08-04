"""环境与项目结构验证工具。"""

from pathlib import Path

from .curriculum_validation import (
    Capability,
    ContinuationDecision,
    CourseRequirements,
    CourseStatus,
    ProgressWindow,
    SnapshotCapabilities,
    SnapshotRecord,
    assert_snapshot_eligible,
    capability_eligible,
    coverage_matrix,
    decide_continuation,
    stratified_sample,
    validate_snapshot,
)


def validate_project_structure(root: Path | None = None) -> tuple[str, ...]:
    from .validate_project_structure import validate_project_structure as validate

    return validate(root)


__all__ = [
    "Capability",
    "ContinuationDecision",
    "CourseRequirements",
    "CourseStatus",
    "ProgressWindow",
    "SnapshotCapabilities",
    "SnapshotRecord",
    "assert_snapshot_eligible",
    "capability_eligible",
    "coverage_matrix",
    "decide_continuation",
    "stratified_sample",
    "validate_project_structure",
    "validate_snapshot",
]
