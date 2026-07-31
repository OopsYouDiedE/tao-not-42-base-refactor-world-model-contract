"""观察驱动课程的定义、准入与长期续跑合同。"""

from curriculum.bank import Capability, SnapshotRecord
from curriculum.runtime import (
    ContinuationDecision,
    CourseRequirements,
    CourseStatus,
    ProgressWindow,
    SnapshotCapabilities,
    assert_snapshot_eligible,
    decide_continuation,
)

__all__ = [
    "Capability",
    "ContinuationDecision",
    "CourseRequirements",
    "CourseStatus",
    "ProgressWindow",
    "SnapshotCapabilities",
    "SnapshotRecord",
    "assert_snapshot_eligible",
    "decide_continuation",
]
