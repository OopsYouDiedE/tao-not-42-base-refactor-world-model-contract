import pytest

from train.curriculum_runtime import (
    CourseRequirements,
    CourseStatus,
    ProgressWindow,
    SnapshotCapabilities,
    assert_snapshot_eligible,
    decide_continuation,
)


def test_correct_long_task_progress_extends_and_checkpoints() -> None:
    result = decide_continuation(
        ProgressWindow(False, False, False, 1200, 900, False, True, 3)
    )
    assert result.status is CourseStatus.PROGRESSING
    assert result.extend_budget and result.save_checkpoint


def test_failure_requires_evidence_not_just_no_success() -> None:
    unknown = decide_continuation(
        ProgressWindow(False, False, False, 5, 5, True, True, 32)
    )
    impossible = decide_continuation(
        ProgressWindow(False, True, False, 5, 5, True, True, 32)
    )
    assert unknown.status is CourseStatus.UNKNOWN
    assert impossible.status is CourseStatus.INFEASIBLE


def test_missing_materials_routes_to_preparation() -> None:
    result = decide_continuation(
        ProgressWindow(False, False, True, 0, 0, True, False, 0)
    )
    assert result.status is CourseStatus.PREPARATION_REQUIRED


@pytest.mark.parametrize(
    "requirement",
    [
        CourseRequirements(entities=True),
        CourseRequirements(scheduled_ticks=True),
        CourseRequirements(cross_dimension=True),
    ],
)
def test_unsupported_snapshot_courses_are_rejected(requirement: CourseRequirements) -> None:
    with pytest.raises(ValueError, match="不能公平复位"):
        assert_snapshot_eligible(requirement, SnapshotCapabilities())
