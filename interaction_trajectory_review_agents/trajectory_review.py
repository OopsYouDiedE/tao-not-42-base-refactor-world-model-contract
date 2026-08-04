"""对已执行轨迹做确定性的合同与质量审核。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectoryReview:
    trajectory_id: str
    contract_valid: bool
    task_success: bool
    quality_score: float
    executed_ticks: int
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["issues"] = list(self.issues)
        return value


def review_trajectory(
    trajectory: dict[str, Any],
    summary: dict[str, Any],
    *,
    action_budget_ticks: int,
) -> TrajectoryReview:
    """依据环境事实审核协议、预算、生成状态与任务结果。"""
    trajectory_id = str(summary.get("trajectory_id") or trajectory.get("trajectory_id"))
    execution_ticks = trajectory.get("execution_ticks")
    generations = trajectory.get("generation_records")
    issues: list[str] = []
    if trajectory.get("action_protocol") != "standard-input-action/v1":
        issues.append("action_protocol_mismatch")
    if trajectory.get("action_protocol_version") != "v1":
        issues.append("action_protocol_version_mismatch")
    if trajectory.get("action_backend") != "keyboard_and_mouse_only":
        issues.append("action_backend_mismatch")
    if trajectory.get("action_adapter") != "CraftGroundKeyboardMouseAdapter":
        issues.append("action_adapter_mismatch")
    if not isinstance(execution_ticks, list):
        issues.append("execution_ticks_missing")
        executed_ticks = 0
    else:
        executed_ticks = len(execution_ticks)
    if executed_ticks != int(summary.get("executed_ticks", -1)):
        issues.append("executed_tick_count_mismatch")
    if len(summary.get("executed_actions", ())) != executed_ticks:
        issues.append("executed_action_count_mismatch")
    if float(summary.get("wall_clock_duration_seconds", -1)) < 0:
        issues.append("wall_clock_duration_missing")
    if executed_ticks > action_budget_ticks:
        issues.append("action_budget_exceeded")
    if not isinstance(generations, list):
        issues.append("generation_records_missing")
    elif any(item.get("status") not in {"completed", "failed"} for item in generations):
        issues.append("unfinished_generation")
    if summary.get("trajectory_error"):
        issues.append("trajectory_error")

    task_success = bool(summary.get("trajectory_success"))
    efficiency = max(0.0, 1.0 - executed_ticks / action_budget_ticks)
    quality_score = (1.0 if task_success else 0.0) + 0.25 * efficiency
    if issues:
        quality_score -= min(0.5, 0.1 * len(issues))
    return TrajectoryReview(
        trajectory_id=trajectory_id,
        contract_valid=not issues,
        task_success=task_success,
        quality_score=round(quality_score, 6),
        executed_ticks=executed_ticks,
        issues=tuple(issues),
    )
