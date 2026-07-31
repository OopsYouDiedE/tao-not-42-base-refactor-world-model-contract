"""闭环动作计划的滚动续算契约。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil

from lumine.action_codec import LumineActionChunk

MIN_PLAN_TICKS = 8
MIN_REPLAN_LEAD_TICKS = 4


def replan_remaining_ticks(plan_ticks: int) -> int:
    """返回应启动下一轮推理时的剩余 tick 数。"""
    if plan_ticks < MIN_PLAN_TICKS:
        raise ValueError(f"动作计划至少需要 {MIN_PLAN_TICKS} tick")
    return max(MIN_REPLAN_LEAD_TICKS, ceil(plan_ticks / 4))


@dataclass(frozen=True)
class ScheduledAction:
    target_tick: int
    chunk: LumineActionChunk
    plan_id: str
    plan_index: int


@dataclass(frozen=True)
class PlanSubmission:
    plan_id: str
    start_tick: int
    plan_ticks: int
    accepted_ticks: int
    expired_ticks: int
    replan_tick: int


class RollingActionQueue:
    """按绝对 tick 保存动作，并给出异步续算触发状态。"""

    def __init__(self) -> None:
        self._actions: deque[ScheduledAction] = deque()
        self._replan_tick: int | None = None
        self._active_plan_id: str | None = None

    @property
    def depth(self) -> int:
        return len(self._actions)

    @property
    def active_plan_id(self) -> str | None:
        return self._active_plan_id

    def clear(self) -> None:
        self._actions.clear()
        self._replan_tick = None
        self._active_plan_id = None

    def submit(
        self,
        plan_id: str,
        chunks: tuple[LumineActionChunk, ...],
        *,
        start_tick: int,
        current_tick: int,
    ) -> PlanSubmission:
        if not plan_id:
            raise ValueError("plan_id 不能为空")
        plan_ticks = len(chunks)
        lead = replan_remaining_ticks(plan_ticks)
        expired = max(0, min(plan_ticks, current_tick - start_tick))
        accepted = [
            ScheduledAction(start_tick + index, chunk, plan_id, index)
            for index, chunk in enumerate(chunks)
            if start_tick + index >= current_tick
        ]
        if not accepted:
            raise ValueError("动作计划到达时已经全部过期")

        # 新计划只覆盖其接管时刻之后尚未执行的旧计划，旧计划前缀继续提供时延缓冲。
        self._actions = deque(action for action in self._actions if action.target_tick < start_tick)
        self._actions.extend(accepted)
        self._active_plan_id = plan_id
        self._replan_tick = start_tick + plan_ticks - lead
        return PlanSubmission(
            plan_id=plan_id,
            start_tick=start_tick,
            plan_ticks=plan_ticks,
            accepted_ticks=len(accepted),
            expired_ticks=expired,
            replan_tick=self._replan_tick,
        )

    def pop(self, current_tick: int) -> ScheduledAction | None:
        while self._actions and self._actions[0].target_tick < current_tick:
            self._actions.popleft()
        if self._actions and self._actions[0].target_tick == current_tick:
            return self._actions.popleft()
        return None

    def status(self, current_tick: int) -> dict[str, int | str | bool | None]:
        return {
            "active_plan_id": self._active_plan_id,
            "queued_ticks": len(self._actions),
            "replan_tick": self._replan_tick,
            "should_replan": self._replan_tick is None or current_tick >= self._replan_tick,
        }
