"""编译器驱动的环境循环。

标准输入动作文本直接交给 `ActionSequenceCompiler`：编译、下溢策略和续跑预算都由它
持有。这里只做一件事——反复向编译器拉取当前 tick，把动作送进槽位，再确认完成。环境
不决定执行什么，只负责执行并回报事实。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from online_interactive_environments import (
    ActionSequenceCompiler,
    DecisionKind,
    Submission,
    UnderflowPolicy,
)

from .kernel import EnvironmentHandle, StepOutcome

DEFAULT_ACTION_SEQUENCE = (
    "Device KeyboardMouse\n"
    "Tick 60\n"
    "<action>W Space MouseLeft x60 ; Observe ; W Space MouseLeft x40</action>"
)


@dataclass(frozen=True)
class SessionTick:
    """一个已执行 tick 的完整事实：环境侧结果加上它来自队列还是下溢。"""

    tick: int
    source: str
    observe: bool
    outcome: StepOutcome
    #: 该 tick 是否消耗了续跑预算。队列耗尽后的下溢为真；队列内空隙的下溢为假。
    overrun: bool = False

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.outcome.inputs


@dataclass
class SessionStats:
    """一个槽位自上次 `reset` 以来的累计事实。"""

    submitted_sequences: int = 0
    submitted_ticks: int = 0
    executed_ticks: int = 0
    overrun_ticks: int = 0
    observe_ticks: int = 0
    expired_ticks: int = 0
    overwritten_ticks: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    step_elapsed_ms_total: float = 0.0
    source_counts: dict[str, int] = field(default_factory=dict)

    @property
    def mean_step_elapsed_ms(self) -> float:
        if not self.executed_ticks:
            return 0.0
        return self.step_elapsed_ms_total / self.executed_ticks

    def as_dict(self) -> dict[str, object]:
        return {
            "submitted_sequences": self.submitted_sequences,
            "submitted_ticks": self.submitted_ticks,
            "executed_ticks": self.executed_ticks,
            "overrun_ticks": self.overrun_ticks,
            "observe_ticks": self.observe_ticks,
            "expired_ticks": self.expired_ticks,
            "overwritten_ticks": self.overwritten_ticks,
            "total_reward": round(self.total_reward, 4),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "mean_step_elapsed_ms": round(self.mean_step_elapsed_ms, 3),
            "source_counts": dict(self.source_counts),
        }


class ManualActionSession:
    """把一个槽位接到一个编译器上，按编译器的决策推进环境。"""

    def __init__(
        self,
        handle: EnvironmentHandle,
        *,
        underflow: UnderflowPolicy = UnderflowPolicy.WAIT,
        max_overrun_ticks: int | None = 0,
    ) -> None:
        self.handle = handle
        self.stats = SessionStats()
        self.compiler = ActionSequenceCompiler(
            underflow,
            auto_observe=False,
            max_overrun_ticks=max_overrun_ticks,
        )

    @property
    def underflow(self) -> UnderflowPolicy:
        return self.compiler.underflow

    @underflow.setter
    def underflow(self, policy: UnderflowPolicy) -> None:
        self.compiler.underflow = policy

    @property
    def max_overrun_ticks(self) -> int | None:
        return self.compiler.max_overrun_ticks

    @max_overrun_ticks.setter
    def max_overrun_ticks(self, value: int | None) -> None:
        self.compiler.max_overrun_ticks = value

    @property
    def buffered_ticks(self) -> int:
        return self.compiler.buffered_ticks

    @property
    def current_tick(self) -> int:
        return self.compiler.current_tick

    def submit(self, text: str) -> Submission:
        """把标准输入动作文本交给编译器。

        提交前在克隆的设备状态上预演整段转译，使非法输入在触碰环境之前被拒绝；
        编译与 tick 展开由编译器负责。
        """
        preview = self.handle.preview_adapter()
        submission = self.compiler.submit(text)
        for tick in range(submission.start_tick, submission.start_tick + submission.accepted_ticks):
            preview.convert(self.compiler.scheduled_action(tick))
        self.stats.submitted_sequences += 1
        self.stats.submitted_ticks += submission.accepted_ticks
        self.stats.expired_ticks += submission.expired_ticks
        self.stats.overwritten_ticks += submission.overwritten_ticks
        return submission

    def run(self, *, limit: int | None = None) -> Iterator[SessionTick]:
        """反复拉取并执行，直到编译器返回 WAIT 或环境结束。"""
        produced = 0
        while limit is None or produced < limit:
            if self.stats.terminated or self.stats.truncated:
                return
            decision = self.compiler.pull()
            if decision.kind is DecisionKind.WAIT:
                return
            if decision.kind is DecisionKind.OBSERVE:
                # CraftGround 没有独立观察通道，最近一帧即为该 tick 的观察。
                self.compiler.observed()
                self.stats.observe_ticks += 1
                continue
            if decision.action is None:
                raise RuntimeError("转译器返回了没有动作的 ACTION 决策")
            outcome = self.handle.apply(decision.action)
            executed_tick = decision.tick
            source = decision.source or "unknown"
            # 队列为空时的下溢才算续跑；队列里还有排队 tick 时这只是跨越空隙。
            # `commit` 会消耗队列，因此在它之前取这个事实。
            overrun = source != "sequence" and not self.compiler.buffered_ticks
            self.compiler.commit(decision)
            self._record(outcome, source, overrun=overrun)
            produced += 1
            yield SessionTick(
                tick=executed_tick,
                source=source,
                observe=decision.action.observe,
                outcome=outcome,
                overrun=overrun,
            )

    def reset(self, *, world: bool = False) -> None:
        """清空编译器与统计；`world` 为真时重开世界，否则倒档到根快照。"""
        underflow = self.compiler.underflow
        budget = self.compiler.max_overrun_ticks
        self.compiler.reset()
        self.compiler.underflow = underflow
        self.compiler.max_overrun_ticks = budget
        self.stats = SessionStats()
        if world:
            self.handle.reset_world()
        else:
            self.handle.reset_to()

    def _record(self, outcome: StepOutcome, source: str, *, overrun: bool) -> None:
        self.stats.executed_ticks += 1
        self.stats.total_reward += outcome.reward
        self.stats.step_elapsed_ms_total += outcome.step_elapsed_ms
        self.stats.terminated = outcome.terminated
        self.stats.truncated = outcome.truncated
        self.stats.source_counts[source] = self.stats.source_counts.get(source, 0) + 1
        if overrun:
            self.stats.overrun_ticks += 1
