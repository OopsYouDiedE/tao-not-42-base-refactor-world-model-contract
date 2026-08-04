from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from online_environment_interaction_agents import (
    TeacherRequest,
    TeacherResponse,
    TeacherTrajectoryExecutor,
)
from online_interactive_environments.craftground import CraftGroundKeyboardMouseAdapter


class FakeBackend:
    provider = "fake-teacher"
    model = "fake-model"

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        return TeacherResponse(
            "Device KeyboardMouse\nTick 0\n<action>W MouseMove 2 -1 ; MouseLeft ; NoOp</action>",
            self.provider,
            self.model,
            "request-1",
            20,
            10,
            25.0,
        )


class FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[dict[str, bool | float]] = []

    def step(self, action: dict[str, bool | float]):
        self.actions.append(action)
        tick = len(self.actions)
        return {"tick": tick}, float(tick), False, False, {"tick": tick}


class DelayedStreamBackend:
    provider = "delayed-teacher"
    model = "delayed-model"

    def stream(self, request: TeacherRequest, on_chunk):
        time.sleep(0.13)
        text = "Device KeyboardMouse\nTick 0\n<action>W</action>"
        on_chunk(text)
        return TeacherResponse(
            text,
            self.provider,
            self.model,
            "delayed-request",
            10,
            2,
            130.0,
        )


def _action_factory() -> dict[str, bool | float]:
    fields = {
        "forward",
        "back",
        "left",
        "right",
        "jump",
        "sneak",
        "sprint",
        "attack",
        "use",
        "drop",
        "inventory",
        "camera_yaw",
        "camera_pitch",
        *(f"hotbar.{slot}" for slot in range(1, 10)),
    }
    return {field: 0.0 if field.startswith("camera_") else False for field in fields}


def test_executor_uses_recording_compiler_and_exports_trajectory(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    executor = TeacherTrajectoryExecutor(
        environment,
        FakeBackend(),
        adapter=CraftGroundKeyboardMouseAdapter(action_factory=_action_factory),
    )

    result = executor.execute_generation(
        TeacherRequest("system", "task", "step"),
        observation={"tick": 0},
        remaining_action_ticks=3,
    )
    output = executor.export(tmp_path / "trajectory.json", trajectory_id="test-1")
    markdown_output = executor.export_markdown(tmp_path / "trajectory.md", trajectory_id="test-1")
    trajectory = json.loads(output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")

    assert result.completed_ticks == 3
    assert result.total_reward == 6.0
    assert result.observation_requested is True
    assert executor.compiler.record_generations is True
    assert environment.actions[0]["forward"] is True
    assert environment.actions[0]["camera_yaw"] == 0.3
    assert environment.actions[1]["attack"] is True
    assert trajectory["compiler_record_generations"] is True
    assert trajectory["generation_records"][0]["status"] == "completed"
    assert trajectory["generation_records"][0]["accepted_ticks"] == 4
    assert trajectory["generation_records"][0]["telemetry"]["request_id"] == "request-1"
    assert len(trajectory["execution_ticks"]) == 3
    assert "## Observe 0 → 模型输出 0" in markdown
    assert "### 教师模型原始输出" in markdown
    assert "<action>W MouseMove 2 -1 ; MouseLeft ; NoOp</action>" in markdown
    assert "| 0 | 否 | 否 | `W MouseMove 2 -1` | 1.0 | 否 |" in markdown


def test_executor_records_model_latency_without_converting_it_to_ticks(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    executor = TeacherTrajectoryExecutor(
        environment,
        DelayedStreamBackend(),
        adapter=CraftGroundKeyboardMouseAdapter(action_factory=_action_factory),
    )

    result = executor.execute_generation(
        TeacherRequest("system", "task", "step"),
        observation={"tick": 0},
        remaining_action_ticks=1,
    )
    output = executor.export(tmp_path / "trajectory.json", trajectory_id="delayed")
    trajectory = json.loads(output.read_text(encoding="utf-8"))

    assert result.completed_ticks == 1
    assert len(environment.actions) == 1
    assert environment.actions[0]["forward"] is True
    waiting = trajectory["waiting_summary"]
    assert waiting["total_waiting_ticks"] == 0
    assert waiting["max_waiting_ticks"] == 0
    assert waiting["max_waiting_ms"] >= 100
    assert waiting["total_uncovered_wait_ms"] >= 100
    assert waiting["total_fill_executed_ticks"] == 0
    assert waiting["total_overwritten_future_ticks"] == 0
    assert waiting["measurement"] == "wall_clock_inference_gap_and_logical_underflow"
    assert waiting["max_generation_id"] == "generation-0"
    assert trajectory["generation_records"][0]["telemetry"]["total_generation_ms"] == 130.0


class FixedResponseBackend:
    provider = "fixed"
    model = "fixed-model"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        return TeacherResponse(
            self.text,
            self.provider,
            self.model,
            "fixed-request",
            1,
            1,
            1.0,
        )


class SequencedDelayedBackend:
    provider = "sequenced"
    model = "sequenced-model"

    def __init__(self) -> None:
        self.responses = [
            (0.0, "Device KeyboardMouse\nTick 0\n<action>W ; Observe W ; W x2</action>"),
            (0.02, "Device KeyboardMouse\nTick 0\n<action>A ; Observe A</action>"),
        ]

    def stream(self, request, on_chunk):
        delay, text = self.responses.pop(0)
        time.sleep(delay)
        on_chunk(text)
        return TeacherResponse(text, self.provider, self.model, None, 1, 1, delay * 1000)


class SlowEnvironment(FakeEnvironment):
    def step(self, action):
        time.sleep(0.05)
        return super().step(action)


def test_executor_runs_fill_during_inference_and_overwrites_future(tmp_path: Path) -> None:
    environment = SlowEnvironment()
    executor = TeacherTrajectoryExecutor(
        environment,
        SequencedDelayedBackend(),
        adapter=CraftGroundKeyboardMouseAdapter(action_factory=_action_factory),
    )
    request = TeacherRequest("system", "task", "step")

    first = executor.execute_generation(request, observation={"tick": 0}, remaining_action_ticks=8)
    second = executor.execute_generation(
        request, observation=first.observation, remaining_action_ticks=7
    )
    output = executor.export(tmp_path / "trajectory.json", trajectory_id="async-fill")
    trajectory = json.loads(output.read_text(encoding="utf-8"))

    assert first.completed_ticks == 1
    assert second.completed_ticks >= 2
    assert trajectory["execution_ticks"][1]["latency_fill"] is True
    assert trajectory["execution_ticks"][1]["generation_id"] == "generation-0"
    assert trajectory["execution_ticks"][1]["inference_generation_id"] == "generation-1"
    assert trajectory["generation_latency"]["generation-1"]["fill_executed_ticks"] >= 1
    assert trajectory["generation_latency"]["generation-1"]["overwritten_future_ticks"] >= 1


@pytest.mark.parametrize(
    "text, error",
    (
        (
            "说明\nDevice KeyboardMouse\nTick 0\n<action>W</action>",
            "不符合动作协议",
        ),
        (
            "Device KeyboardMouse\nTick 0\n<action>W x2</action>",
            "超过剩余预算",
        ),
        (
            "Device Gamepad\nTick 0\n<action>A</action>",
            "不支持设备",
        ),
        (
            "Device KeyboardMouse\nTick 0\n<action>W ; 1 2</action>",
            "只能选择一个快捷栏",
        ),
    ),
)
def test_executor_rejects_invalid_generation_before_environment_steps(
    text: str,
    error: str,
) -> None:
    environment = FakeEnvironment()
    executor = TeacherTrajectoryExecutor(
        environment,
        FixedResponseBackend(text),
        adapter=CraftGroundKeyboardMouseAdapter(action_factory=_action_factory),
    )

    with pytest.raises(RuntimeError, match=error):
        executor.execute_generation(
            TeacherRequest("system", "task", "step"),
            observation={"tick": 0},
            remaining_action_ticks=1 if "x2" in text else 4,
        )

    assert environment.actions == []
    assert executor.compiler.current_tick == 0
    assert executor.compiler.generation_records[0].status.value == "failed"
    assert executor.model_decisions["generation-0"]["raw_response"] == text
