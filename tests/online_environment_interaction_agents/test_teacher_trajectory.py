from __future__ import annotations

import json
from pathlib import Path

import pytest

from online_environment_interaction_agents import (
    OpenAICompatibleConfig,
    TeacherModelError,
    TeacherRequest,
    TeacherResponse,
    TeacherTrajectoryGenerator,
    parse_teacher_decision,
)


class FakeBackend:
    provider = "fake"
    model = "fake-model"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        return TeacherResponse(
            text=self.text,
            provider=self.provider,
            model=self.model,
            request_id="request-1",
            input_tokens=10,
            output_tokens=5,
            elapsed_ms=12.5,
        )


def _request() -> TeacherRequest:
    return TeacherRequest("system", "task", "step")


def test_generator_accepts_strict_action_and_records_audit(tmp_path: Path) -> None:
    record_path = tmp_path / "records" / "teacher.jsonl"
    generator = TeacherTrajectoryGenerator(
        FakeBackend("\nDevice KeyboardMouse\nTick 0\n<action>W x2 ; NoOp</action>\n"),
        record_path=record_path,
    )

    result = generator.generate_step(
        _request(),
        trajectory_id="trajectory-1",
        step_index=0,
        remaining_action_ticks=3,
        expected_device="KeyboardMouse",
    )

    assert len(result.action.ticks) == 3
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "accepted"
    assert record["tick_count"] == 3
    assert record["request_id"] == "request-1"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (
            "说明\nDevice KeyboardMouse\nTick 0\n<action>W</action>",
            "不符合动作协议",
        ),
        ("Device Gamepad\nTick 0\n<action>A</action>", "与预期"),
        ("Device KeyboardMouse\nTick 0\n<action>W x4</action>", "超过剩余预算"),
    ],
)
def test_generator_rejects_invalid_teacher_output(tmp_path: Path, text: str, error: str) -> None:
    record_path = tmp_path / "teacher.jsonl"
    generator = TeacherTrajectoryGenerator(FakeBackend(text), record_path=record_path)

    with pytest.raises(TeacherModelError, match=error):
        generator.generate_step(
            _request(),
            trajectory_id="trajectory-1",
            step_index=1,
            remaining_action_ticks=2,
            expected_device="KeyboardMouse",
        )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "rejected"
    assert record["raw_output"] == text


def test_openai_config_loads_dedicated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEACHER_API_URL", "https://example.test/v1")
    monkeypatch.setenv("TEACHER_API_KEY", "secret")
    monkeypatch.setenv("TEACHER_MODEL", "teacher-model")

    config = OpenAICompatibleConfig.from_environment()

    assert config.model == "teacher-model"
    assert config.base_url == "https://example.test/v1"


def test_decision_envelope_preserves_pure_control() -> None:
    decision = parse_teacher_decision(
        "\nDevice KeyboardMouse\nTick 0\n<action>MouseMove 30 0 ; NoOp</action>\n"
    )

    assert decision.control.startswith("Device KeyboardMouse")


def test_generator_accepts_pure_control(tmp_path: Path) -> None:
    text = "Device KeyboardMouse\nTick 0\n<action>NoOp</action>"
    generator = TeacherTrajectoryGenerator(FakeBackend(text), record_path=tmp_path / "audit.jsonl")

    result = generator.generate_step(
        _request(),
        trajectory_id="t",
        step_index=0,
        remaining_action_ticks=1,
        expected_device="KeyboardMouse",
    )

    assert result.raw_output == text
