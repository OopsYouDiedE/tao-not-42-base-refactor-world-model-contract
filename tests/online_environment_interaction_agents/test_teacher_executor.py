"""教师执行器测试；动作在真实 CraftGround 上执行，不使用环境替身。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from online_environment_interaction_agents import (
    ScriptedBackend,
    TeacherRequest,
    TeacherTrajectoryExecutor,
)
from online_interactive_environments.craftground import EnvironmentHandle, EnvironmentKernel

pytestmark = pytest.mark.craftground

FIRST = "Device KeyboardMouse\nTick 0\n<action>W MouseMove 2 -1 ; MouseLeft ; NoOp</action>"


@pytest.fixture(scope="module")
def kernel() -> Iterator[EnvironmentKernel]:
    with EnvironmentKernel.launch(
        slots=1,
        port_base=18780,
        image_width=160,
        image_height=90,
        use_shared_memory=False,
    ) as launched:
        yield launched


@pytest.fixture
def handle(kernel: EnvironmentKernel) -> Iterator[EnvironmentHandle]:
    """每个用例复用同一个 JVM，但先回到根快照，使设备与世界状态一致。"""
    slot = kernel.handles()[0]
    slot.reset_world()
    yield slot


def test_executor_exports_both_ledgers_joined_by_generation(
    handle: EnvironmentHandle,
    tmp_path: Path,
) -> None:
    executor = TeacherTrajectoryExecutor(handle, ScriptedBackend(FIRST))

    result = executor.execute_generation(
        TeacherRequest("system", "task", "step"),
        observation=handle.observe(),
        remaining_action_ticks=3,
    )
    output = executor.export(tmp_path / "trajectory.json", trajectory_id="scripted-1")
    markdown_output = executor.export_markdown(
        tmp_path / "trajectory.md", trajectory_id="scripted-1"
    )
    trajectory = json.loads(output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")

    assert result.completed_ticks == 3
    assert result.observation_requested is True
    assert executor.compiler.record_generations is True
    assert trajectory["action_protocol"] == "standard-input-action/v1"
    assert trajectory["action_adapter"] == "CraftGroundKeyboardMouseAdapter"
    assert trajectory["generation_records"][0]["status"] == "completed"
    assert trajectory["generation_records"][0]["accepted_ticks"] == 4
    ticks = trajectory["execution_ticks"]
    assert len(ticks) == 3
    assert ticks[0]["native_action"]["forward"] is True
    assert ticks[0]["native_action"]["camera_yaw"] == pytest.approx(0.3)
    assert ticks[1]["native_action"]["attack"] is True
    assert all(tick["generation_id"] == "generation-0" for tick in ticks)
    assert "## Observe 0 → 模型输出 0" in markdown
    assert "### 教师模型原始输出" in markdown
    assert "<action>W MouseMove 2 -1 ; MouseLeft ; NoOp</action>" in markdown


def test_executor_advances_across_consecutive_generations(handle: EnvironmentHandle) -> None:
    executor = TeacherTrajectoryExecutor(
        handle,
        ScriptedBackend(
            "Device KeyboardMouse\nTick 0\n<action>W ; Observe ; W x2</action>",
            "Device KeyboardMouse\nTick 0\n<action>A ; Observe ; A</action>",
        ),
    )
    request = TeacherRequest("system", "task", "step")

    first = executor.execute_generation(
        request, observation=handle.observe(), remaining_action_ticks=8
    )
    second = executor.execute_generation(
        request, observation=first.observation, remaining_action_ticks=7
    )

    assert first.completed_ticks == 1
    assert first.observation_requested is True
    assert second.completed_ticks >= 1
    assert executor.compiler.current_tick == first.completed_ticks + second.completed_ticks
    assert [record.status.value for record in executor.compiler.generation_records] == [
        "completed",
        "completed",
    ]


@pytest.mark.parametrize(
    "text, error",
    (
        ("说明\nDevice KeyboardMouse\nTick 0\n<action>W</action>", "不符合动作协议"),
        ("Device KeyboardMouse\nTick 0\n<action>W x2</action>", "超过剩余预算"),
        ("Device Gamepad\nTick 0\n<action>A</action>", "不支持设备"),
        ("Device KeyboardMouse\nTick 0\n<action>W ; 1 2</action>", "只能选择一个快捷栏"),
    ),
)
def test_executor_rejects_invalid_generation_before_touching_the_environment(
    handle: EnvironmentHandle,
    text: str,
    error: str,
) -> None:
    executor = TeacherTrajectoryExecutor(handle, ScriptedBackend(text))

    with pytest.raises(RuntimeError, match=error):
        executor.execute_generation(
            TeacherRequest("system", "task", "step"),
            observation=handle.observe(),
            remaining_action_ticks=1 if "x2" in text else 4,
        )

    assert executor.compiler.current_tick == 0
    assert executor.execution_ticks == []
    assert executor.compiler.generation_records[0].status.value == "failed"
    assert executor.model_decisions["generation-0"]["raw_response"] == text
