from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from environment_validation_tools import run_four_teacher_trajectories
from online_environment_interaction_agents import TeacherRequest, TeacherResponse


class FakeBackend:
    provider = "fake-provider"
    model = "same-model"

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        return TeacherResponse(
            "Device KeyboardMouse\nTick 0\n<action>NoOp</action>",
            self.provider,
            self.model,
            "request",
            1,
            1,
            1.0,
        )


class FakeEnvironment:
    def __init__(self, *, x_offset: float = 0.0) -> None:
        self.pending_commands: list[str] = []
        self.position = [6.0 + x_offset, 113.0, 2.0]
        self.snapshot_position: list[float] | None = None
        self.closed = False

    def reset(self, options: dict[str, Any]):
        return self._observation(), self._info()

    def add_command(self, command: str) -> None:
        self.pending_commands.append(command)

    def step(self, action: Any):
        for command in self.pending_commands:
            if command.startswith("memorysnapshot save"):
                self.snapshot_position = list(self.position)
            elif command.startswith("memorysnapshot load"):
                assert self.snapshot_position is not None
                self.position = list(self.snapshot_position)
            elif command.startswith("tp @p ~"):
                offset = float(command.split("~", 1)[1].split()[0])
                self.position[0] += offset
            elif command.startswith("tp @p "):
                coordinates = command.split()[2:5]
                self.position = [float(value) for value in coordinates]
        self.pending_commands.clear()
        return self._observation(), 0.0, False, False, self._info()

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> dict[str, Any]:
        return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

    def _info(self) -> dict[str, Any]:
        block = SimpleNamespace(translation_key="block.minecraft.grass_block")
        full = SimpleNamespace(
            x=self.position[0],
            y=self.position[1],
            z=self.position[2],
            yaw=0.0,
            pitch=0.0,
            health=20.0,
            inventory=[],
            raycast_result=SimpleNamespace(target_block=block),
        )
        return {"full": full}


def test_four_arm_entry_uses_same_backend_and_emits_consumable_results(
    tmp_path: Path,
) -> None:
    environments: list[FakeEnvironment] = []
    factory_arguments: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> FakeEnvironment:
        environment = FakeEnvironment(x_offset=float(len(environments)))
        environments.append(environment)
        factory_arguments.append(kwargs)
        return environment

    output = run_four_teacher_trajectories.run(
        tmp_path / "run",
        action_budget_ticks=1,
        max_generations=1,
        warmup_ticks=0,
        backend_name="fake",
        backend=FakeBackend(),
        environment_factory=factory,
        enforce_wsl=False,
        use_shared_memory=False,
    )
    result = json.loads((output.parent / "result.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line)
        for line in (output.parent / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert output.is_file()
    assert len({item["backend_name"] for item in result["trajectories"]}) == 1
    assert len({item["model"] for item in result["trajectories"]}) == 1
    assert all(item["executed_ticks"] == 1 for item in result["trajectories"])
    assert all(item["wall_clock_duration_seconds"] >= 0 for item in result["trajectories"])
    assert all(
        item["action_adapter"] == "CraftGroundKeyboardMouseAdapter"
        for item in result["trajectories"]
    )
    assert all(len(item["executed_actions"]) == 1 for item in result["trajectories"])
    assert all(item["artifact_paths"]["trajectory_json"] for item in result["trajectories"])
    assert all(item["contract_valid"] for item in result["trajectory_reviews"])
    assert abs(sum(item["relative_advantage"] for item in result["comparison_samples"])) < 1e-6
    assert result["comparison_review"]["valid"] is True
    assert result["shared_start"]["restore_probe_passed"] is True
    assert result["shared_start"]["environment_transport_backend"] == "socket"
    assert all(item["environment_transport_backend"] == "socket" for item in result["trajectories"])
    assert len({item["instance_id"] for item in factory_arguments}) == 4
    assert all(item["baseline_world_path"] is None for item in factory_arguments)
    assert all(environment.closed for environment in environments)
    assert progress[0]["event"] == "run_started"
    assert progress[-1]["event"] == "run_completed"
    assert sum(item["event"] == "runtime_creation_completed" for item in progress) == 4
    assert sum(item["event"] == "environment_reset_completed" for item in progress) == 4
    assert sum(item["event"] == "trajectory_started" for item in progress) == 4
    assert sum(item["event"] == "generation_completed" for item in progress) == 4
    assert sum(item["event"] == "action_executed" for item in progress) == 4
    assert sum(item["event"] == "trajectory_completed" for item in progress) == 4
    assert sum(item["event"] == "comparison_completed" for item in progress) == 1
    assert all("timestamp" in item for item in progress)


def test_backend_configuration_uses_explicit_teacher_environment() -> None:
    backend = run_four_teacher_trajectories._load_backend(
        "openai-api",
        {
            "TEACHER_API_URL": "https://example.test/v1",
            "TEACHER_API_KEY": "secret",
            "TEACHER_MODEL": "model",
        },
    )

    assert backend.config.base_url == "https://example.test/v1"
    assert backend.config.api_key == "secret"
    assert backend.model == "model"


def test_log_success_uses_total_target_count() -> None:
    inventory = [
        SimpleNamespace(translation_key="item.minecraft.oak_log", count=3),
        SimpleNamespace(translation_key="item.minecraft.birch_log", count=1),
    ]
    full = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=0.0,
        yaw=0.0,
        pitch=0.0,
        health=20.0,
        inventory=inventory,
        raycast_result=SimpleNamespace(target_block=None),
    )
    info = {"full": full}

    assert run_four_teacher_trajectories._log_count(info) == 4
    assert run_four_teacher_trajectories._has_log(info, 4) is True
    assert run_four_teacher_trajectories._has_log(info, 5) is False


def test_single_arm_skips_group_relative_advantage(tmp_path: Path) -> None:
    output = run_four_teacher_trajectories.run(
        tmp_path / "single",
        action_budget_ticks=1,
        max_generations=1,
        warmup_ticks=0,
        backend_name="fake",
        backend=FakeBackend(),
        environment_factory=lambda **kwargs: FakeEnvironment(),
        enforce_wsl=False,
        use_shared_memory=False,
        trajectory_count=1,
    )
    result = json.loads((output.parent / "result.json").read_text(encoding="utf-8"))

    assert result["trajectory_count"] == 1
    assert len(result["trajectories"]) == 1
    assert result["comparison_samples"][0]["relative_advantage"] == 0.0
    assert result["comparison_samples"][0]["rank"] == 1


def test_request_returns_one_history_frame_and_executed_action(tmp_path: Path) -> None:
    previous_image = tmp_path / "previous.png"
    current_image = tmp_path / "current.png"
    previous_image.touch()
    current_image.touch()
    request = run_four_teacher_trajectories._request(
        "system",
        trajectory_id="T01",
        round_index=1,
        environment_tick=10,
        remaining_ticks=20,
        observation_path=current_image,
        previous_observation_path=previous_image,
        latest_state={
            "position": [1, 2, 3],
            "yaw": 4,
            "pitch": 5,
            "health": 20,
            "inventory": [],
            "raycast_block": None,
        },
        previous_state={
            "position": [0, 2, 3],
            "yaw": 4,
            "pitch": 5,
            "health": 20,
            "inventory": [],
            "raycast_block": None,
        },
        previous_action="Device KeyboardMouse\nTick 0\n<action>W</action>",
        previous_result={"completed_ticks": 1, "reward": 0.0},
    )

    assert request.observation_paths == (previous_image, current_image)
    assert "previous_action: Device KeyboardMouse" in request.step_context
    assert "model_memory_from_previous_round" not in request.step_context
    assert "<assessment>" not in request.step_context
    assert "准星命中方块: 环境未返回" in request.step_context
    assert "目标太远" not in request.step_context


def test_executed_action_history_uses_only_committed_ticks() -> None:
    history = run_four_teacher_trajectories._format_executed_action_history(
        [
            {"inputs": ["W"]},
            {"inputs": ["W"]},
            {"inputs": ["MouseLeft"]},
        ]
    )

    assert history == "W x2 ; MouseLeft"


def test_default_rollout_contract_is_four_trajectories_and_ten_rounds() -> None:
    parameters = inspect.signature(run_four_teacher_trajectories.run).parameters

    assert parameters["trajectory_count"].default == 4
    assert parameters["max_generations"].default == 10
