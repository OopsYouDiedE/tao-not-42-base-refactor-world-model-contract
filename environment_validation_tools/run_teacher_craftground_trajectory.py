"""在 WSL CraftGround 中执行一轮教师动作并导出转译器轨迹。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from online_environment_interaction_agents import (
    ClaudeCLIBackend,
    CLIConfig,
    CodexCLIBackend,
    TeacherRequest,
    TeacherTrajectoryExecutor,
)
from online_interactive_environments.craftground import create_environment


def _save_observation(observation: Any, path: Path) -> None:
    if not isinstance(observation, dict) or observation.get("rgb") is None:
        raise RuntimeError("CraftGround 观察缺少 rgb")
    Image.fromarray(observation["rgb"]).save(path)


def _summary(value: Any) -> Any:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    result: dict[str, Any] = {"keys": sorted(value)}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)):
            result[key] = item
    return result


def run(
    output_directory: Path,
    *,
    backend_name: str,
    model: str,
    executable: str,
    action_budget_ticks: int,
    port: int,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    if backend_name == "codex":
        backend = CodexCLIBackend(
            CLIConfig(
                model=model,
                executable=executable,
                timeout_seconds=240,
                command_arguments=("--ignore-rules",),
            )
        )
    else:
        backend = ClaudeCLIBackend(
            CLIConfig(model=model, executable=executable, timeout_seconds=240)
        )
    environment = create_environment(
        port=port,
        find_free_port=False,
        use_shared_memory=False,
        verbose=False,
    )
    metadata: dict[str, Any] = {
        "test_kind": "teacher_standard_input_action_closed_loop",
        "backend": backend.provider,
        "model": model,
        "environment_transport_backend": "socket",
        "environment_port": port,
        "wall_clock_started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        observation, reset_info = environment.reset(options={"fast_reset": False})
        start_image = output_directory / "start.png"
        _save_observation(observation, start_image)
        metadata["reset_info"] = _summary(reset_info)
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "online_environment_interaction_agents"
            / "TRAJECTORY_GENERATION_PROMPT.md"
        )
        request = TeacherRequest(
            prompt_path.read_text(encoding="utf-8"),
            "\n".join(
                (
                    "<trajectory_task>",
                    "trajectory_id: craftground-teacher-wsl-001",
                    "task: 根据当前 Minecraft 画面安全地产生一段短动作",
                    "success_criteria: 动作符合协议并由环境执行",
                    "failure_criteria: 协议错误、环境终止或超过预算",
                    "device: KeyboardMouse",
                    "action_protocol: standard-input-action/v1",
                    f"action_budget_ticks: {action_budget_ticks}",
                    "observation_policy: 本轮已由 Observe 触发；执行至少一个动作后，在需要新观察的动作 tick 写 Observe；Observe 必须与首个填充动作位于同一 tick，其后提供共 4 至 12 tick 的安全填充",
                    "available_inputs: W A S D Space Shift Ctrl MouseLeft MouseRight MouseMove NoOp Observe",
                    "control_mapping: Minecraft 默认键鼠映射",
                    "</trajectory_task>",
                )
            ),
            "\n".join(
                (
                    "<trajectory_step>",
                    "step_index: 0",
                    "environment_tick: 0",
                    f"remaining_action_ticks: {action_budget_ticks}",
                    "latest_observation_id: start",
                    "latest_state: {}",
                    "previous_action: null",
                    "previous_result: null",
                    "trajectory_summary: 尚未执行动作",
                    "termination_status: running",
                    "</trajectory_step>",
                    "请根据随本消息提供的最新观察图片输出下一段动作。",
                    "本次模型调用已经由 Observe 触发。只输出一个动作块，不要在首 tick 重复 Observe。中间 Observe 必须写成 Observe W、Observe MouseLeft 或 Observe NoOp，禁止裸写 Observe；其后必须保留合理填充动作。",
                )
            ),
            (start_image,),
        )
        executor = TeacherTrajectoryExecutor(environment, backend)
        result = executor.execute_generation(
            request,
            observation=observation,
            info=reset_info,
            remaining_action_ticks=action_budget_ticks,
        )
        _save_observation(result.observation, output_directory / "end.png")
        trajectory_json_path = executor.export(
            output_directory / "trajectory.json",
            trajectory_id="craftground-teacher-wsl-001",
        )
        trajectory_path = executor.export_markdown(
            output_directory / "trajectory.md",
            trajectory_id="craftground-teacher-wsl-001",
        )
        metadata.update(
            wall_clock_finished_at=datetime.now(timezone.utc).isoformat(),
            completed_ticks=result.completed_ticks,
            total_reward=result.total_reward,
            terminated=result.terminated,
            truncated=result.truncated,
            final_info=_summary(result.info),
            trajectory_path=str(trajectory_path),
            trajectory_json_path=str(trajectory_json_path),
        )
        return trajectory_path
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata_path = output_directory / "result.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("codex", "claude"), default="codex")
    parser.add_argument("--model", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--action-budget-ticks", type=int, default=4)
    parser.add_argument("--port", type=int, default=19900)
    arguments = parser.parse_args()
    print(
        run(
            arguments.output,
            backend_name=arguments.backend,
            model=arguments.model,
            executable=arguments.executable,
            action_budget_ticks=arguments.action_budget_ticks,
            port=arguments.port,
        )
    )


if __name__ == "__main__":
    main()
