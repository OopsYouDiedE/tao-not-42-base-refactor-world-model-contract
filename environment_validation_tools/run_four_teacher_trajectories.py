"""从同一逻辑 CraftGround 快照运行四条同策略教师轨迹。"""

from __future__ import annotations

import json
import os
import platform
import shlex
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from interaction_trajectory_review_agents import review_trajectory
from model_judgment_review_agents import review_comparison
from online_environment_interaction_agents import (
    AnthropicCompatibleBackend,
    AnthropicCompatibleConfig,
    ClaudeCLIBackend,
    CLIConfig,
    CodexCLIBackend,
    OpenAICompatibleBackend,
    OpenAICompatibleConfig,
    TeacherModelError,
    TeacherRequest,
    TeacherTrajectoryExecutor,
)
from online_interactive_environments.craftground import (
    CraftGroundKeyboardMouseAdapter,
    MemorySnapshotCoordinator,
    ParallelRolloutRunner,
    RolloutRequest,
    SnapshotRegion,
    create_environment,
)
from relative_advantage_comparison_training import (
    ComparisonSample,
    build_comparison_group,
)

TRAJECTORY_COUNT = 4
SNAPSHOT_ID = "teacher-log-shared-start"
NORMALIZATION_COMMANDS = (
    "gamerule doDaylightCycle false",
    "gamerule doWeatherCycle false",
    "gamerule doMobSpawning false",
    "gamerule randomTickSpeed 0",
    "time set day",
    "weather clear",
    "kill @e[type=!minecraft:player]",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ProgressReporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **details: Any) -> None:
        payload = {"timestamp": _utc_now(), "event": event, **details}
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            print(f"[progress] {line}", flush=True)


def _save_rgb(observation: Any, path: Path) -> None:
    if not isinstance(observation, dict) or observation.get("rgb") is None:
        raise RuntimeError("CraftGround 观察缺少 rgb")
    Image.fromarray(observation["rgb"]).save(path)


def _required_setting(settings: Mapping[str, str], name: str) -> str:
    value = settings.get(name)
    if not value:
        raise RuntimeError(f"缺少环境变量：{name}")
    return value


def _load_backend(
    backend_name: str,
    settings: Mapping[str, str] | None = None,
) -> Any:
    """只从显式环境配置创建后端，不读取 CLI 私有凭据文件。"""
    values = os.environ if settings is None else settings
    model = _required_setting(values, "TEACHER_MODEL")
    timeout = float(values.get("TEACHER_TIMEOUT_SECONDS", "240"))
    if backend_name == "openai-api":
        return OpenAICompatibleBackend(
            OpenAICompatibleConfig(
                base_url=_required_setting(values, "TEACHER_API_URL"),
                api_key=_required_setting(values, "TEACHER_API_KEY"),
                model=model,
                timeout_seconds=timeout,
                wire_api=values.get("TEACHER_WIRE_API", "chat_completions"),
            )
        )
    if backend_name == "anthropic-api":
        return AnthropicCompatibleBackend(
            AnthropicCompatibleConfig(
                base_url=_required_setting(values, "TEACHER_API_URL"),
                auth_token=_required_setting(values, "TEACHER_API_KEY"),
                model=model,
                timeout_seconds=timeout,
            )
        )
    executable = _required_setting(values, "TEACHER_CLI_EXECUTABLE")
    arguments = tuple(shlex.split(values.get("TEACHER_CLI_ARGUMENTS", "")))
    config = CLIConfig(
        model=model,
        executable=executable,
        timeout_seconds=timeout,
        command_arguments=arguments,
    )
    if backend_name == "codex-cli":
        return CodexCLIBackend(config)
    if backend_name == "claude-cli":
        return ClaudeCLIBackend(config)
    raise ValueError(f"不支持的教师后端：{backend_name}")


def _request(
    prompt: str,
    *,
    trajectory_id: str,
    round_index: int,
    environment_tick: int,
    remaining_ticks: int,
    observation_path: Path,
    previous_observation_path: Path | None,
    latest_state: dict[str, Any],
    previous_state: dict[str, Any] | None,
    previous_action: str | None,
    previous_result: dict[str, Any] | None,
    target_log_count: int = 1,
) -> TeacherRequest:
    return TeacherRequest(
        prompt,
        "\n".join(
            (
                "<trajectory_task>",
                f"trajectory_id: {trajectory_id}",
                f"task: 在当前 Minecraft 生存环境中探索并找到树，获得至少 {target_log_count} 块原木",
                f"success_criteria: 物品栏中所有原木方块的总数至少为 {target_log_count}",
                "failure_criteria: 死亡、环境异常或动作预算耗尽",
                "device: KeyboardMouse",
                "action_protocol: standard-input-action/v1",
                f"action_budget_ticks: {remaining_ticks}",
                "observation_policy: 当前轮已由 Observe 触发；在认为必须取得新画面的动作 tick 写 Observe（可位于序列中间）；Observe 必须与首个填充动作写在同一 tick，并在其后提供共 4 至 12 tick 的安全填充",
                "available_inputs: W A S D Space Shift Ctrl MouseLeft MouseRight MouseMove NoOp Observe",
                "control_mapping: Minecraft 默认键鼠映射；MouseMove 每单位改变 0.15 度，x 正数向右，y 正数向下；找到树后靠近并持续 MouseLeft 攻击树干，掉落后靠近拾取",
                "environment_setup: 未人工放置原木，需要完成观察、探索、寻路、破坏和拾取",
                "</trajectory_task>",
            )
        ),
        "\n".join(
            (
                "<trajectory_step>",
                f"step_index: {round_index}",
                f"environment_tick: {environment_tick}",
                f"remaining_action_ticks: {remaining_ticks}",
                f"latest_observation_id: observe-{round_index:03d}",
                "current_state:",
                _format_observable_state(latest_state),
                "previous_state:",
                _format_observable_state(previous_state),
                f"previous_action: {previous_action or '无；这是第一轮'}",
                f"previous_result: {_format_previous_result(previous_result)}",
                "termination_status: running",
                "</trajectory_step>",
                "附件顺序：如果有两张图片，第一张是上一轮决策前的观察，第二张是当前观察；第一轮只有当前观察。",
                "本次调用已经由 Observe 触发。控制块首 tick 不得重复 Observe。需要再次观察时必须写成 Observe W、Observe MouseLeft 或 Observe NoOp，禁止裸写 Observe；其后继续输出填充动作。",
                "只输出 standard-input-action/v1 动作序列，不要输出分析、解释、计划、Markdown 或其他文字。",
            )
        ),
        (
            (previous_observation_path, observation_path)
            if previous_observation_path is not None
            else (observation_path,)
        ),
    )


def _format_observable_state(state: dict[str, Any] | None) -> str:
    if not state:
        return "  无；这是第一轮"
    inventory = state.get("inventory") or []
    inventory_text = (
        ", ".join(f"{item.get('item', 'unknown')} x{item.get('count', 0)}" for item in inventory)
        or "空"
    )
    position = state.get("position") or []
    position_text = ", ".join(str(value) for value in position) or "未知"
    return "\n".join(
        (
            f"  位置: {position_text}",
            f"  yaw: {state.get('yaw', '未知')}",
            f"  pitch: {state.get('pitch', '未知')}",
            f"  生命值: {state.get('health', '未知')}",
            f"  物品栏: {inventory_text}",
            f"  准星命中方块: {state.get('raycast_block') or '环境未返回'}",
        )
    )


def _format_previous_result(result: dict[str, Any] | None) -> str:
    if not result:
        return "无；这是第一轮"
    return "; ".join(f"{key}={value}" for key, value in result.items())


def _format_executed_action_history(execution_ticks: list[dict[str, Any]]) -> str:
    if not execution_ticks:
        return "无"
    groups: list[tuple[str, int]] = []
    for item in execution_ticks:
        inputs = " ".join(item.get("inputs") or ()) or "NoOp"
        if groups and groups[-1][0] == inputs:
            groups[-1] = (inputs, groups[-1][1] + 1)
        else:
            groups.append((inputs, 1))
    return " ; ".join(action if count == 1 else f"{action} x{count}" for action, count in groups)


def _state_summary(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict) or info.get("full") is None:
        return {}
    full = info["full"]
    inventory = [
        {"item": item.translation_key, "count": item.count} for item in full.inventory if item.count
    ]
    raycast = getattr(full, "raycast_result", None)
    target = getattr(raycast, "target_block", None)
    return {
        "position": [full.x, full.y, full.z],
        "yaw": full.yaw,
        "pitch": full.pitch,
        "health": full.health,
        "inventory": inventory,
        "raycast_block": getattr(target, "translation_key", None) or None,
    }


def _logical_fingerprint(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": [round(float(value), 4) for value in state.get("position", ())],
        "yaw": round(float(state.get("yaw", 0.0)), 3),
        "pitch": round(float(state.get("pitch", 0.0)), 3),
        "health": round(float(state.get("health", 0.0)), 3),
        "inventory": state.get("inventory", []),
        "raycast_block": state.get("raycast_block"),
    }


def _assert_same_logical_state(states: tuple[dict[str, Any], ...], context: str) -> None:
    fingerprints = tuple(_logical_fingerprint(state) for state in states)
    if not fingerprints or not fingerprints[0].get("position"):
        raise RuntimeError(f"{context}缺少可验证的完整玩家状态")
    if any(value != fingerprints[0] for value in fingerprints[1:]):
        raise RuntimeError(f"{context}不一致：{json.dumps(fingerprints, ensure_ascii=False)}")


def _log_count(info: Any) -> int:
    state = _state_summary(info)
    return sum(
        int(item["count"])
        for item in state.get("inventory", [])
        if item["item"].split(".")[-1].endswith("_log")
    )


def _has_log(info: Any, target_log_count: int = 1) -> bool:
    return _log_count(info) >= target_log_count


def _require_linux_runtime() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("CraftGround 实际执行必须位于 Linux；Windows 必须通过 WSL 2 调用")


def run(
    output_directory: Path,
    *,
    action_budget_ticks: int = 512,
    max_generations: int = 10,
    warmup_ticks: int = 20,
    backend_name: str | None = None,
    backend: Any | None = None,
    port_base: int | None = None,
    environment_factory: Callable[..., Any] = create_environment,
    enforce_wsl: bool = True,
    use_shared_memory: bool = True,
    baseline_world_path: Path | None = None,
    target_log_count: int = 1,
    trajectory_count: int = TRAJECTORY_COUNT,
    initialization_workers: int | None = None,
) -> Path:
    """执行四条可比较分支，并写出轨迹、审核和相对优势产物。"""
    if (
        action_budget_ticks < 1
        or max_generations < 1
        or warmup_ticks < 0
        or target_log_count < 1
        or trajectory_count < 1
        or (initialization_workers is not None and initialization_workers < 1)
    ):
        raise ValueError("预算和生成轮数必须为正，warmup_ticks 不能为负")
    if enforce_wsl:
        _require_linux_runtime()
    from craftground.environment.action_space import no_op_v2

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    progress = _ProgressReporter(output_directory / "progress.jsonl")
    selected_backend_name = backend_name or os.getenv("TEACHER_BACKEND", "openai-api")
    selected_backend = backend or _load_backend(selected_backend_name)
    selected_port_base = port_base or int(os.getenv("CRAFTGROUND_PORT_BASE", "18300"))
    run_id = uuid.uuid4().hex
    prompt = (
        Path(__file__).resolve().parents[1]
        / "online_environment_interaction_agents"
        / "TRAJECTORY_GENERATION_PROMPT.md"
    ).read_text(encoding="utf-8")
    trajectory_specs = tuple(
        (f"T{index + 1:02d}", selected_backend_name) for index in range(trajectory_count)
    )

    progress.emit(
        "run_started",
        output_directory=str(output_directory),
        backend=selected_backend_name,
        model=selected_backend.model,
        trajectory_count=trajectory_count,
        action_budget_ticks=action_budget_ticks,
        max_generations=max_generations,
    )
    environment_values = []
    for index in range(trajectory_count):
        instance_id = f"four-teacher-{run_id}-{index}"
        progress.emit(
            "runtime_creation_started",
            environment_slot=index,
            instance_id=instance_id,
            port=selected_port_base + index,
        )
        environment_values.append(
            environment_factory(
                port=selected_port_base + index,
                instance_id=instance_id,
                use_shared_memory=use_shared_memory,
                baseline_world_path=baseline_world_path,
                verbose=False,
            )
        )
        progress.emit(
            "runtime_creation_completed",
            environment_slot=index,
            instance_id=instance_id,
            port=selected_port_base + index,
        )
    environments = tuple(environment_values)

    baseline_world_instances = tuple(
        getattr(environment, "tao_baseline_world", None) for environment in environments
    )
    if baseline_world_path is not None:
        if any(value is None for value in baseline_world_instances):
            for environment in environments:
                environment.close()
            raise RuntimeError("所有环境必须安装固定基准存档")
        installed_baselines = tuple(
            value for value in baseline_world_instances if value is not None
        )
        source_hashes = {value["source_sha256"] for value in installed_baselines}
        instance_paths = {value["instance_world_path"] for value in installed_baselines}
        if len(source_hashes) != 1 or len(instance_paths) != trajectory_count:
            for environment in environments:
                environment.close()
            raise RuntimeError("固定基准存档哈希不一致或实例路径未隔离")

    try:

        def initialize(index_and_environment: tuple[int, Any]) -> tuple[Any, Any]:
            index, environment = index_and_environment
            progress.emit("environment_reset_started", environment_slot=index)
            observation, info = environment.reset(options={"fast_reset": False})
            for _ in range(warmup_ticks):
                observation, _, _, _, info = environment.step(no_op_v2())
            for command in NORMALIZATION_COMMANDS:
                environment.add_command(command)
            for _ in range(3):
                observation, _, _, _, info = environment.step(no_op_v2())
            progress.emit("environment_reset_completed", environment_slot=index)
            return observation, info

        with ThreadPoolExecutor(
            max_workers=initialization_workers or trajectory_count,
            thread_name_prefix="craftground-initialize",
        ) as initialization_executor:
            initialized = tuple(initialization_executor.map(initialize, enumerate(environments)))
        start_observations = tuple(value[0] for value in initialized)
        start_states = tuple(_state_summary(value[1]) for value in initialized)
        if any(
            _logical_fingerprint(state) != _logical_fingerprint(start_states[0])
            for state in start_states[1:]
        ):
            target_position = start_states[0]["position"]
            target_yaw = start_states[0]["yaw"]
            target_pitch = start_states[0]["pitch"]
            synchronized = []
            for environment in environments:
                environment.add_command(
                    "tp @p "
                    + " ".join(str(value) for value in target_position)
                    + f" {target_yaw} {target_pitch}"
                )
                synchronized_step = None
                for _ in range(4):
                    synchronized_step = environment.step(no_op_v2())
                if synchronized_step is None:
                    raise RuntimeError("玩家起点同步没有执行环境 step")
                synchronized.append(synchronized_step)
            start_observations = tuple(value[0] for value in synchronized)
            start_states = tuple(_state_summary(value[4]) for value in synchronized)
        _assert_same_logical_state(start_states, "规范化后的四实例状态")

        shared_position = start_states[0]["position"]
        shared_yaw = start_states[0]["yaw"]
        shared_pitch = start_states[0]["pitch"]
        player_restore_commands = (
            "clear @p",
            "tp @p "
            + " ".join(str(value) for value in shared_position)
            + f" {shared_yaw} {shared_pitch}",
        )

        expected_start_fingerprint = _logical_fingerprint(start_states[0])

        def restore_player_start(environment: Any) -> tuple[tuple[Any, ...], int]:
            restored_step = None
            for attempt in range(1, 6):
                for command in player_restore_commands:
                    environment.add_command(command)
                for _ in range(4):
                    restored_step = environment.step(no_op_v2())
                if restored_step is None:
                    raise RuntimeError("玩家共享起点恢复没有执行环境 step")
                restored_state = _state_summary(restored_step[4])
                if _logical_fingerprint(restored_state) == expected_start_fingerprint:
                    return restored_step, attempt
            raise RuntimeError("玩家共享起点恢复在 5 次提交和状态回读后仍未通过")

        region = SnapshotRegion.around_player(start_states[0]["position"])
        coordinator = MemorySnapshotCoordinator(environments)
        snapshot = coordinator.capture_all(SNAPSHOT_ID, region)
        progress.emit("shared_snapshot_captured", snapshot_id=snapshot.snapshot_id)
        _save_rgb(start_observations[0], output_directory / "shared-start.png")

        for index, environment in enumerate(environments):
            environment.add_command(f"tp @p ~{index + 1} ~ ~")
            environment.step(no_op_v2())
        coordinator.reset_all(snapshot)
        with ThreadPoolExecutor(max_workers=trajectory_count) as verification_executor:
            restored = tuple(verification_executor.map(restore_player_start, environments))
        restored_states = tuple(_state_summary(value[0][4]) for value in restored)
        _assert_same_logical_state(restored_states, "快照倒档后的四实例状态")
        if any(
            _logical_fingerprint(state) != _logical_fingerprint(start_states[0])
            for state in restored_states
        ):
            raise RuntimeError("快照倒档未恢复到已记录的共享起点")
        progress.emit(
            "restore_probe_completed",
            passed=True,
            attempts=[value[1] for value in restored],
        )

        manifest = {
            "run_id": run_id,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_region": {
                "minimum": list(region.minimum),
                "maximum": list(region.maximum),
            },
            "normalization_commands": list(NORMALIZATION_COMMANDS),
            "player_restore_commands": list(player_restore_commands),
            "restore_probe_attempts": [value[1] for value in restored],
            "state_fingerprints": [_logical_fingerprint(state) for state in start_states],
            "restore_probe_passed": True,
            "backend_name": selected_backend_name,
            "provider": selected_backend.provider,
            "model": selected_backend.model,
            "environment_transport_backend": ("shared_memory" if use_shared_memory else "socket"),
            "baseline_world": {
                "required": baseline_world_path is not None,
                "source_path": (
                    str(baseline_world_path.resolve()) if baseline_world_path is not None else None
                ),
                "instances": list(baseline_world_instances),
            },
        }
        (output_directory / "shared-start-state.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        def simulate(environment: Any, spec: tuple[str, str]) -> dict[str, Any]:
            trajectory_id, arm_backend_name = spec
            trajectory_directory = output_directory / trajectory_id
            trajectory_directory.mkdir(parents=True, exist_ok=True)
            restored_step, player_restore_attempts = restore_player_start(environment)
            observation, _, terminated, truncated, info = restored_step
            wall_started_at = _utc_now()
            monotonic_started_at = time.perf_counter()
            progress.emit(
                "trajectory_started",
                trajectory_id=trajectory_id,
                action_budget_ticks=action_budget_ticks,
                max_generations=max_generations,
            )
            executor = TeacherTrajectoryExecutor(
                environment,
                selected_backend,
                adapter=CraftGroundKeyboardMouseAdapter(),
            )
            executed_ticks = 0
            total_reward = 0.0
            previous_result = None
            previous_action = None
            previous_state = None
            previous_observation_path = None
            trajectory_error = None
            trajectory_success = _has_log(info, target_log_count)
            for round_index in range(max_generations):
                if (
                    terminated
                    or truncated
                    or trajectory_success
                    or executed_ticks >= action_budget_ticks
                ):
                    break
                observation_path = trajectory_directory / f"observe-{round_index:03d}.png"
                _save_rgb(observation, observation_path)
                round_execution_start = len(executor.execution_ticks)
                round_reward_start = executor.total_reward
                result = None
                rejection = None
                for semantic_attempt in range(1, 4):
                    attempt_start_tick = executor.compiler.current_tick
                    attempt_start_reward = executor.total_reward
                    retry_result = previous_result
                    if rejection is not None:
                        retry_result = {
                            **(previous_result or {}),
                            "format_rejection": rejection,
                            "semantic_attempt": semantic_attempt,
                        }
                    try:
                        progress.emit(
                            "generation_started",
                            trajectory_id=trajectory_id,
                            round_index=round_index,
                            semantic_attempt=semantic_attempt,
                            environment_tick=executor.compiler.current_tick,
                            remaining_ticks=action_budget_ticks - executed_ticks,
                        )
                        result = executor.execute_generation(
                            _request(
                                prompt,
                                trajectory_id=trajectory_id,
                                round_index=round_index,
                                environment_tick=executor.compiler.current_tick,
                                remaining_ticks=action_budget_ticks - executed_ticks,
                                observation_path=observation_path,
                                previous_observation_path=previous_observation_path,
                                latest_state=_state_summary(info),
                                previous_state=previous_state,
                                previous_action=previous_action,
                                previous_result=retry_result,
                                target_log_count=target_log_count,
                            ),
                            observation=observation,
                            info=info,
                            remaining_action_ticks=action_budget_ticks - executed_ticks,
                        )
                        progress.emit(
                            "generation_completed",
                            trajectory_id=trajectory_id,
                            round_index=round_index,
                            semantic_attempt=semantic_attempt,
                            completed_ticks=result.completed_ticks,
                            reward=result.total_reward,
                            terminated=result.terminated,
                            truncated=result.truncated,
                            observation_requested=result.observation_requested,
                        )
                        break
                    except TeacherModelError as error:
                        rejection = f"{type(error).__name__}: {error}"
                        progress.emit(
                            "generation_rejected",
                            trajectory_id=trajectory_id,
                            round_index=round_index,
                            semantic_attempt=semantic_attempt,
                            error=rejection,
                        )
                        progressed_ticks = executor.compiler.current_tick - attempt_start_tick
                        if progressed_ticks:
                            executed_ticks += progressed_ticks
                            total_reward += executor.total_reward - attempt_start_reward
                            observation = executor.latest_observation
                            info = executor.latest_info
                            observation_path = trajectory_directory / (
                                f"observe-{round_index:03d}-retry-{semantic_attempt:02d}.png"
                            )
                            _save_rgb(observation, observation_path)
                            previous_result = {
                                "completed_ticks": progressed_ticks,
                                "reward": executor.total_reward - attempt_start_reward,
                                "terminated": False,
                                "truncated": False,
                                "fill_progress_recovered_after_rejection": True,
                            }
                            if executed_ticks >= action_budget_ticks:
                                break
                    except Exception as error:
                        # 单条 arm 的任何失败都记录到该 arm 的结果里，不影响其余三条。
                        trajectory_error = f"{type(error).__name__}: {error}"
                        progress.emit(
                            "trajectory_failed",
                            trajectory_id=trajectory_id,
                            round_index=round_index,
                            error=trajectory_error,
                        )
                        break
                if trajectory_error is not None:
                    break
                if result is None:
                    trajectory_error = rejection
                    break
                state_before_action = _state_summary(info)
                observation = result.observation
                info = result.info
                terminated = result.terminated
                truncated = result.truncated
                executed_ticks += result.completed_ticks
                total_reward += result.total_reward
                previous_result = {
                    "completed_ticks": len(executor.execution_ticks) - round_execution_start,
                    "reward": executor.total_reward - round_reward_start,
                    "terminated": terminated,
                    "truncated": truncated,
                }
                previous_state = state_before_action
                previous_action = _format_executed_action_history(
                    executor.execution_ticks[round_execution_start:]
                )
                progress.emit(
                    "action_executed",
                    trajectory_id=trajectory_id,
                    round_index=round_index,
                    action=previous_action,
                    completed_ticks=previous_result["completed_ticks"],
                    total_executed_ticks=executed_ticks,
                    reward=previous_result["reward"],
                    total_reward=total_reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                previous_observation_path = observation_path
                trajectory_success = _has_log(info, target_log_count)
                if not result.observation_requested:
                    break

            _save_rgb(observation, trajectory_directory / "end.png")
            trajectory_json = executor.export(
                trajectory_directory / "trajectory.json", trajectory_id=trajectory_id
            )
            trajectory_markdown = executor.export_markdown(
                trajectory_directory / "trajectory.md", trajectory_id=trajectory_id
            )
            wall_finished_at = _utc_now()
            summary = {
                "trajectory_id": trajectory_id,
                "backend_name": arm_backend_name,
                "provider": selected_backend.provider,
                "model": selected_backend.model,
                "wall_clock_started_at": wall_started_at,
                "wall_clock_finished_at": wall_finished_at,
                "wall_clock_duration_seconds": round(time.perf_counter() - monotonic_started_at, 6),
                "action_protocol": "standard-input-action/v1",
                "action_protocol_version": "v1",
                "action_backend": "keyboard_and_mouse_only",
                "action_adapter": "CraftGroundKeyboardMouseAdapter",
                "environment_transport_backend": (
                    "shared_memory" if use_shared_memory else "socket"
                ),
                "player_restore_attempts": player_restore_attempts,
                "executed_actions": [
                    {
                        "tick": tick["tick"],
                        "inputs": tick["inputs"],
                        "native_action": tick["native_action"],
                    }
                    for tick in executor.execution_ticks
                ],
                "executed_ticks": executed_ticks,
                "total_reward": total_reward,
                "generation_count": len(executor.compiler.generation_records),
                "trajectory_error": trajectory_error,
                "trajectory_success": trajectory_success,
                "final_log_count": _log_count(info),
                "target_log_count": target_log_count,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_state": _state_summary(info),
                "waiting_summary": executor.waiting_summary(),
                "artifact_paths": {
                    "trajectory_json": str(trajectory_json),
                    "trajectory_markdown": str(trajectory_markdown),
                    "final_observation": str(trajectory_directory / "end.png"),
                },
                "trajectory_json": str(trajectory_json),
                "trajectory_markdown": str(trajectory_markdown),
            }
            progress.emit(
                "trajectory_completed",
                trajectory_id=trajectory_id,
                duration_seconds=summary["wall_clock_duration_seconds"],
                executed_ticks=executed_ticks,
                generation_count=summary["generation_count"],
                total_reward=total_reward,
                success=trajectory_success,
                error=trajectory_error,
            )
            return summary

        requests = tuple(
            RolloutRequest(
                request_id=spec[0],
                subagent_id=f"{selected_backend_name}-arm-{index + 1}",
                snapshot=snapshot,
                payload=spec,
                simulate=simulate,
            )
            for index, spec in enumerate(trajectory_specs)
        )
        rollout_results = ParallelRolloutRunner(coordinator, max_workers=trajectory_count).run(
            requests
        )

        reviews = []
        summaries = []
        for rollout in rollout_results:
            summary = {
                **rollout.output,
                "environment_slot": rollout.environment_slot,
                "waited_ms": rollout.waited_ms,
                "restore_ms": rollout.restore_ms,
                "rollout_ms": rollout.rollout_ms,
            }
            summaries.append(summary)
            trajectory = json.loads(Path(summary["trajectory_json"]).read_text(encoding="utf-8"))
            reviews.append(
                review_trajectory(
                    trajectory,
                    summary,
                    action_budget_ticks=action_budget_ticks,
                )
            )
        if len(reviews) == 1:
            review = reviews[0]
            comparisons = (
                ComparisonSample(
                    trajectory_id=review.trajectory_id,
                    score=review.quality_score,
                    relative_advantage=0.0,
                    rank=1,
                    selected=True,
                ),
            )
        else:
            comparisons = build_comparison_group(reviews)
        comparison_review = review_comparison(comparisons)
        progress.emit(
            "comparison_completed",
            valid=comparison_review.valid,
            samples=[item.to_dict() for item in comparisons],
            selected_trajectory_ids=list(comparison_review.selected_trajectory_ids),
        )
        result_payload = {
            "test_kind": "same_policy_shared_start_four_arm_rollout",
            "action_protocol": "standard-input-action/v1",
            "backend": selected_backend_name,
            "provider": selected_backend.provider,
            "model": selected_backend.model,
            "action_budget_ticks_per_arm": action_budget_ticks,
            "target_log_count": target_log_count,
            "trajectory_count": trajectory_count,
            "shared_start": manifest,
            "trajectories": summaries,
            "trajectory_reviews": [item.to_dict() for item in reviews],
            "comparison_samples": [item.to_dict() for item in comparisons],
            "comparison_review": comparison_review.to_dict(),
        }
        (output_directory / "result.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        progress.emit("run_completed", result_path=str(output_directory / "result.json"))

        index_lines = [
            "# 四分支同策略同起点原木任务轨迹",
            "",
            f"- 后端：`{selected_backend_name}`",
            f"- 模型：`{selected_backend.model}`",
            f"- 快照：`{SNAPSHOT_ID}`",
            "- 起点与倒档验证：通过",
            f"- 比较复核：{'通过' if comparison_review.valid else '失败'}",
            "",
        ]
        comparison_by_id = {item.trajectory_id: item for item in comparisons}
        review_by_id = {item.trajectory_id: item for item in reviews}
        for summary in summaries:
            trajectory_id = summary["trajectory_id"]
            comparison = comparison_by_id[trajectory_id]
            review = review_by_id[trajectory_id]
            index_lines.extend(
                (
                    f"## {trajectory_id}",
                    "",
                    f"- 环境槽位：{summary['environment_slot']}",
                    f"- 墙钟：`{summary['wall_clock_started_at']}` → `{summary['wall_clock_finished_at']}`",
                    f"- 持续时间：{summary['wall_clock_duration_seconds']:.6f} 秒",
                    f"- 协议 / 适配器：`{summary['action_protocol']}` / `{summary['action_adapter']}`",
                    f"- 快照恢复：{summary['restore_ms']:.3f} ms",
                    f"- 并行推演：{summary['rollout_ms']:.3f} ms",
                    f"- 执行 tick：{summary['executed_ticks']} / {action_budget_ticks}",
                    f"- 奖励：{summary['total_reward']}",
                    f"- 终止 / 截断：{summary['terminated']} / {summary['truncated']}",
                    f"- 异常：{summary['trajectory_error'] or '无'}",
                    f"- 原木数量：{summary['final_log_count']} / {target_log_count}",
                    f"- 完成目标：{'是' if summary['trajectory_success'] else '否'}",
                    f"- 合同审核：{'通过' if review.contract_valid else '失败'}",
                    f"- 相对优势：{comparison.relative_advantage}（排名 {comparison.rank}）",
                    f"- 轨迹：[trajectory.md](<{trajectory_id}/trajectory.md>)",
                    "",
                )
            )
        index_path = output_directory / "README.md"
        index_path.write_text("\n".join(index_lines), encoding="utf-8")
        return index_path
    finally:
        for environment in environments:
            environment.close()


def main() -> None:
    import argparse

    from shared_tools.configuration import load_env_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-budget-ticks", type=int, default=512)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument(
        "--backend",
        choices=("openai-api", "anthropic-api", "codex-cli", "claude-cli"),
    )
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--socket-ipc", action="store_true")
    parser.add_argument("--baseline-world", type=Path)
    parser.add_argument("--target-log-count", type=int, default=1)
    parser.add_argument("--trajectory-count", type=int, default=TRAJECTORY_COUNT)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    if arguments.env_file is not None:
        load_env_file(arguments.env_file)
    print(
        run(
            arguments.output,
            action_budget_ticks=arguments.action_budget_ticks,
            max_generations=arguments.max_generations,
            warmup_ticks=arguments.warmup_ticks,
            backend_name=arguments.backend,
            port_base=arguments.port_base,
            use_shared_memory=not arguments.socket_ipc,
            baseline_world_path=arguments.baseline_world,
            target_log_count=arguments.target_log_count,
            trajectory_count=arguments.trajectory_count,
        )
    )


if __name__ == "__main__":
    main()
