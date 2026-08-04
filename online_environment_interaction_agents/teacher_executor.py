"""教师生成、动作转译、环境执行和轨迹导出闭环。"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from online_interactive_environments import (
    ActionSequenceCompiler,
    DecisionKind,
    GenerationTelemetry,
    UnderflowPolicy,
    parse_action_sequence_strict,
)
from online_interactive_environments.craftground.action_adapter import (
    CraftGroundKeyboardMouseAdapter,
)

from .teacher_trajectory import (
    TeacherBackend,
    TeacherModelError,
    TeacherRequest,
    parse_teacher_decision,
)


class StepEnvironment(Protocol):
    def step(self, action: dict[str, bool | float]) -> tuple[Any, float, bool, bool, Any]: ...


@dataclass(frozen=True)
class ExecutedTeacherGeneration:
    generation_id: str
    start_tick: int
    completed_ticks: int
    total_reward: float
    terminated: bool
    truncated: bool
    observation_requested: bool
    observation: Any
    info: Any
    control: str = ""


class TeacherTrajectoryExecutor:
    """以开启记录的转译器执行教师动作，并导出转译器事实。"""

    def __init__(
        self,
        environment: StepEnvironment,
        backend: TeacherBackend,
        *,
        adapter: CraftGroundKeyboardMouseAdapter | None = None,
    ) -> None:
        self.environment = environment
        self.backend = backend
        self.adapter = adapter or CraftGroundKeyboardMouseAdapter()
        self.compiler = ActionSequenceCompiler(
            UnderflowPolicy.WAIT,
            record_generations=True,
            auto_observe=True,
        )
        self.execution_ticks: list[dict[str, Any]] = []
        self.generation_contexts: dict[str, dict[str, Any]] = {}
        self.generation_latency: dict[str, dict[str, Any]] = {}
        self.model_decisions: dict[str, dict[str, str]] = {}
        self._tick_generation_ids: dict[int, str] = {}
        self.latest_observation: Any = None
        self.latest_info: Any = None
        self.total_reward = 0.0
        self.started_at = datetime.now(timezone.utc)

    def execute_generation(
        self,
        request: TeacherRequest,
        *,
        observation: Any,
        info: Any = None,
        remaining_action_ticks: int,
    ) -> ExecutedTeacherGeneration:
        if remaining_action_ticks < 1:
            raise ValueError("remaining_action_ticks 必须大于零")
        self.latest_observation = observation
        self.latest_info = info
        start_tick = self.compiler.current_tick
        generation_id = self.compiler.begin_generation(
            provider=self.backend.provider,
            model=self.backend.model,
        )
        if generation_id is None:
            raise RuntimeError("转译器生成记录未开启")
        self.generation_contexts[generation_id] = {
            "trigger": "Observe",
            "task_context": request.task_context,
            "step_context": request.step_context,
            "observation_paths": [str(path.resolve()) for path in request.observation_paths],
        }
        self.generation_latency[generation_id] = {
            "fill_executed_ticks": 0,
            "fill_execution_ms": 0.0,
            "uncovered_wait_ms": 0.0,
            "overwritten_future_ticks": 0,
        }
        response = None
        generation_finished = False
        completed_ticks = 0
        total_reward = 0.0
        terminated = False
        truncated = False

        def generate_response():
            stream = getattr(self.backend, "stream", None)
            return (
                self.backend.generate(request)
                if stream is None
                else stream(request, lambda chunk: None)
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(generate_response)
                fill_started = time.perf_counter()
                while not future.done() and not (terminated or truncated):
                    decision = self.compiler.pull()
                    if decision.kind is DecisionKind.OBSERVE:
                        self.compiler.observed()
                        continue
                    if decision.kind is DecisionKind.WAIT:
                        break
                    if decision.kind is not DecisionKind.ACTION or decision.action is None:
                        raise RuntimeError("转译器返回了未知决策")
                    if completed_ticks >= remaining_action_ticks:
                        raise TeacherModelError("延迟填充动作超过剩余 tick 预算")
                    observation, reward, terminated, truncated, info = self._execute_decision(
                        decision,
                        fallback_generation_id=generation_id,
                        latency_fill=True,
                    )
                    total_reward += float(reward)
                    completed_ticks += 1
                    self.generation_latency[generation_id]["fill_executed_ticks"] += 1
                self.generation_latency[generation_id]["fill_execution_ms"] = round(
                    (time.perf_counter() - fill_started) * 1000, 3
                )
                wait_started = time.perf_counter()
                response = future.result()
                self.generation_latency[generation_id]["uncovered_wait_ms"] = round(
                    (time.perf_counter() - wait_started) * 1000, 3
                )
            if terminated or truncated:
                self.compiler.end_generation(error="环境在延迟填充期间终止")
                generation_finished = True
                return ExecutedTeacherGeneration(
                    generation_id,
                    start_tick,
                    completed_ticks,
                    total_reward,
                    bool(terminated),
                    bool(truncated),
                    False,
                    observation,
                    info,
                )
            self.model_decisions[generation_id] = {
                "raw_response": response.text,
            }
            model_decision = parse_teacher_decision(response.text)
            self.model_decisions[generation_id]["non_control_text"] = (
                model_decision.non_control_text
            )
            if model_decision.non_control_text:
                raise TeacherModelError("教师输出不符合动作协议：动作控制块之外包含非协议文字")
            sequence = parse_action_sequence_strict(model_decision.control)
            if sequence.device != "KeyboardMouse":
                raise TeacherModelError(f"CraftGround 不支持设备：{sequence.device}")
            available_ticks = remaining_action_ticks - completed_ticks
            if len(sequence.ticks) > available_ticks:
                raise TeacherModelError(
                    f"教师输出 {len(sequence.ticks)} tick，超过剩余预算 {available_ticks} tick"
                )
            if not sequence.ticks or sequence.ticks[0].observe:
                raise TeacherModelError("教师输出没有产生 Observe 之前的可执行 tick")
            preview_adapter = CraftGroundKeyboardMouseAdapter(
                selected_hotbar=self.adapter.selected_hotbar,
                action_factory=self.adapter.action_factory,
            )
            for tick in sequence.ticks:
                preview_adapter.convert(tick)
            submission = self.compiler.submit(model_decision.control)
            if submission.accepted_ticks < 1:
                raise TeacherModelError("教师输出没有产生可执行 tick")
            self.generation_latency[generation_id]["overwritten_future_ticks"] = (
                submission.overwritten_ticks
            )
            self._tick_generation_ids = {
                tick: source
                for tick, source in self._tick_generation_ids.items()
                if tick < submission.start_tick
            }
            for tick in range(
                submission.start_tick,
                submission.start_tick + submission.accepted_ticks,
            ):
                self._tick_generation_ids[tick] = generation_id
            telemetry = GenerationTelemetry(
                request_id=response.request_id,
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_generation_ms=response.elapsed_ms,
            )
            self.compiler.end_generation(telemetry=telemetry)
            self.model_decisions[generation_id]["control"] = model_decision.control
            generation_finished = True
        except Exception as error:
            telemetry = None
            if response is not None:
                telemetry = GenerationTelemetry(
                    request_id=response.request_id,
                    provider=response.provider,
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_generation_ms=response.elapsed_ms,
                )
            if not generation_finished:
                self.compiler.end_generation(telemetry=telemetry, error=str(error))
            self.compiler.discard_buffered_from_current_tick()
            if isinstance(error, TeacherModelError):
                raise
            raise TeacherModelError(str(error)) from error

        observation_requested = False
        while not (terminated or truncated or observation_requested):
            decision = self.compiler.pull()
            if decision.kind is DecisionKind.OBSERVE:
                self.compiler.observed()
                observation_requested = True
                break
            if decision.kind is DecisionKind.WAIT:
                raise RuntimeError("完整动作序列提交后出现意外 WAIT")
            if decision.kind is not DecisionKind.ACTION or decision.action is None:
                raise RuntimeError("转译器返回了未知决策")
            if completed_ticks >= remaining_action_ticks:
                raise TeacherModelError("教师流式动作超过剩余 tick 预算")
            if decision.device != "KeyboardMouse":
                raise TeacherModelError(f"CraftGround 不支持设备：{decision.device}")
            observation, reward, terminated, truncated, info = self._execute_decision(
                decision,
                fallback_generation_id=generation_id,
                latency_fill=False,
            )
            total_reward += float(reward)
            completed_ticks += 1
        return ExecutedTeacherGeneration(
            generation_id,
            start_tick,
            completed_ticks,
            total_reward,
            bool(terminated),
            bool(truncated),
            observation_requested,
            observation,
            info,
            model_decision.control,
        )

    def _execute_decision(
        self,
        decision: Any,
        *,
        fallback_generation_id: str,
        latency_fill: bool,
    ) -> tuple[Any, float, bool, bool, Any]:
        if decision.device != "KeyboardMouse" or decision.action is None:
            raise TeacherModelError(f"CraftGround 不支持设备：{decision.device}")
        native_action = self.adapter.convert(decision.action)
        step_started = time.perf_counter()
        observation, reward, terminated, truncated, info = self.environment.step(native_action)
        step_elapsed_ms = (time.perf_counter() - step_started) * 1000
        source_generation_id = self._tick_generation_ids.pop(decision.tick, fallback_generation_id)
        self.compiler.commit(decision)
        self.latest_observation = observation
        self.latest_info = info
        self.total_reward += float(reward)
        self.execution_ticks.append(
            {
                "tick": decision.tick,
                "generation_id": source_generation_id,
                "inference_generation_id": fallback_generation_id,
                "latency_fill": latency_fill,
                "device": decision.device,
                "inputs": list(decision.action.inputs),
                "observe": decision.action.observe,
                "native_action": native_action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "step_elapsed_ms": round(step_elapsed_ms, 3),
                "info": _compact_environment_value(info),
            }
        )
        return observation, reward, terminated, truncated, info

    def export(self, path: Path, *, trajectory_id: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        waiting_summary = self.waiting_summary()
        payload = {
            "trajectory_id": trajectory_id,
            "action_protocol": "standard-input-action/v1",
            "action_protocol_version": "v1",
            "action_backend": "keyboard_and_mouse_only",
            "action_adapter": type(self.adapter).__name__,
            "compiler_record_generations": self.compiler.record_generations,
            "started_at": self.started_at.isoformat(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "current_tick": self.compiler.current_tick,
            "waiting_summary": waiting_summary,
            "generation_records": [
                _json_value(record) for record in self.compiler.generation_records
            ],
            "generation_contexts": self.generation_contexts,
            "generation_latency": self.generation_latency,
            "model_decisions": self.model_decisions,
            "execution_ticks": self.execution_ticks,
        }
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def export_markdown(self, path: Path, *, trajectory_id: str) -> Path:
        waiting_summary = self.waiting_summary()
        payload = {
            "trajectory_id": trajectory_id,
            "action_protocol": "standard-input-action/v1",
            "action_protocol_version": "v1",
            "action_backend": "keyboard_and_mouse_only",
            "action_adapter": type(self.adapter).__name__,
            "compiler_record_generations": self.compiler.record_generations,
            "started_at": self.started_at.isoformat(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "current_tick": self.compiler.current_tick,
            "waiting_summary": waiting_summary,
            "generation_records": [
                _json_value(record) for record in self.compiler.generation_records
            ],
            "generation_contexts": self.generation_contexts,
            "generation_latency": self.generation_latency,
            "model_decisions": self.model_decisions,
            "execution_ticks": self.execution_ticks,
        }
        return _write_trajectory_markdown(payload, path)

    def waiting_summary(self) -> dict[str, Any]:
        records = self.compiler.generation_records
        longest = max(
            records, key=lambda record: record.max_consecutive_waiting_ticks, default=None
        )
        latency_items = list(self.generation_latency.items())
        longest_wall = max(
            latency_items,
            key=lambda item: float(item[1].get("uncovered_wait_ms", 0.0)),
            default=None,
        )
        return {
            "total_waiting_ticks": self.compiler.total_waiting_ticks,
            "max_waiting_ticks": self.compiler.max_waiting_ticks,
            "total_uncovered_wait_ms": round(
                sum(float(value.get("uncovered_wait_ms", 0.0)) for _, value in latency_items),
                3,
            ),
            "max_waiting_ms": (
                None
                if longest_wall is None
                else float(longest_wall[1].get("uncovered_wait_ms", 0.0))
            ),
            "total_fill_executed_ticks": sum(
                int(value.get("fill_executed_ticks", 0)) for _, value in latency_items
            ),
            "total_overwritten_future_ticks": sum(
                int(value.get("overwritten_future_ticks", 0)) for _, value in latency_items
            ),
            "measurement": "wall_clock_inference_gap_and_logical_underflow",
            "max_generation_id": (
                longest_wall[0]
                if longest_wall is not None
                else (None if longest is None else longest.generation_id)
            ),
            "max_generation_sequence": (None if longest is None else longest.sequence_number),
        }


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_environment_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_environment_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            return list(value)
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": list(shape)}
    return {"type": type(value).__name__}


def export_trajectory_markdown(json_path: Path, markdown_path: Path) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return _write_trajectory_markdown(payload, markdown_path)


def _write_trajectory_markdown(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    generations = payload.get("generation_records", [])
    execution_ticks = payload.get("execution_ticks", [])
    contexts = payload.get("generation_contexts", {})
    generation_latency = payload.get("generation_latency", {})
    waiting_summary = payload.get("waiting_summary", {})
    lines = [
        f"# 教师交互轨迹：{payload.get('trajectory_id', 'unknown')}",
        "",
        "## 轨迹概览",
        "",
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| 动作协议 | `{payload.get('action_protocol', '')}` |",
        f"| 协议版本 | `{payload.get('action_protocol_version', '')}` |",
        f"| 执行后端 | `{payload.get('action_backend', '')}` |",
        f"| 动作适配器 | `{payload.get('action_adapter', '')}` |",
        f"| Observe / 模型输出轮数 | {len(generations)} |",
        f"| 已执行环境 tick | {len(execution_ticks)} |",
        f"| 累计 WAIT | {waiting_summary.get('total_waiting_ticks', 0)} tick |",
        (
            f"| 最长连续 WAIT | {waiting_summary.get('max_waiting_ticks', 0)} tick / "
            f"{waiting_summary.get('max_waiting_ms') or '不换算墙钟'} |"
        ),
        f"| 最长 WAIT 来源 | `{waiting_summary.get('max_generation_id') or ''}` |",
        f"| 转译器记录 | `{payload.get('compiler_record_generations', False)}` |",
        f"| 起始时间 | `{payload.get('started_at', '')}` |",
        f"| 导出时间 | `{payload.get('exported_at', '')}` |",
        "",
        "## 时序",
        "",
        "```text",
        "Observe → 教师模型输出 → 转译器展开 → 环境逐 tick 执行 → 下一次 Observe",
        "```",
        "",
    ]
    for index, generation in enumerate(generations):
        generation_id = generation.get("generation_id", f"generation-{index}")
        context = contexts.get(generation_id, {})
        ticks = [tick for tick in execution_ticks if tick.get("generation_id") == generation_id]
        telemetry = generation.get("telemetry", {})
        latency = generation_latency.get(generation_id, {})
        lines.extend(
            (
                f"## Observe {index} → 模型输出 {index}",
                "",
                (
                    f"本轮由 **{context.get('trigger', 'Observe')}** 触发，"
                    "模型输出经过转译器后提交给环境。"
                ),
                "",
            )
        )
        observation_paths = context.get("observation_paths", [])
        if not observation_paths and index == 0:
            inferred = path.with_name("start.png")
            if inferred.is_file():
                observation_paths = [str(inferred)]
        for observation_index, observation_path in enumerate(observation_paths):
            target = _relative_markdown_path(Path(observation_path), path.parent)
            lines.extend(
                (
                    f"### 触发观察 {observation_index}",
                    "",
                    f"![Observe {index}](<{target}>)",
                    "",
                )
            )
        if context.get("task_context"):
            lines.extend(("### 任务上下文", "", "````text", context["task_context"], "````", ""))
        if context.get("step_context"):
            lines.extend(
                ("### Observe 上下文", "", "````text", context["step_context"], "````", "")
            )
        raw_output = "".join(generation.get("input_chunks", []))
        lines.extend(
            (
                "### 教师模型原始输出",
                "",
                "````text",
                raw_output,
                "````",
                "",
                "### 生成与转译记录",
                "",
                "| 字段 | 值 |",
                "| --- | --- |",
                f"| Generation ID | `{generation_id}` |",
                f"| 状态 | `{generation.get('status', '')}` |",
                f"| Provider | `{telemetry.get('provider', '')}` |",
                f"| 模型 | `{telemetry.get('model', '')}` |",
                f"| Request ID | `{telemetry.get('request_id', '')}` |",
                f"| 模型生成耗时 | {telemetry.get('total_generation_ms', '')} ms |",
                f"| 推理期间已执行填充 | {latency.get('fill_executed_ticks', 0)} tick |",
                f"| 填充执行耗时 | {latency.get('fill_execution_ms', 0)} ms |",
                f"| 未覆盖等待时长 | {latency.get('uncovered_wait_ms', 0)} ms |",
                f"| 覆盖旧未来动作 | {latency.get('overwritten_future_ticks', 0)} tick |",
                f"| 累计 WAIT | {generation.get('waiting_ticks', 0)} tick |",
                (
                    "| 首内容前 WAIT | "
                    f"{generation.get('waiting_before_first_content_ticks', 0)} tick |"
                ),
                (
                    "| 首动作前 WAIT | "
                    f"{generation.get('waiting_before_first_action_ticks', 0)} tick |"
                ),
                (f"| 最长连续 WAIT | {generation.get('max_consecutive_waiting_ticks', 0)} tick |"),
                f"| 接受 tick | {generation.get('accepted_ticks', 0)} |",
                f"| 过期 tick | {generation.get('expired_ticks', 0)} |",
                f"| 覆盖 tick | {generation.get('overwritten_ticks', 0)} |",
                "",
                "### 环境执行序列",
                "",
                "| Tick | Observe | 延迟填充 | 协议动作 | 奖励 | 终止 | 环境 step 耗时 |",
                "| ---: | :---: | :---: | --- | ---: | :---: | ---: |",
            )
        )
        for tick in ticks:
            inputs = " ".join(tick.get("inputs", [])) or "NoOp"
            lines.append(
                f"| {tick.get('tick', '')} | "
                f"{'是' if tick.get('observe') else '否'} | "
                f"{'是' if tick.get('latency_fill') else '否'} | `{inputs}` | "
                f"{tick.get('reward', 0)} | "
                f"{'是' if tick.get('terminated') or tick.get('truncated') else '否'} | "
                f"{tick.get('step_elapsed_ms', '')} ms |"
            )
        if not ticks:
            lines.append("| - | - | - | 未执行 | - | - | - |")
        lines.extend(("", "### 本轮结果", ""))
        total_reward = sum(float(tick.get("reward", 0)) for tick in ticks)
        terminated = any(tick.get("terminated") for tick in ticks)
        truncated = any(tick.get("truncated") for tick in ticks)
        lines.extend(
            (
                "| 字段 | 值 |",
                "| --- | --- |",
                f"| 执行 tick | {len(ticks)} |",
                f"| 累计奖励 | {total_reward} |",
                f"| Terminated | `{terminated}` |",
                f"| Truncated | `{truncated}` |",
                "",
            )
        )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _relative_markdown_path(target: Path, directory: Path) -> str:
    try:
        return Path(os.path.relpath(target.resolve(), directory.resolve())).as_posix()
    except ValueError:
        return target.as_posix()
