"""并行基准测试 GPT-5.6 教师模型的动作协议能力与真实调用时延。"""

from __future__ import annotations

import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from online_environment_interaction_agents.teacher_trajectory import (
    CLIConfig,
    CodexCLIBackend,
    TeacherRequest,
)
from online_interactive_environments import parse_action_sequence

MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    wall_clock_started_at: str
    wall_clock_finished_at: str
    total_generation_ms: float | None
    first_content_ms: float | None
    protocol_valid: bool
    tick_count: int
    observe_indices: tuple[int, ...]
    intermediate_observe: bool
    fill_ticks_after_first_observe: int
    output: str
    error: str | None


def _request(image: Path) -> TeacherRequest:
    return TeacherRequest(
        system_prompt=(
            "你是 CraftGround 在线交互教师。只输出 standard-input-action/v1 动作块，不要解释。"
            "整个回答必须只有一个 Device 行、一个 Tick 行和一个 <action>...</action> 块；"
            "所有 tick 必须写在同一个 action 块内并用分号分隔，严禁为每个 tick 重复 Device、Tick 或 action 标签。"
            "Observe 表示你认为必须取得新画面的动作位置，它可以位于序列中间且不占环境 tick。"
            "Observe 后继续输出 4 至 12 tick 安全、合理的短时延填充动作。"
            "这些填充仍属于当前旧决策，不依赖新观察；下一轮模型结果到达后会覆盖尚未执行的填充。"
        ),
        task_context=(
            "任务：根据同一张起点画面规划一段接近并采集原木的动作。"
            "可用输入：W A S D Space Shift Ctrl MouseLeft MouseRight MouseMove NoOp Observe。"
            "总输出不超过 48 tick。"
        ),
        step_context=(
            "本轮已经由观察触发。首 tick 不得 Observe。请在需要重新确认视角或目标的位置写中间 Observe，"
            "并在其后保留可随时中断的安全填充，防止下一轮推理期间环境停止。合法结构示例：\nDevice KeyboardMouse\nTick 0\n"
            "<action>W x4 ; Observe W ; W MouseMove 1 0 x4</action>"
        ),
        observation_paths=(image,),
    )


def _run_model(model: str, image: Path, executable: str) -> BenchmarkResult:
    backend = CodexCLIBackend(CLIConfig(model=model, executable=executable, timeout_seconds=240))
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    first_content_ms: float | None = None

    def on_chunk(_: str) -> None:
        nonlocal first_content_ms
        if first_content_ms is None:
            first_content_ms = (time.perf_counter() - started) * 1000

    output = ""
    error: str | None = None
    valid = False
    tick_count = 0
    observe_indices: tuple[int, ...] = ()
    fill_ticks = 0
    generation_ms: float | None = None
    try:
        response = backend.stream(_request(image), on_chunk)
        output = response.text.strip()
        generation_ms = response.elapsed_ms
        if len(re.findall(r"(?m)^Device ", output)) != 1:
            raise ValueError("整个回答必须只有一个 Device 行")
        if len(re.findall(r"(?m)^Tick ", output)) != 1:
            raise ValueError("整个回答必须只有一个 Tick 行")
        if output.count("<action>") != 1 or output.count("</action>") != 1:
            raise ValueError("整个回答必须只有一个 action 块")
        with warnings.catch_warnings(record=True) as parser_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            sequence = parse_action_sequence(output)
        if parser_warnings:
            messages = "；".join(str(item.message) for item in parser_warnings)
            raise ValueError(f"动作协议依赖容错解析：{messages}")
        if sequence.device != "KeyboardMouse":
            raise ValueError(f"设备必须是 KeyboardMouse，实际为 {sequence.device}")
        tick_count = len(sequence.ticks)
        observe_indices = tuple(index for index, tick in enumerate(sequence.ticks) if tick.observe)
        if observe_indices:
            fill_ticks = tick_count - observe_indices[0]
        valid = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished_at = datetime.now(timezone.utc)
    return BenchmarkResult(
        model=model,
        wall_clock_started_at=started_at.isoformat(),
        wall_clock_finished_at=finished_at.isoformat(),
        total_generation_ms=None if generation_ms is None else round(generation_ms, 3),
        first_content_ms=None if first_content_ms is None else round(first_content_ms, 3),
        protocol_valid=valid,
        tick_count=tick_count,
        observe_indices=observe_indices,
        intermediate_observe=bool(observe_indices and observe_indices[0] < tick_count - 1),
        fill_ticks_after_first_observe=fill_ticks,
        output=output,
        error=error,
    )


def run(output_directory: Path, *, image: Path, executable: str = "codex") -> Path:
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    wall_started = datetime.now(timezone.utc)
    monotonic_started = time.perf_counter()
    results: list[BenchmarkResult] = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {executor.submit(_run_model, model, image, executable): model for model in MODELS}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda value: MODELS.index(value.model))
    payload = {
        "benchmark": "gpt-5.6-teacher-same-observation/v1",
        "models": list(MODELS),
        "observation_path": str(image),
        "wall_clock_started_at": wall_started.isoformat(),
        "wall_clock_finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_duration_seconds": round(time.perf_counter() - monotonic_started, 6),
        "parallel": True,
        "cli_streaming_note": (
            "Codex CLI 当前在 item.completed/agent_message 事件才交付文本；first_content_ms "
            "通常等于完整 agent_message 可用时间，不代表 token 级首字延迟。"
        ),
        "results": [asdict(result) for result in results],
    }
    (output_directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# GPT-5.6 CraftGround 教师同条件基准",
        "",
        f"观察帧：`{image}`",
        "",
        "| 模型 | 协议有效 | 总生成时长 ms | 首内容可用 ms | tick | 中间 Observe | Observe 位置 | 后续填充 tick | 异常 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.model}` | {'是' if result.protocol_valid else '否'} | "
            f"{result.total_generation_ms or ''} | {result.first_content_ms or ''} | {result.tick_count} | "
            f"{'是' if result.intermediate_observe else '否'} | {list(result.observe_indices)} | "
            f"{result.fill_ticks_after_first_observe} | {result.error or '无'} |"
        )
    lines.extend(
        (
            "",
            "## 时延解释",
            "",
            "当前 Codex CLI 在完整 `agent_message` 完成时才交付动作文本。因此首内容可用时间接近总生成时间，"
            "无法依靠 token 级流式解析提前执行首动作。加速优先级是减少模型调用次数、缩短输出、并行推理，"
            "以及在 Observe 后执行安全填充；若要获得真正的首动作流式加速，需要改用能够逐文本增量返回的接口。",
            "",
            "## 原始输出",
        )
    )
    for result in results:
        lines.extend(
            (
                "",
                f"### {result.model}",
                "",
                "```text",
                result.output or result.error or "",
                "```",
            )
        )
    report = output_directory / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--executable", default="codex")
    arguments = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = arguments.output or Path("runs") / f"gpt56-teacher-benchmark-{timestamp}"
    print(run(output, image=arguments.image, executable=arguments.executable))


if __name__ == "__main__":
    main()
