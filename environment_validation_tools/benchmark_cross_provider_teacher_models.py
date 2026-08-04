"""使用同一观察和协议比较 Claude CLI 与 Codex CLI 教师模型。"""

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
    ClaudeCLIBackend,
    CLIConfig,
    CodexCLIBackend,
    TeacherRequest,
)
from online_interactive_environments import parse_action_sequence

MODEL_SPECS = (
    ("claude-opus-5", "claude-cli"),
    ("claude-sonnet-5", "claude-cli"),
    ("gpt-5.6-terra", "codex-cli"),
)


@dataclass(frozen=True)
class CrossProviderResult:
    model: str
    provider: str
    wall_clock_started_at: str
    wall_clock_finished_at: str
    total_generation_ms: float | None
    first_content_ms: float | None
    protocol_valid: bool
    tick_count: int
    observe_indices: tuple[int, ...]
    intermediate_observe: bool
    fill_ticks_after_first_observe: int
    fill_length_in_recommended_range: bool
    output: str
    error: str | None


def _request(image: Path) -> TeacherRequest:
    return TeacherRequest(
        system_prompt=(
            "你是 CraftGround 在线交互教师。只输出 standard-input-action/v1 动作块，不要解释。"
            "整个回答必须只有一个 Device 行、一个 Tick 行和一个 <action>...</action> 块。"
            "所有逻辑 tick 写在同一 action 块内并用分号分隔。"
            "Observe 是异步重规划边界，必须与首个填充动作写在同一 tick，禁止裸写 Observe。"
            "Observe 后的动作仍属于当前旧决策，是等待下一轮模型结果期间的延迟填充，不依赖新观察，"
            "并允许随时被下一轮正式动作覆盖。Observe 后提供 4 至 12 tick 安全填充。"
            "每个输出序列最多包含一个显式 Observe，填充内部禁止再次 Observe。"
        ),
        task_context=(
            "任务：根据同一张 Minecraft 起点画面，规划一段接近并采集原木的动作。"
            "可用输入：W A S D Space Shift Ctrl MouseLeft MouseRight MouseMove NoOp Observe。"
            "总输出不超过 48 tick。"
        ),
        step_context=(
            "本轮已经由观察触发，首 tick 不得 Observe。只输出一个动作块。"
            "当继续依赖当前观察已经不可靠时写中间 Observe，并在后面预先给出旧决策安全填充，"
            "防止下一轮推理期间环境停止。填充不得假设新观察内容。"
            "合法示例：\nDevice KeyboardMouse\nTick 0\n"
            "<action>W x4 ; Observe W ; W MouseMove 1 0 x4</action>"
        ),
        observation_paths=(image,),
    )


def _backend(model: str, provider: str, claude: str, codex: str):
    config = CLIConfig(
        model=model, executable=claude if provider == "claude-cli" else codex, timeout_seconds=240
    )
    return ClaudeCLIBackend(config) if provider == "claude-cli" else CodexCLIBackend(config)


def _run_one(
    model: str,
    provider: str,
    image: Path,
    claude: str,
    codex: str,
) -> CrossProviderResult:
    backend = _backend(model, provider, claude, codex)
    wall_started = datetime.now(timezone.utc)
    started = time.perf_counter()
    first_content_ms: float | None = None

    def on_chunk(_: str) -> None:
        nonlocal first_content_ms
        if first_content_ms is None:
            first_content_ms = (time.perf_counter() - started) * 1000

    output = ""
    error: str | None = None
    generation_ms: float | None = None
    valid = False
    tick_count = 0
    observe_indices: tuple[int, ...] = ()
    fill_ticks = 0
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
        if tick_count > 48:
            raise ValueError(f"动作展开为 {tick_count} tick，超过 48 tick 上限")
        observe_indices = tuple(index for index, tick in enumerate(sequence.ticks) if tick.observe)
        if observe_indices:
            fill_ticks = tick_count - observe_indices[0]
        valid = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return CrossProviderResult(
        model=model,
        provider=provider,
        wall_clock_started_at=wall_started.isoformat(),
        wall_clock_finished_at=datetime.now(timezone.utc).isoformat(),
        total_generation_ms=None if generation_ms is None else round(generation_ms, 3),
        first_content_ms=None if first_content_ms is None else round(first_content_ms, 3),
        protocol_valid=valid,
        tick_count=tick_count,
        observe_indices=observe_indices,
        intermediate_observe=bool(observe_indices and observe_indices[0] < tick_count - 1),
        fill_ticks_after_first_observe=fill_ticks,
        fill_length_in_recommended_range=4 <= fill_ticks <= 12,
        output=output,
        error=error,
    )


def run(
    output_directory: Path,
    *,
    image: Path,
    claude_executable: str = "claude",
    codex_executable: str = "codex",
) -> Path:
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    image = image.resolve()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    results: list[CrossProviderResult] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _run_one, model, provider, image, claude_executable, codex_executable
            ): model
            for model, provider in MODEL_SPECS
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda value: [spec[0] for spec in MODEL_SPECS].index(value.model))
    payload = {
        "benchmark": "cross-provider-observe-fill/v1",
        "observation_path": str(image),
        "parallel": True,
        "wall_clock_started_at": started_at.isoformat(),
        "wall_clock_finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_duration_seconds": round(time.perf_counter() - started, 6),
        "results": [asdict(result) for result in results],
    }
    (output_directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Claude Opus 5、Claude Sonnet 5 与 GPT-5.6 Terra 同标准测试",
        "",
        f"观察帧：`{image}`",
        "",
        "| 模型 | CLI | 协议有效 | 总时长 ms | 首内容 ms | tick | 中间 Observe | 填充 tick | 建议范围 | 异常 |",
        "| --- | --- | :---: | ---: | ---: | ---: | :---: | ---: | :---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.model}` | `{result.provider}` | {'是' if result.protocol_valid else '否'} | "
            f"{result.total_generation_ms or ''} | {result.first_content_ms or ''} | {result.tick_count} | "
            f"{'是' if result.intermediate_observe else '否'} | {result.fill_ticks_after_first_observe} | "
            f"{'是' if result.fill_length_in_recommended_range else '否'} | {result.error or '无'} |"
        )
    lines.extend(("", "## 原始输出"))
    for result in results:
        lines.extend(
            ("", f"### {result.model}", "", "```text", result.output or result.error or "", "```")
        )
    report = output_directory / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--claude-executable", default="claude")
    parser.add_argument("--codex-executable", default="codex")
    arguments = parser.parse_args()
    print(
        run(
            arguments.output,
            image=arguments.image,
            claude_executable=arguments.claude_executable,
            codex_executable=arguments.codex_executable,
        )
    )


if __name__ == "__main__":
    main()
