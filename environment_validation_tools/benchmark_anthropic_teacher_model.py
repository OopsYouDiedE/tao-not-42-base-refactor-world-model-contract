"""对单张观察执行一次 Anthropic Messages 视觉教师基准。"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from online_environment_interaction_agents import (
    AnthropicCompatibleBackend,
    AnthropicCompatibleConfig,
    OpenAICompatibleBackend,
    OpenAICompatibleConfig,
    TeacherRequest,
)
from online_interactive_environments import parse_action_sequence
from shared_tools.configuration import load_env_file


def _request(image: Path) -> TeacherRequest:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "online_environment_interaction_agents"
        / "TRAJECTORY_GENERATION_PROMPT.md"
    ).read_text(encoding="utf-8")
    return TeacherRequest(
        prompt,
        "\n".join(
            (
                "<trajectory_task>",
                "trajectory_id: anthropic-api-latency-check",
                "task: 根据当前 Minecraft 画面规划接近并采集原木的动作",
                "success_criteria: 产生严格有效的键鼠动作序列",
                "device: KeyboardMouse",
                "action_protocol: standard-input-action/v1",
                "action_budget_ticks: 64",
                "</trajectory_task>",
            )
        ),
        "只输出一个 Device、Tick 和 <action> 动作块；中间最多一个 Observe，Observe 所在 tick 到末尾保留 4 至 12 tick 填充。",
        (image,),
    )


def run(image: Path, output: Path, *, wire_api: str = "anthropic") -> Path:
    timeout = float(os.getenv("TEACHER_TIMEOUT_SECONDS", "240"))
    if wire_api == "openai":
        config = OpenAICompatibleConfig(
            base_url=os.environ["TEACHER_API_URL"],
            api_key=os.environ["TEACHER_API_KEY"],
            model=os.environ["TEACHER_MODEL"],
            timeout_seconds=timeout,
        )
        backend = OpenAICompatibleBackend(config)
    else:
        config = AnthropicCompatibleConfig(
            base_url=os.environ["TEACHER_API_URL"],
            auth_token=os.environ["TEACHER_API_KEY"],
            model=os.environ["TEACHER_MODEL"],
            timeout_seconds=timeout,
        )
        backend = AnthropicCompatibleBackend(config)
    first_content_ms: float | None = None
    started = time.perf_counter()

    def on_chunk(chunk: str) -> None:
        nonlocal first_content_ms
        if chunk and first_content_ms is None:
            first_content_ms = (time.perf_counter() - started) * 1000

    response = backend.stream(_request(image.resolve()), on_chunk)
    error = None
    tick_count = None
    try:
        tick_count = len(parse_action_sequence(response.text.strip()).ticks)
    except ValueError as exception:
        error = str(exception)
    payload = {
        "test_kind": f"{wire_api}-api-single-observation-latency",
        "wall_clock_finished_at": datetime.now(timezone.utc).isoformat(),
        "base_url": config.base_url,
        "provider": backend.provider,
        "model": backend.model,
        "total_generation_ms": round(response.elapsed_ms, 3),
        "first_content_ms": None if first_content_ms is None else round(first_content_ms, 3),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "protocol_valid": error is None,
        "tick_count": tick_count,
        "error": error,
        "output": response.text,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = output / "REPORT.md"
    report.write_text(
        "\n".join(
            (
                "# Anthropic API 单图延迟测试",
                "",
                "| 指标 | 结果 |",
                "| --- | ---: |",
                f"| 模型 | `{backend.model}` |",
                f"| 首内容 | {payload['first_content_ms']} ms |",
                f"| 总时长 | {payload['total_generation_ms']} ms |",
                f"| 输入 token | {payload['input_tokens']} |",
                f"| 输出 token | {payload['output_tokens']} |",
                f"| 协议有效 | {payload['protocol_valid']} |",
                f"| 展开 tick | {payload['tick_count']} |",
                "",
            )
        ),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--wire-api", choices=("anthropic", "openai"), default="anthropic")
    arguments = parser.parse_args()
    if arguments.env_file is not None:
        load_env_file(arguments.env_file)
    print(run(arguments.image, arguments.output, wire_api=arguments.wire_api))


if __name__ == "__main__":
    main()
