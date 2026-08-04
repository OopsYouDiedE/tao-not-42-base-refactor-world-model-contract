"""用一条 WSL 命令执行固定存档、四轨迹和教师测评完整流程。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from environment_validation_tools.create_craftground_baseline_world import (
    create_baseline_world,
)
from environment_validation_tools.run_four_teacher_trajectories import run as run_four
from shared_tools.configuration import load_env_file


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"complete-teacher-evaluation-{timestamp}"


def _relative_link(target: Path, report_directory: Path) -> str:
    return target.resolve().relative_to(report_directory.resolve()).as_posix()


def _write_report(
    output_directory: Path,
    *,
    trajectory_directory: Path,
    baseline_world: Path,
    baseline_created: bool,
) -> Path:
    result_path = trajectory_directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected = ", ".join(result["comparison_review"]["selected_trajectory_ids"])
    lines = [
        "# CraftGround 四轨迹教师测评报告",
        "",
        "## 运行配置",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| IPC | `{result['shared_start']['environment_transport_backend']}` |",
        f"| 教师后端 | `{result['backend']}` |",
        f"| 教师模型 | `{result['model']}` |",
        f"| 动作协议 | `{result['action_protocol']}` |",
        f"| 单轨迹动作预算 | {result['action_budget_ticks_per_arm']} tick |",
        f"| 基准存档 | `{baseline_world.resolve()}` |",
        f"| 本轮创建基准存档 | {'是' if baseline_created else '否'} |",
        f"| 共享起点恢复探针 | {'通过' if result['shared_start']['restore_probe_passed'] else '失败'} |",
        f"| 比较审核 | {'通过' if result['comparison_review']['valid'] else '失败'} |",
        f"| 最佳轨迹 | {selected or '无'} |",
        "",
        "## 轨迹结果",
        "",
        "| 轨迹 | 环境槽位 | 教师生成轮次 | 执行 tick | 墙钟秒 | 任务成功 | 异常 | 排名 | 相对优势 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    comparisons = {value["trajectory_id"]: value for value in result["comparison_samples"]}
    for trajectory in result["trajectories"]:
        comparison = comparisons[trajectory["trajectory_id"]]
        lines.append(
            "| {trajectory_id} | {environment_slot} | {generation_count} | "
            "{executed_ticks} | {duration:.6f} | {success} | {error} | "
            "{rank} | {advantage:.6f} |".format(
                trajectory_id=trajectory["trajectory_id"],
                environment_slot=trajectory["environment_slot"],
                generation_count=trajectory["generation_count"],
                executed_ticks=trajectory["executed_ticks"],
                duration=trajectory["wall_clock_duration_seconds"],
                success="是" if trajectory["trajectory_success"] else "否",
                error=trajectory["trajectory_error"] or "无",
                rank=comparison["rank"],
                advantage=comparison["relative_advantage"],
            )
        )
    lines.extend(
        (
            "",
            "## 产物索引",
            "",
            f"- [结构化总结果]({_relative_link(result_path, output_directory)})",
            f"- [共享起点状态]({_relative_link(trajectory_directory / 'shared-start-state.json', output_directory)})",
            f"- [共享起点图像]({_relative_link(trajectory_directory / 'shared-start.png', output_directory)})",
            f"- [轨迹运行报告]({_relative_link(trajectory_directory / 'README.md', output_directory)})",
        )
    )
    for trajectory in result["trajectories"]:
        trajectory_id = trajectory["trajectory_id"]
        lines.append(
            f"- [{trajectory_id} 轨迹报告]({_relative_link(trajectory_directory / trajectory_id / 'trajectory.md', output_directory)})"
        )
    report_path = output_directory / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_complete(
    output_directory: Path,
    *,
    baseline_world: Path | None = None,
    port_base: int = 19800,
    action_budget_ticks: int = 512,
    max_generations: int = 10,
    warmup_ticks: int = 20,
    backend: str = "codex-cli",
    model: str = "gpt-5.6-sol",
    target_log_count: int = 1,
    trajectory_count: int = 4,
) -> Path:
    """创建或复用基准存档，执行四轨迹并生成顶层 Markdown 报告。"""
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    os.environ["TEACHER_BACKEND"] = backend
    os.environ["TEACHER_MODEL"] = model
    selected_baseline = baseline_world.resolve() if baseline_world is not None else None
    baseline_created = selected_baseline is None
    if selected_baseline is None:
        baseline_directory = output_directory / "baseline"
        create_baseline_world(
            baseline_directory,
            port=port_base - 1,
            warmup_ticks=warmup_ticks,
        )
        selected_baseline = baseline_directory / "baseline-world"
    trajectory_directory = output_directory / "trajectories"
    run_four(
        trajectory_directory,
        action_budget_ticks=action_budget_ticks,
        max_generations=max_generations,
        warmup_ticks=warmup_ticks,
        backend_name=backend,
        port_base=port_base,
        use_shared_memory=False,
        baseline_world_path=selected_baseline,
        target_log_count=target_log_count,
        trajectory_count=trajectory_count,
    )
    return _write_report(
        output_directory,
        trajectory_directory=trajectory_directory,
        baseline_world=selected_baseline,
        baseline_created=baseline_created,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-world", type=Path)
    parser.add_argument("--port-base", type=int, default=19800)
    parser.add_argument("--action-budget-ticks", type=int, default=512)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--backend", default="codex-cli")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--target-log-count", type=int, default=1)
    parser.add_argument("--trajectory-count", type=int, default=4)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    if arguments.env_file is not None:
        load_env_file(arguments.env_file)
    print(
        run_complete(
            arguments.output or _default_output(),
            baseline_world=arguments.baseline_world,
            port_base=arguments.port_base,
            action_budget_ticks=arguments.action_budget_ticks,
            max_generations=arguments.max_generations,
            warmup_ticks=arguments.warmup_ticks,
            backend=arguments.backend,
            model=arguments.model,
            target_log_count=arguments.target_log_count,
            trajectory_count=arguments.trajectory_count,
        )
    )


if __name__ == "__main__":
    main()
