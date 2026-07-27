# -*- coding: utf-8 -*-
"""把 run_llm_log_collection 产出的 trajectory.json 渲染成可读的 Markdown 轨迹。

图像不复制，直接用 frames/ 下的相对路径嵌入，所以生成的 md 必须和 frames/
放在同一个目录里（也就是 --run-directory 本身）。

    python -m rl_training_environments.craftground.render_trajectory_markdown \
        runs/llm-log-collection/v4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _inventory_text(inventory: Optional[Dict[str, int]]) -> str:
    """背包字典渲染成一行。空背包显式写出来，不要留白。"""
    if not inventory:
        return "空"
    return "、".join(f"{name}×{count}" for name, count in sorted(inventory.items()))


def _inventory_delta(before: Dict[str, int], after: Dict[str, int]) -> str:
    """只列变化项。没变化就返回空串，让调用方决定是否整行省略。"""
    keys = set(before or {}) | set(after or {})
    parts: List[str] = []
    for key in sorted(keys):
        delta = (after or {}).get(key, 0) - (before or {}).get(key, 0)
        if delta:
            parts.append(f"{key} {delta:+d}")
    return "、".join(parts)


def _position_text(position: Optional[List[float]]) -> str:
    if not position:
        return "未知"
    return "({:.1f}, {:.1f}, {:.1f})".format(*position)


def _distance(before: Optional[List[float]], after: Optional[List[float]]) -> float:
    """水平位移。y 不计——跳跃和落地不是"走过去了"。"""
    if not before or not after:
        return 0.0
    return ((after[0] - before[0]) ** 2 + (after[2] - before[2]) ** 2) ** 0.5


def _crosshair_text(raycast: Optional[Dict[str, Any]]) -> str:
    """准星指向。raycast 里 crosshair 为 None 表示指着空气。"""
    if not raycast:
        return "未记录"
    crosshair = raycast.get("crosshair")
    if not crosshair:
        return "空气"
    name = crosshair.get("name", "?")
    distance = crosshair.get("distance")
    if distance is None:
        return name
    return f"{name} {distance:.1f}m"


def _stage_of_round(stages: List[Dict[str, Any]], round_index: int) -> Optional[str]:
    """把轮次归到阶段。

    新格式直接带 first_round/last_round。旧轨迹只有 `rounds_done`，那是**该阶段结束时的
    累计轮数**（同一 session 跨阶段续跑），不是阶段自身轮数，所以区间 = 相邻值之间。
    """
    previous = 0
    for stage in stages:
        first, last = stage.get("first_round"), stage.get("last_round")
        if not (isinstance(first, int) and isinstance(last, int)):
            done = stage.get("rounds_done")
            if not isinstance(done, int):
                return None
            first, last = previous + 1, done
        if first <= round_index <= last:
            return stage.get("stage")
        previous = last
    return None


def _stage_span(stage: Dict[str, Any], previous_done: int) -> Tuple[Optional[int], Optional[int]]:
    """取阶段轮次区间，新格式优先，旧格式按累计值推断。"""
    first, last = stage.get("first_round"), stage.get("last_round")
    if isinstance(first, int) and isinstance(last, int):
        return first, last
    done = stage.get("rounds_done")
    if isinstance(done, int):
        return previous_done + 1, done
    return None, None


def _render_header(data: Dict[str, Any]) -> List[str]:
    """文档头：任务、运行参数、终局结论。先给结论再给过程。"""
    rounds: List[Dict[str, Any]] = data.get("rounds", [])
    latencies = [r.get("latency_seconds") or 0.0 for r in rounds]
    guards = [r for r in rounds if r.get("tripped_guard")]
    warnings = sum(len(r.get("parse_warnings") or []) for r in rounds)
    final_inventory = rounds[-1].get("inventory_after") if rounds else {}

    lines = [
        f"# 轨迹：{data.get('model', '?')} 玩 Minecraft（seed {data.get('seed', '?')}）",
        "",
        "## 任务",
        "",
        "```",
        (data.get("task_text") or "").strip(),
        "```",
        "",
        "## 运行概览",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 模型 | `{data.get('model', '?')}` |",
        f"| 种子 / biome | {data.get('seed', '?')} / {data.get('spawn_biome', '?')} |",
        f"| 出生坐标 | {_position_text(data.get('spawn_position'))} |",
        f"| 轮数 | {len(rounds)} |",
        f"| 总 tick | {data.get('total_ticks', '?')} |",
        f"| 墙钟 | {data.get('wall_clock_seconds', 0):.0f}s |",
    ]
    if latencies:
        lines.append(
            f"| 推理延迟 均值/最大 | {sum(latencies) / len(latencies):.1f}s / "
            f"{max(latencies):.1f}s |")
    lines += [
        f"| 守卫打断轮次 | {len(guards)} / {len(rounds)} |",
        f"| 解析告警 | {warnings} |",
        f"| 终局背包 | **{_inventory_text(final_inventory)}** |",
        f"| 终止原因 | {data.get('stop_reason', '?')} |",
        "",
    ]

    stages = data.get("stage_results") or []
    if stages:
        lines += ["### 分阶段结果", "", "| 阶段 | 轮次区间 | 轮数 | 结束原因 |",
                  "|---|---|---:|---|"]
        previous = 0
        for stage in stages:
            first, last = _stage_span(stage, previous)
            if first is None or last is None:
                span, count = "?", "?"
            else:
                span, count = f"{first}–{last}", last - first + 1
                previous = last
            lines.append("| {} | {} | {} | {} |".format(
                stage.get("stage", "?"), span, count, stage.get("stop_reason", "?")))
        lines.append("")

    limits = data.get("limits") or {}
    if limits:
        lines += ["### 停止条件", "", "| 上限 | 值 |", "|---|---:|"]
        for name, value in limits.items():
            lines.append(f"| `{name}` | {value} |")
        lines.append("")
    return lines


def _render_round(round_data: Dict[str, Any], frames_directory: Path,
                  stage: Optional[str] = None) -> List[str]:
    """单轮：模型原文 → 编译结果 → 执行事实 → 画面。顺序对应模型的决策链。"""
    index = round_data.get("round_index", "?")
    planned = round_data.get("planned_ticks", 0)
    executed = round_data.get("executed_ticks", 0)
    guard = round_data.get("tripped_guard")

    title = f"## 第 {index} 轮"
    if stage:
        title += f" · {stage}"
    if guard:
        title += f"（守卫 `{guard}` 打断：{executed}/{planned} tick）"
    elif executed != planned:
        title += f"（{executed}/{planned} tick）"
    lines = [title, ""]

    why = (round_data.get("raw_model_text") or "")
    lines += ["模型原文：", "", "```", why.strip() or "（空）", "```", ""]

    canonical = (round_data.get("canonical_text") or "").strip()
    if canonical and canonical != why.strip():
        lines += ["编译后的规范形式：", "", "```", canonical, "```", ""]

    for warning in round_data.get("parse_warnings") or []:
        lines.append(f"> ⚠️ 解析告警：{warning}")
    if round_data.get("parse_warnings"):
        lines.append("")

    lines += ["| 事实 | 值 |", "|---|---|"]
    lines.append("| 执行 | {}/{} tick{} |".format(
        executed, planned, f"，守卫 `{guard}` 打断" if guard else ""))
    delta = _inventory_delta(
        round_data.get("inventory_before") or {}, round_data.get("inventory_after") or {})
    if delta:
        lines.append(f"| **背包变化** | **{delta}** |")
    lines.append(f"| 背包 | {_inventory_text(round_data.get('inventory_after'))} |")
    lines.append("| 位置 | {} → {}（位移 {:.2f}m） |".format(
        _position_text(round_data.get("position_before")),
        _position_text(round_data.get("position_after")),
        _distance(round_data.get("position_before"), round_data.get("position_after"))))
    lines.append("| 朝向 yaw | {:+.1f}° → {:+.1f}°（请求 {:+.1f}°） |".format(
        round_data.get("yaw_before") or 0.0, round_data.get("yaw_after") or 0.0,
        round_data.get("requested_yaw_delta") or 0.0))
    lines.append("| 朝向 pitch | {:+.1f}° → {:+.1f}°（请求 {:+.1f}°，正=抬头） |".format(
        round_data.get("pitch_before") or 0.0, round_data.get("pitch_after") or 0.0,
        round_data.get("requested_pitch_delta") or 0.0))
    lines.append("| 准星 | {} → {} |".format(
        _crosshair_text(round_data.get("raycast_before")),
        _crosshair_text(round_data.get("raycast_after"))))
    lines.append(f"| 推理延迟 | {round_data.get('latency_seconds') or 0.0:.1f}s |")
    usage = round_data.get("usage") or {}
    if usage:
        lines.append("| token 入/出 | {} / {} |".format(
            usage.get("input_tokens", "?"), usage.get("output_tokens", "?")))
    lines.append("")

    notes = round_data.get("truncation_notes") or []
    if notes:
        lines.append("回给模型的实测反馈：")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    paths: List[str] = list(round_data.get("observation_frame_paths") or [])
    current = round_data.get("current_frame_path")
    if current and current not in paths:
        paths.append(current)
    for path in paths:
        if not (frames_directory / path).exists():
            continue
        label = "轮末画面" if path == current else f"观察点 {path}"
        lines.append(f"![{label}](frames/{path})")
        lines.append("")
    return lines


def render_markdown(data: Dict[str, Any], frames_directory: Path) -> str:
    stages = data.get("stage_results") or []
    lines = _render_header(data)
    lines += ["---", ""]
    for round_data in data.get("rounds", []):
        stage = _stage_of_round(stages, round_data.get("round_index", 0))
        lines += _render_round(round_data, frames_directory, stage)
        lines += ["---", ""]
    lines += [
        "## 提示词",
        "",
        "每轮发给模型的提示词由 `segment_prompt_builder.py` 拼装，格式说明见",
        "[../prompt_format.md](../prompt_format.md)，逐字快照见",
        "[../prompt_snapshot.txt](../prompt_snapshot.txt)。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path,
                        help="含 trajectory.json 与 frames/ 的运行目录")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 md 路径，默认 <run_directory>/trajectory.md")
    args = parser.parse_args()

    trajectory_path = args.run_directory / "trajectory.json"
    data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    output = args.output or (args.run_directory / "trajectory.md")
    text = render_markdown(data, args.run_directory / "frames")
    output.write_text(text, encoding="utf-8")
    print(f"写入 {output}（{len(data.get('rounds', []))} 轮）")


if __name__ == "__main__":
    main()
