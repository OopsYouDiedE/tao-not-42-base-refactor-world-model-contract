"""生成正式项目动作协议的 CraftGround 闭环报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate(run: Path, output: Path) -> None:
    evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    eval_by_id = {item["trajectory_id"]: item for item in evaluation["trajectories"]}
    trajectories = [
        json.loads((run / trajectory_id / "trajectory.json").read_text(encoding="utf-8"))
        for trajectory_id in ("T1", "T2", "T3", "T4")
    ]
    lines = [
        "# Terra × CraftGround 正式动作协议闭环实验",
        "",
        "## 结论",
        "",
        "本报告只统计项目正式 Lumine 命名 token 协议产生的轨迹。旧的 CraftGround V2 JSON",
        "实验已经标记为无效协议实验。T4 在第 9 次模型指令、tick 25 打开 Chest GUI。",
        "",
        "## 协议",
        "",
        "```text",
        "[\"<|action_start|> ; Mouse -35 0 W ; W ; MouseRight <|action_end|>\"]",
        "Reason: visual evidence and duration choice",
        "```",
        "",
        "每个分号是一个 50 ms tick。服务端调用 `decode_lumine_action()`，逐 chunk 转换成内部",
        "CraftGround V2 动作，并为每个 tick 保存一张 RGB。",
        "",
        "## 汇总",
        "",
        "| 轨迹 | 成功 | 模型指令 | tick | 模拟秒 | CraftGround墙钟 | 分数 | 相对优势 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trajectory in trajectories:
        item = eval_by_id[trajectory["trajectory_id"]]
        lines.append(
            f"| {trajectory['trajectory_id']} | {'是' if item['success'] else '否'} | "
            f"{trajectory['turn']} | {trajectory['tick']} | {trajectory['tick']/20:.2f} | "
            f"{trajectory['simulation_wall_ms']:.2f} ms | {item['score']} | {item['adv_i']:+.2f} |"
        )
    lines.extend([
        "",
        "## Prompt",
        "",
        "正式 Prompt 原文：[`policy_prompt.md`](../runs/craftground-lumine-terra-batch4/policy_prompt.md)。",
        "代码来源：`datasets/minestudio_finetune/sft_protocol.py` 的 `history_to_future_action`。",
        "",
        "## 共同起点",
        "",
        "![initial](../runs/craftground-lumine-terra-batch4/initial.png)",
        "",
    ])
    for trajectory in trajectories:
        trajectory_id = trajectory["trajectory_id"]
        lines.extend([f"## {trajectory_id} 动作与逐 tick 图像", ""])
        lines.extend([
            "| 轮次 | tick范围 | 正式动作块 | CraftGround执行 |",
            "|---:|---|---|---:|",
        ])
        for record in trajectory["records"]:
            action = record["action_text"].replace("|", "\\|")
            lines.append(
                f"| {record['turn']} | {record['start_tick']}–{record['end_tick']} | `{action}` | "
                f"{record['simulation_wall_ms']:.2f} ms |"
            )
        lines.extend([
            "",
            f"完整 Prompt、原始模型输出和解析后 chunks："
            f"[`{trajectory_id}/trajectory.json`](../runs/craftground-lumine-terra-batch4/{trajectory_id}/trajectory.json)。",
            "",
        ])
        for frame in sorted((run / trajectory_id).glob("frame_*.png")):
            relative = f"../runs/craftground-lumine-terra-batch4/{trajectory_id}/{frame.name}"
            lines.extend([f"### {trajectory_id} {frame.stem}", "", f"![{trajectory_id} {frame.stem}]({relative})", ""])
    lines.extend([
        "## 独立评估",
        "",
        f"Batch 平均分：`{evaluation['batch_summary']['batch_mean_score']}`。",
        "",
    ])
    for item in evaluation["ranking"]:
        lines.append(
            f"- 第 {item['rank']} 名：{item['trajectory_id']}，分数 {item['score']}，"
            f"优势 {item['adv_i']:+.2f}。"
        )
    lines.extend([
        "",
        "评估原文：[`evaluation.json`](../runs/craftground-lumine-terra-batch4/evaluation.json)。",
        "",
        "## 控制代码",
        "",
        "- 动作协议：[`datasets/action_codec.py`](../datasets/action_codec.py)",
        "- SFT Prompt：[`datasets/minestudio_finetune/sft_protocol.py`](../datasets/minestudio_finetune/sft_protocol.py)",
        "- CraftGround 适配服务：[`tools/craftground_closed_loop_server.py`](../tools/craftground_closed_loop_server.py)",
        "- 本报告生成器：[`tools/generate_lumine_closed_loop_report.py`](../tools/generate_lumine_closed_loop_report.py)",
        "",
        "## 下次运行",
        "",
        "```bash",
        "PYTHONPATH=. .venv/bin/python tools/craftground_closed_loop_server.py \\",
        "  --runtime /tmp/tao-craftground-reset-runtime \\",
        "  --output runs/craftground-lumine-terra-batch4 \\",
        "  --port 18400 --max-ticks 400 --max-turns 10",
        "```",
        "",
        "服务接收项目动作文本：",
        "",
        "```json",
        "{",
        "  \"action_text\": \"<|action_start|> ; W ; Mouse -20 10 W ; MouseRight <|action_end|>\",",
        "  \"model\": {\"model\": \"gpt-5.6-terra\", \"prompt_kind\": \"project_history_to_future_action\"}",
        "}",
        "```",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
