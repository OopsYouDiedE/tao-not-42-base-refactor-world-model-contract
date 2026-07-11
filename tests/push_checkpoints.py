#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""checkpoint 持久化:上传到 HuggingFace 公开仓(云训练机是易失的,用户 2026-07-10 授权)。

对外接口:
    MAPPING — (本地文件/目录, 仓名) 对照表;仓名模式=训练集-训练方式-结构-版本-次数
    main() — CLI:python -m tests.push_checkpoints [--only <仓名子串>]

命名模式(用户定):<训练集>-<训练方式>-<结构>-<版本>-<次数>,如 vpt-bc-pixeltower-v1-run3。
版本 v1 = 当前 PixelTower 结构(90×160/S=4/11bin/20键);结构改动即升版本。
恢复:huggingface-cli download unjustify/<仓名> --local-dir <目录>。
token 经 utils.io.get_hf_token(.env 的 HF_TOKEN)。幂等,重复上传由 HF 内容寻址去重。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import get_hf_token  # noqa: E402

OWNER = "unjustify"

# (本地路径 glob, 仓名) —— 目录则上传其中 best/last/metrics 三类文件
MAPPING = [
    ("runs/checkpoints/bc_vpt/best_run1_step600.pt", "vpt-bc-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_vpt/metrics_run1.jsonl", "vpt-bc-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_vpt/best_run2_step1200.pt", "vpt-bc-pixeltower-v1-run2"),
    ("runs/checkpoints/bc_vpt/metrics_run2.jsonl", "vpt-bc-pixeltower-v1-run2"),
    # bc_vpt/best.pt 是 canonical 别名:2026-07-10 起=run5 best 的拷贝,归档在 run5 仓,
    # 不再映射到 run3 仓(避免覆盖 run3 的历史 best.pt)。
    ("runs/checkpoints/bc_vpt/last.pt", "vpt-bc-pixeltower-v1-run3"),
    ("runs/checkpoints/bc_vpt/metrics.jsonl", "vpt-bc-pixeltower-v1-run3"),
    ("runs/checkpoints/bc_vpt2/best.pt", "vpt-bc-pixeltower-v1-run4"),  # 滚动池 lr2e-4(过拟合止于 20k,存档)
    ("runs/checkpoints/bc_vpt2/last.pt", "vpt-bc-pixeltower-v1-run4"),
    ("runs/checkpoints/bc_vpt2/metrics.jsonl", "vpt-bc-pixeltower-v1-run4"),
    ("runs/checkpoints/bc_vpt3/best.pt", "vpt-bc-pixeltower-v1-run5"),  # 滚动池持续学习 lr5e-5(在训)
    ("runs/checkpoints/bc_vpt3/last.pt", "vpt-bc-pixeltower-v1-run5"),
    ("runs/checkpoints/bc_vpt3/metrics.jsonl", "vpt-bc-pixeltower-v1-run5"),
    # hindsight relabel BC(goal 通道首次真监督;训练集含事件倒推语言标签)
    ("runs/checkpoints/bc_vpt4/best.pt", "vpt-bc-hindsight-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_vpt4/last.pt", "vpt-bc-hindsight-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_vpt4/metrics.jsonl", "vpt-bc-hindsight-pixeltower-v1-run1"),
    ("runs/data/vpt_early_goal_vocab.json", "vpt-bc-hindsight-pixeltower-v1-run1"),
    # VPT 教师蒸馏第一轮(2026-07-11 L4 小池受控:方向正确,数据量主导;含 w=0 对照证据)
    ("runs/checkpoints/bc_distill1_w05/best.pt", "vpt-bcdistill-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_distill1_w05/last.pt", "vpt-bcdistill-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_distill1_w05/metrics.jsonl", "vpt-bcdistill-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_distill1_w05/acceptance.json", "vpt-bcdistill-pixeltower-v1-run1"),
    ("runs/checkpoints/bc_control_w0/best.pt", "vpt-bc-pixeltower-v1-run6"),  # w=0 对照(小池,定罪证据)
    ("runs/checkpoints/bc_control_w0/metrics.jsonl", "vpt-bc-pixeltower-v1-run6"),
    # 蒸馏第二轮(2026-07-11 5090:init-from bc_vpt4 + KL w0.5;小池 12→41 段,闭环 0/5,
    # 行为退化实锤——负结果存档,定罪证据)
    ("runs/checkpoints/bc_distill2/best.pt", "vpt-bcdistill-pixeltower-v1-run2"),
    ("runs/checkpoints/bc_distill2/last.pt", "vpt-bcdistill-pixeltower-v1-run2"),
    ("runs/checkpoints/bc_distill2/metrics.jsonl", "vpt-bcdistill-pixeltower-v1-run2"),
    # 2b=2 的续训(41 段小池发散判停,holdout 0.71→0.94,存档)
    ("runs/checkpoints/bc_distill2b/best.pt", "vpt-bcdistill-pixeltower-v1-run3"),
    ("runs/checkpoints/bc_distill2b/metrics.jsonl", "vpt-bcdistill-pixeltower-v1-run3"),
    # 3=扩张池重开(init-from bc_vpt4,22G 池上限;在训,push 时点即档)
    ("runs/checkpoints/bc_distill3/best.pt", "vpt-bcdistill-pixeltower-v1-run4"),
    ("runs/checkpoints/bc_distill3/last.pt", "vpt-bcdistill-pixeltower-v1-run4"),
    ("runs/checkpoints/bc_distill3/metrics.jsonl", "vpt-bcdistill-pixeltower-v1-run4"),
    # grpo run1(canonical 暖启动,Sonnet-low pairwise)已冻结在 HF run1 仓;
    # runs/grpo_pixel/tower.pt 是"最近一次 GRPO run"的活文件,不再映射 run1(防覆盖)。
    ("runs/grpo_pixel/metrics.jsonl", "craftground-grpo-pixeltower-v1-run1"),
    # grpo run2 = bc_vpt4(hindsight 真 goal)暖启动的 A/B 行为对照 run(2026-07-10)
    ("runs/grpo_pixel/run_bcvpt4_sonnetlow/tower.pt", "craftground-grpo-pixeltower-v1-run2"),
    ("runs/grpo_pixel/run_bcvpt4_sonnetlow/metrics_bcvpt4.jsonl", "craftground-grpo-pixeltower-v1-run2"),
    ("runs/grpo_pixel/run_bcvpt4_sonnetlow/behavior_stats.json", "craftground-grpo-pixeltower-v1-run2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只同步仓名含该子串的条目")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=get_hf_token())
    done_repos = set()
    n = 0
    for local, repo in MAPPING:
        if args.only and args.only not in repo:
            continue
        p = Path(local)
        if not p.exists():
            continue
        rid = f"{OWNER}/{repo}"
        if rid not in done_repos:
            api.create_repo(rid, private=False, exist_ok=True)
            done_repos.add(rid)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=rid)
        print(f"↑ {local} → {rid} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
        n += 1
    print(f"✅ 上传 {n} 个文件,仓:{sorted(done_repos)}")


if __name__ == "__main__":
    main()
