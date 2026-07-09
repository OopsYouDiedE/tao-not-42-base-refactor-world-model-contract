#!/usr/bin/env python3
"""GRPO-R2 判官 I/O 契约体检(真 Haiku,不需要任何训练权重)。

**测的不是判官质量**(那个已用真考卷验过:Haiku 20/20 vs Qwen1.5B 0.6,commit 8555568),
而是**输出格式契约**——因为 `grpo_r2._parse_ranks()` 极其严格:

    got = {int(m[1]): float(m[2]) for m in re.finditer(r"第\\s*(\\d+)\\s*条\\s*[:：]\\s*名次\\s*([\\d.]+)", out)}
    return got if len(got) == k and set(got) == set(range(k)) else None

要求 Haiku 恰好吐出 **0 起始**的 `第0条..第{k-1}条`。而 `_chunk_prompt` 也确实以
`### 第{j}条`(j 从 0)标注。**LLM 天然倾向 1-based 编号** ⇒ 一旦漂移,`_parse_ranks`
返回 None,两轮重试后整块**静默退化为 `fallback_score`**(只数背包事件),
而 `fallback_chunks` 只是计数器,不会让 run 失败。

⇒ 真跑 GRPO 前先跑这个。`fallback_rate` 高就说明判官信号根本没进优势,
   你以为在跑判官驱动的 GRPO,实际在跑里程碑计数驱动的 GRPO。

用法:
    PYTHONPATH=. python tests/probe_judge_io_haiku.py --trials 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# grpo_r2 顶层 import grpo_r1(会拖 craftground/YOLOE)。stub 掉,只借它的 RUBRIC/解析器。
_stub = types.ModuleType("train.fovea_twotower.grpo_r1")
_stub.ENV = None
_stub.update = None
sys.modules.setdefault("train.fovea_twotower.grpo_r1", _stub)

from train.fovea_twotower import grpo_r2  # noqa: E402


def fake_rollout(frames: list[np.ndarray], quality: int) -> dict:
    """按 RUBRIC 的阶梯造 4 档质量。仅用于格式体检,不做判官准确性断言。"""
    n = 400
    keys = np.zeros((n, 8), np.int8)
    vis = np.zeros(n, bool)
    pose = np.zeros((n, 3), np.float32)
    rec = {"steps": n, "explored_delta": 0, "goal_consistent_steps": 0,
           "declared_goal": "找树", "inv_events": set(), "inv_steps": {},
           "success": False}
    if quality >= 1:                       # 有动作但无方向
        keys[:, grpo_r2.FWD] = (np.arange(n) % 3 == 0)
        pose[:, 0] = np.sin(np.arange(n) / 20.0)
        rec["explored_delta"] = 3
    if quality >= 2:                       # 有目标性:持续接近并攻击
        keys[:, grpo_r2.FWD] = 1
        keys[150:, grpo_r2.ATK] = 1
        vis[100:] = True
        pose[:, 0] = np.linspace(0, 30, n)
        rec["explored_delta"] = 21
        rec["goal_consistent_steps"] = 300
    if quality >= 3:                       # 拿到原木
        rec["inv_events"] = {"log"}
        rec["inv_steps"] = {"log": 232}
        rec["goal_log"] = [[0, "找树"], [232, "做木板"]]
    return {"frames": frames, "rec": rec, "keys": keys, "vis": vis, "pose": pose}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--frames-dir", default="/workspace/assets")
    ap.add_argument("--out", default="runs/grpo_r2_iotest", help="必须在仓库/工作区内")
    ap.add_argument("--allowed-tools", default="", help='如 "Read";空=复现 grpo_r2 现状')
    args = ap.parse_args()

    pool = sorted(Path(args.frames_dir).glob("mc_*.png"))
    if not pool:
        sys.exit(f"no frames in {args.frames_dir} (需要几张 640x360 的 MC 截图)")
    imgs = [np.asarray(Image.open(p).convert("RGB").resize((320, 180)), dtype=np.uint8)
            for p in pool]
    frames = [imgs[i % len(imgs)] for i in range(8)]

    # 必须用**真实 OUT 路径**(仓库内 runs/grpo_r2):/tmp 在工作区外,会被权限层单独拦掉,
    # 那样测出来的失败是测试自己造的,不是 grpo_r2 的。
    tmp = args.out
    Path(tmp).mkdir(parents=True, exist_ok=True)
    grpo_r2.OUT = tmp
    rolls = [fake_rollout(frames, q) for q in (0, 1, 2, 3)]

    ok = 0
    for t in range(args.trials):
        prompt = grpo_r2._chunk_prompt(t, 0, list(range(4)), rolls)
        cmd = ["claude", "-p", "--model", "haiku"]
        if args.allowed_tools:
            cmd += ["--allowedTools", args.allowed_tools]
        cmd.append(prompt)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=480)
        out = p.stdout
        ranks = grpo_r2._parse_ranks(out, 4)
        status = "PARSED" if ranks else "FALLBACK"
        ok += bool(ranks)
        print(f"trial {t}: {status}  ranks={ranks}")
        if not ranks:
            print("    raw reply (前 200 字):", repr(out.strip()[:200]))

    rate = ok / args.trials
    print(f"\nparse_ok_rate = {rate:.2f}  ({ok}/{args.trials})")
    print(f"fallback_rate = {1 - rate:.2f}  <- 非零就意味着判官信号在这些块上没进优势")
    if rate < 1.0:
        print("\n可能的修法:")
        print("  a) RUBRIC 改用 1-based 编号,_chunk_prompt 与 _parse_ranks 同步 +1;")
        print("  b) _parse_ranks 放宽:接受 1-based 并归一化;")
        print("  c) 让判官回 JSON,别回中文模板。")


if __name__ == "__main__":
    main()
