#!/usr/bin/env python3
"""GRPO-R2:判官打分版驱动器(用户 07-08 定案)。

相对优势不再用程序统计——需要更稠密的奖励且防刷分:每组 rollout 的
证据(图8帧联络表+行为文本)交外部强 LLM(Haiku)打 0-10 过程分,组内
z 归一化当优势。锚点=先拿到木头(log),0=瘫痪…7=原木…10=铁。
机器统计降级为喂判官的证据;可见性指标标注"可刷,仅参考"。
判官两次解析失败→回退里程碑机器分(只算背包事件,不含可刷项)并记 flag。
"""
import argparse
import json
import os
import re
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from train.fovea_twotower.grpo_harness import group_advantage
from train.fovea_twotower.grpo_r1 import ENV, update

OUT = "runs/grpo_r2"
FWD, ATK = 0, 7                                       # V2_KEYS 索引

RUBRIC = """任务背景:智能体在 Minecraft 生存模式里的长程任务是独立获得铁。当前训练锚点:先拿到木头(原木)。
下面每条是一次尝试的证据:一张 8 帧时间均匀抽样的联络表图(按先后从左到右、上到下)+行为统计文本。
给每条打 0-10 过程分(可用一位小数),判据是向"拿到木头→镐→铁"推进的真实进度与意图质量:
- 0 = 全程瘫痪(不移动不按键)
- 1~3 = 有动作但无方向(原地打转、乱跳、无目标游走)
- 4~6 = 有目标性:持续朝树木接近、对树攻击、路线明确
- 7 = 拿到原木;8 = 进一步合成木板;9 = 木镐/圆石/石镐;10 = 拿到铁
防刷分警告:文本里的"目标可见步数"可以靠原地乱转刷高,不可单独作为进度证据;必须结合图中场景与移动/攻击行为判断。图文矛盾时以图为准。
同水平允许同分,不要为拉开差距而编造。
先用 Read 工具逐张读取下面列出的联络表图,再逐条作答。
输出格式严格为每行一条『第N条: X』(X 为 0-10 数字),不输出其他内容。"""


def contact_sheet(frames, path):
    if len(frames) == 0:
        Image.new("RGB", (640, 180), (0, 0, 0)).save(path)
        return
    h, w = frames[0].shape[:2]
    rows, cols = 2, 4
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, f in enumerate(frames[:8]):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    Image.fromarray(sheet).save(path)


def evidence_text(r):
    rec, keys, vis, pose = r["rec"], r["keys"], r["vis"], r["pose"]
    ms = "、".join(f"{k}(第{v}步)" for k, v in
                  sorted(rec.get("inv_steps", {}).items(), key=lambda x: x[1])) or "无"
    disp = float(np.abs(np.diff(pose[:, [0, 2]], axis=0)).sum()) if len(pose) > 1 else 0.0
    n = max(len(keys), 1)
    coupled = float((vis & (keys[:, FWD] | keys[:, ATK])).mean()) if len(keys) else 0.0
    return (f"里程碑:{ms};总步数 {rec['steps']};水平位移 {disp:.0f} 格;"
            f"探索覆盖 {rec['explored_delta']} 格(4m 网格);"
            f"前进键占比 {float(keys[:, FWD].mean()) if len(keys) else 0:.2f};"
            f"攻击键占比 {float(keys[:, ATK].mean()) if len(keys) else 0:.2f};"
            f"目标可见且同时前进/攻击占比 {coupled:.2f};"
            f"纯目标可见步数 {rec['goal_consistent_steps']}(可刷指标,仅参考);"
            f"宣告子目标『{rec['declared_goal']}』" +
            (";已拿到铁" if rec.get("success") else ""))


def fallback_score(rec):
    """回退:只算背包事件(不可刷),不含可见性项。"""
    return len(rec["inv_events"]) + (4.0 if rec.get("success") else 0.0)


def judge_group(g, rolls):
    lines = []
    for i, r in enumerate(rolls):
        img = os.path.abspath(f"{OUT}/g{g}_r{i}.png")
        contact_sheet(r["frames"], img)
        lines.append(f"### 第{i}条\n联络表图:{img}\n{evidence_text(r)}")
    prompt = RUBRIC + "\n\n" + "\n".join(lines)
    open(f"{OUT}/g{g}_judge_prompt.txt", "w").write(prompt)
    for att in range(2):
        try:
            out = subprocess.run(
                ["claude", "-p", "--model", "haiku"], input=prompt,
                capture_output=True, text=True, timeout=600).stdout
        except subprocess.TimeoutExpired:
            continue
        open(f"{OUT}/g{g}_judge_reply.txt", "w").write(out)
        got = {int(m.group(1)): float(m.group(2)) for m in
               re.finditer(r"第\s*(\d+)\s*条\s*[:：]\s*([\d.]+)", out)}
        if len(got) == len(rolls):
            return [got[i] for i in range(len(rolls))], False
        print(f"[g{g}] 判官解析 {len(got)}/{len(rolls)},重试 att{att}", flush=True)
    print(f"[g{g}] 判官两次失败→回退里程碑机器分", flush=True)
    return [fallback_score(r["rec"]) for r in rolls], True


def run_group(g, seed_rng, ckpt, args):
    wseed = str(int(seed_rng.integers(1, 2 ** 31)))
    outs, procs = [], []
    for w in range(4):
        out = f"{OUT}/g{g}_w{w}.npz"
        outs.append(out)
        cmd = [".venv/bin/python", "-u",
               "train/fovea_twotower/grpo_rollout_worker.py",
               "--world_seed", wseed, "--episodes", "4",
               "--max_steps", str(args.max_steps), "--ckpt", ckpt,
               "--seed", str(g * 10 + w), "--temp", str(args.temp),
               "--port", str(args.port0 + (g % 4) * 4 + w), "--out", out]
        procs.append(subprocess.Popen(
            cmd, env=ENV, stdout=open(f"{OUT}/g{g}_w{w}.log", "w"),
            stderr=subprocess.STDOUT))
    t0 = time.time()
    while time.time() - t0 < args.group_timeout:
        if all(p.poll() is not None for p in procs):
            break
        time.sleep(10)
    for p in procs:
        if p.poll() is None:
            p.kill()
    rolls = []
    for o in outs:
        try:
            z = np.load(o, allow_pickle=True)
        except FileNotFoundError:
            continue
        recs = json.loads(str(z["recs"]))
        for i, rec in enumerate(recs):
            rec["inv_events"] = set(rec["inv_events"])
            rolls.append(dict(rec=rec, toks=z[f"toks{i}"], cam=z[f"cam{i}"],
                              keys=z[f"keys{i}"], vis=z[f"vis{i}"],
                              pose=z[f"pose{i}"], frames=z[f"frames{i}"]))
    return wseed, rolls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=12)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--temp", type=float, default=1.3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--group_timeout", type=int, default=1800)
    p.add_argument("--port0", type=int, default=8900)
    p.add_argument("--init_ckpt", default="runs/trackcmd_bc_v17/best.pt")
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from train.fovea_twotower.eval_track_cmd import StudentPolicy
    student = StudentPolicy(args.init_ckpt)
    student.tower.train()
    opt = torch.optim.AdamW(student.tower.parameters(), lr=args.lr)
    ck0 = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    seed_rng = np.random.default_rng(43)
    ckpt = args.init_ckpt
    for g in range(args.groups):
        t0 = time.time()
        wseed, rolls = run_group(g, seed_rng, ckpt, args)
        if not rolls:
            print(f"[g{g}] 无 rollout 产出,跳过", flush=True)
            continue
        scores, fb = judge_group(g, rolls)
        adv = group_advantage(scores)
        loss = update(student, opt, rolls, adv)
        ckpt = f"{OUT}/student.pt"
        torch.save(dict(tower=student.tower.state_dict(), cfg=ck0["cfg"],
                        cam_acc=0.0, args=dict(ck0["args"], grpo_r2_group=g)), ckpt)
        wood = float(np.mean([1.0 if "log" in r["rec"]["inv_events"] else 0.0
                              for r in rolls]))
        m = dict(group=g, world_seed=wseed, n=len(rolls),
                 judge_scores=[round(float(s), 2) for s in scores],
                 judge_fallback=fb,
                 score_var=round(float(np.var(scores)), 4),
                 wood_rate=round(wood, 3),
                 milestones={k: sum(1 for r in rolls if k in r["rec"]["inv_events"])
                             for k in ["log", "planks", "wooden_pickaxe",
                                       "cobblestone", "stone_pickaxe"]},
                 loss=round(loss, 4), wall_s=round(time.time() - t0, 0))
        with open(f"{OUT}/metrics.jsonl", "a") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[g{g}] {json.dumps(m, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
