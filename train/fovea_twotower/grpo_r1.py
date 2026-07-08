#!/usr/bin/env python3
"""GRPO-R1 驱动器:组循环 rollout(4 worker×4局同seed)→三层过程优势→更新→下一组。

指标逐组落 runs/grpo_r1/metrics.jsonl(组内方差/里程碑深度分布/意图一致均值)。
更新=优势加权 REINFORCE(cam CE+keys BCE,64 步窗,macro 步已剔除),lr 1e-5,
梯度裁 1.0;每组存 ckpt(下组 worker 热加载)。
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from train.fovea_twotower.grpo_harness import (group_advantage, launch_gate,
                                               score_rollout)

ENV = dict(os.environ, DISPLAY=":99", LIBGL_ALWAYS_SOFTWARE="1",
           CRAFTGROUND_JVM_MAX_MEMORY="2G",
           PYTHONPATH="/home/ame/tao-not-42-base-refactor-world-model-contract")


def run_group(g, seed_rng, ckpt, args):
    wseed = str(int(seed_rng.integers(1, 2 ** 31)))
    outs, procs = [], []
    for w in range(4):
        out = f"runs/grpo_r1/g{g}_w{w}.npz"
        outs.append(out)
        cmd = [".venv/bin/python", "-u",
               "train/fovea_twotower/grpo_rollout_worker.py",
               "--world_seed", wseed, "--episodes", "4",
               "--max_steps", str(args.max_steps), "--ckpt", ckpt,
               "--seed", str(g * 10 + w), "--temp", str(args.temp),
               "--port", str(args.port0 + g * 4 + w), "--out", out]
        procs.append(subprocess.Popen(
            cmd, env=ENV, stdout=open(f"runs/grpo_r1/g{g}_w{w}.log", "w"),
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
                              keys=z[f"keys{i}"]))
    return wseed, rolls


def update(student, opt, rolls, adv, seq=64):
    dev = "cuda"
    tot = n = 0
    for r, a_w in zip(rolls, adv):
        if abs(float(a_w)) < 1e-6 or len(r["toks"]) < seq:
            continue
        T = len(r["toks"])
        prev = np.zeros((T, 22), np.float32)
        prev[1:, 0] = 0.0                                   # 简化:prev 由动作重建
        cam_t = torch.from_numpy(r["cam"]).to(dev)
        key_t = torch.from_numpy(r["keys"].astype(np.float32)).to(dev)
        for i0 in range(0, T - seq, seq):
            tk = torch.from_numpy(r["toks"][i0:i0 + seq])[None].float().to(dev)
            pv = torch.from_numpy(prev[i0:i0 + seq])[None].float().to(dev)
            g = torch.zeros(1, 1, device=dev)
            cam, key = student.tower(tk, g, pv)
            ce = torch.nn.functional.cross_entropy(
                cam[0].reshape(-1, cam.shape[-1]),
                cam_t[i0:i0 + seq].reshape(-1), reduction="mean")
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                key[0].float(), key_t[i0:i0 + seq], reduction="mean")
            loss = float(a_w) * (ce + bce)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.tower.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            n += 1
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--temp", type=float, default=1.3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--group_timeout", type=int, default=1800)
    p.add_argument("--port0", type=int, default=8900)
    p.add_argument("--init_ckpt", default="runs/trackcmd_bc_v17/best.pt")
    args = p.parse_args()
    os.makedirs("runs/grpo_r1", exist_ok=True)
    from train.fovea_twotower.eval_track_cmd import StudentPolicy
    student = StudentPolicy(args.init_ckpt)
    student.tower.train()
    opt = torch.optim.AdamW(student.tower.parameters(), lr=args.lr)
    ck0 = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    seed_rng = np.random.default_rng(42)
    ckpt = args.init_ckpt
    for g in range(args.groups):
        t0 = time.time()
        wseed, rolls = run_group(g, seed_rng, ckpt, args)
        if not rolls:
            print(f"[g{g}] 无 rollout 产出,跳过", flush=True)
            continue
        scores = [score_rollout(r["rec"]) for r in rolls]
        adv = group_advantage(scores)
        loss = update(student, opt, rolls, adv)
        ckpt = "runs/grpo_r1/student.pt"
        torch.save(dict(tower=student.tower.state_dict(), cfg=ck0["cfg"],
                        cam_acc=0.0, args=dict(ck0["args"], grpo_group=g)), ckpt)
        depths = [len(r["rec"]["inv_events"]) +
                  (1 if r["rec"]["iron_lock_steps"] >= 30 else 0) for r in rolls]
        m = dict(group=g, world_seed=wseed, n=len(rolls),
                 scores=[round(s, 3) for s in scores],
                 score_var=round(float(np.var(scores)), 4),
                 depth_hist={str(d): depths.count(d) for d in sorted(set(depths))},
                 consist_mean=round(float(np.mean(
                     [r["rec"]["goal_consistent_steps"] / max(r["rec"]["steps"], 1)
                      for r in rolls])), 3),
                 explored_mean=round(float(np.mean(
                     [r["rec"]["explored_delta"] for r in rolls])), 1),
                 gate=launch_gate(scores), loss=round(loss, 4),
                 wall_s=round(time.time() - t0, 0))
        with open("runs/grpo_r1/metrics.jsonl", "a") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[g{g}] {json.dumps(m, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
