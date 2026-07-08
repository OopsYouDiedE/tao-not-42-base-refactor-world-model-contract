#!/usr/bin/env python3
"""J0 判官上岗考试:提示式 LLM 判官在已知优劣序轨迹对上的排序精度。

构造对(优劣序由构造保证,零标注成本):
  腐化对:真实 rollout rec → 删里程碑/清探索/清一致性 = 必然更差;
  原型对:深里程碑档 vs 浅档(档间序确定)。
证据模板(ReST"方向盲"教训:喂可分级客观证据,不喂裸轨迹)。
门(预登记):已知对排序精度 ≥0.9 → 判官热插拔为主 verdict(机器分降级为证据)。

用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/judge_exam.py \
      --npz runs/grpo_r1/g0_w0.npz runs/grpo_r1/g0_w1.npz runs/grpo_r1/g0_w3.npz
"""
import argparse
import json

import numpy as np
import torch

MS_ORDER = ["log", "planks", "wooden_pickaxe", "cobblestone", "stone_pickaxe"]


def evidence(rec):
    ms = [m for m in MS_ORDER if m in rec["inv_events"]] or ["无"]
    return (f"里程碑达成:{'/'.join(ms)};最长目标锁定 {rec['iron_lock_steps']} 步;"
            f"探索覆盖 {rec['explored_delta']} 格;宣告子目标『{rec['declared_goal']}』"
            f"且行为一致步占比 {rec['goal_consistent_steps']/max(rec['steps'],1):.2f};"
            f"总步数 {rec['steps']};{'已拿到铁' if rec.get('success') else '未拿到铁'}")


def corrupt(rec, rng):
    """三种腐化(必然更差):删里程碑/探索清零(原地打转)/一致性清零(不知在干什么)。"""
    bad = dict(rec)
    mode = rng.integers(3)
    if mode == 0:
        bad["inv_events"] = []
        bad["iron_lock_steps"] = min(rec["iron_lock_steps"], 5)
    elif mode == 1:
        bad["explored_delta"] = 1
        bad["goal_consistent_steps"] = rec["goal_consistent_steps"] // 4
    else:
        bad["goal_consistent_steps"] = 0
        bad["declared_goal"] = rec["declared_goal"]
        bad["explored_delta"] = max(1, rec["explored_delta"] // 3)
    return bad


def archetype_pairs(rng, n=8):
    ps = []
    for _ in range(n):
        d = int(rng.integers(1, len(MS_ORDER)))
        good = dict(inv_events=MS_ORDER[:d + 1], iron_lock_steps=60,
                    explored_delta=25, declared_goal="木镐",
                    goal_consistent_steps=800, steps=2000, success=False)
        bad = dict(inv_events=MS_ORDER[:max(d - 1, 0)], iron_lock_steps=20,
                   explored_delta=8, declared_goal="木镐",
                   goal_consistent_steps=300, steps=2000, success=False)
        ps.append((good, bad))
    return ps


class Judge:
    def __init__(self, base="Qwen/Qwen2.5-1.5B-Instruct", dev="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def pick(self, ra, rb):
        q = ("任务:在 Minecraft 里独立获得铁。下面是两条尝试轨迹的客观摘要,"
             "评判哪条更接近完成任务且过程质量更高(路线正确、知道自己在干什么)。\n"
             f"轨迹A:{evidence(ra)}\n轨迹B:{evidence(rb)}\n"
             "只回答一个字母:A 或 B。")
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        out = self.model.generate(enc["input_ids"].to(self.dev), max_new_tokens=8,
                                  do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        t = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip().upper()
        return "A" if "A" in t[:3] else "B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", nargs="+", required=True)
    p.add_argument("--out", default="runs/judge_exam.json")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    recs = []
    for f in args.npz:
        z = np.load(f, allow_pickle=True)
        recs.extend(json.loads(str(z["recs"])))
    pairs = [(r, corrupt(r, rng)) for r in recs] + archetype_pairs(rng)
    judge = Judge()
    ok = 0
    details = []
    for good, bad in pairs:
        if rng.random() < 0.5:
            ans = judge.pick(good, bad)
            correct = ans == "A"
        else:
            ans = judge.pick(bad, good)
            correct = ans == "B"
        ok += int(correct)
        details.append(correct)
    acc = ok / len(pairs)
    out = dict(n_pairs=len(pairs), acc=round(acc, 3),
               gate_J0=bool(acc >= 0.9),
               n_corruption=len(recs), n_archetype=8,
               judge="Qwen2.5-1.5B-Instruct 零SFT提示式")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
