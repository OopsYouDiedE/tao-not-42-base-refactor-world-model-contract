#!/usr/bin/env python3
"""J0-v2 考卷生成:真实动作差距对(BC temp0.2 vs 近随机 temp6.0,同世界种子)
+ 沿用 J0 的腐化/原型对(同 rng 种子,保证与 Qwen1.5B 的 0.6 可比)。

产出 prompt 文本交给外部强判官(Haiku)一次性作答,答案落 runs/judge_exam_v2.json。
"""
import argparse
import json

import numpy as np

from train.fovea_twotower.judge_exam import MS_ORDER, archetype_pairs, corrupt, evidence


def load_recs(f):
    z = np.load(f, allow_pickle=True)
    return json.loads(str(z["recs"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc", default="runs/grpo_r1/contrast_bc.npz")
    p.add_argument("--rand", default="runs/grpo_r1/contrast_rand.npz")
    p.add_argument("--g0", nargs="+", default=["runs/grpo_r1/g0_w0.npz",
                                               "runs/grpo_r1/g0_w1.npz",
                                               "runs/grpo_r1/g0_w3.npz"])
    p.add_argument("--out", default="runs/judge_exam_v2_pairs.json")
    p.add_argument("--prompt_out", default="runs/judge_exam_v2_prompt.txt")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    pairs = []  # (good, bad, category)
    for gb in load_recs(args.bc):
        for gr in load_recs(args.rand):
            pairs.append((gb, gr, "real_gap"))
    for r in [x for f in args.g0 for x in load_recs(f)]:
        pairs.append((r, corrupt(r, rng), "corruption"))
    for g, b in archetype_pairs(rng):
        pairs.append((g, b, "archetype"))

    items, lines = [], []
    for i, (good, bad, cat) in enumerate(pairs):
        flip = bool(rng.integers(2))
        a, b = (bad, good) if flip else (good, bad)
        items.append(dict(idx=i, category=cat, answer="B" if flip else "A"))
        lines.append(f"### 第{i}对\n轨迹A:{evidence(a)}\n轨迹B:{evidence(b)}")

    prompt = ("任务背景:智能体在 Minecraft 里独立获得铁(长程任务)。下面每对给出两条尝试"
              "轨迹的客观摘要。对每一对,评判哪条更接近完成任务且过程质量更高——判据:"
              "长程动作意图是否明确(里程碑推进、探索而非原地打转)、行为是否符合其宣告的"
              "子目标。逐对作答,输出格式严格为每行一条『第N对: A』或『第N对: B』,"
              "不要输出其他内容。\n\n" + "\n".join(lines))

    json.dump(items, open(args.out, "w"), ensure_ascii=False, indent=1)
    open(args.prompt_out, "w").write(prompt)
    from collections import Counter
    print(f"pairs={len(items)} by_cat={dict(Counter(x['category'] for x in items))}")
    print(f"prompt_chars={len(prompt)} -> {args.prompt_out}")


if __name__ == "__main__":
    main()
