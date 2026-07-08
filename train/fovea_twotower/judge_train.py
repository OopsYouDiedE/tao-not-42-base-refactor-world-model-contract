#!/usr/bin/env python3
"""J1 判官训练:构造对量产 SFT(LoRA qkvo,E2 配方——判优程序进权重,证据走上下文)。

数据(优劣序全部构造性已知,零人工):合成 rec 随机化(里程碑深度/锁定/探索/
一致性)→ 腐化对+原型档差对;对抗对=幻觉锁定(高lock零里程碑)vs 真进展
(低lock有里程碑)——机器层盲区,判官存在的意义。
门(预登记):留出已知对 ≥0.9;对抗留出 ≥0.8。反 Goodhart:标签绝不来自
机器分(那是蒸馏机器层);判官冻结版本化,不随策略在线更新。

用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/judge_train.py
"""
import argparse
import json

import numpy as np
import torch

from train.fovea_twotower.judge_exam import MS_ORDER, corrupt, evidence


def synth_rec(rng):
    d = int(rng.integers(0, len(MS_ORDER) + 1))
    return dict(inv_events=MS_ORDER[:d],
                iron_lock_steps=int(rng.integers(0, 120)),
                explored_delta=int(rng.integers(1, 40)),
                declared_goal=str(rng.choice(["木镐", "原木", "圆石", "raw_iron"])),
                goal_consistent_steps=int(rng.integers(0, 1600)),
                steps=2000, success=bool(d == len(MS_ORDER) and rng.random() < .3))


def make_pairs(rng, n):
    ps = []
    for _ in range(n):
        r = synth_rec(rng)
        kind = rng.integers(3)
        if kind == 0:                                   # 腐化对
            ps.append((r, corrupt(r, rng), "corrupt"))
        elif kind == 1:                                 # 原型档差
            d = len([m for m in MS_ORDER if m in r["inv_events"]])
            worse = dict(r, inv_events=MS_ORDER[:max(d - 1, 0)],
                         goal_consistent_steps=r["goal_consistent_steps"] // 2)
            if d == 0:
                worse = corrupt(r, rng)
            ps.append((r, worse, "arche"))
        else:                                           # 对抗:幻觉锁定 vs 真进展
            fake = dict(synth_rec(rng), inv_events=[],
                        iron_lock_steps=int(rng.integers(800, 2000)),
                        explored_delta=int(rng.integers(1, 5)))
            true = dict(synth_rec(rng), inv_events=["log"],
                        iron_lock_steps=int(rng.integers(10, 60)),
                        explored_delta=int(rng.integers(10, 40)))
            ps.append((true, fake, "adv"))
    return ps


def q_of(ra, rb):
    return ("任务:在 Minecraft 里独立获得铁。下面是两条尝试轨迹的客观摘要,"
            "评判哪条更接近完成任务且过程质量更高(路线正确、知道自己在干什么)。"
            "注意:长时间锁定同一目标但零里程碑、低探索=原地空转,不如有实际"
            "里程碑进展。\n"
            f"轨迹A:{evidence(ra)}\n轨迹B:{evidence(rb)}\n"
            "只回答一个字母:A 或 B。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1600)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--out", default="runs/judge_lora_v1")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(1)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj",
                                             "o_proj"])).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pairs = make_pairs(rng, args.n_train)
    model.train()
    for e in range(args.epochs):
        rng.shuffle(pairs)
        tot = n = 0
        for good, bad, _k in pairs:
            flip = rng.random() < 0.5
            ra, rb, ans = (good, bad, "A") if not flip else (bad, good, "B")
            msgs = [{"role": "user", "content": q_of(ra, rb)},
                    {"role": "assistant", "content": ans}]
            enc = tok.apply_chat_template(msgs, tokenize=True,
                                          return_tensors="pt").cuda()
            lab = enc.clone()
            lab[:, :-3] = -100                      # 只学答案 token
            out = model(enc, labels=lab)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += float(out.loss)
            n += 1
            if n % 400 == 0:
                print(f"[j1] e{e} {n}/{len(pairs)} loss={tot/n:.4f}", flush=True)
    model.save_pretrained(args.out)
    print(f"[j1] saved → {args.out}", flush=True)

    # 留出评估(新种子构造对+对抗对)
    model.eval()
    rng2 = np.random.default_rng(999)
    hold = make_pairs(rng2, 120)
    ok = {"corrupt": [0, 0], "arche": [0, 0], "adv": [0, 0]}
    with torch.no_grad():
        for good, bad, k in hold:
            flip = rng2.random() < 0.5
            ra, rb, ans = (good, bad, "A") if not flip else (bad, good, "B")
            enc = tok.apply_chat_template(
                [{"role": "user", "content": q_of(ra, rb)}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True)
            o = model.generate(enc["input_ids"].cuda(), max_new_tokens=4,
                               do_sample=False, pad_token_id=tok.eos_token_id)
            t = tok.decode(o[0][enc["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip().upper()
            pick = "A" if "A" in t[:3] else "B"
            ok[k][0] += int(pick == ans)
            ok[k][1] += 1
    res = {k: round(v[0] / max(v[1], 1), 3) for k, v in ok.items()}
    known = (ok["corrupt"][0] + ok["arche"][0]) / max(
        ok["corrupt"][1] + ok["arche"][1], 1)
    out = dict(holdout=res, known_acc=round(known, 3),
               gates={"J1-known(>=0.9)": bool(known >= 0.9),
                      "J1-adv(>=0.8)": bool(res["adv"] >= 0.8)})
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("runs/judge_v1_gates.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
