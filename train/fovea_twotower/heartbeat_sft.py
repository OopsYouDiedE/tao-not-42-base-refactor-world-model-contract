#!/usr/bin/env python3
"""R-C 心跳微决策 SFT:慢塔 A1 行为(冻结状态行 schema=未来一切慢塔的输入模板)。

冻结格式(VL/Omni 同款沿用,改此格式=改全栈慢塔输入):
  状态行 `t=<tick> 库存:<item×n,…|空> 可见:<cls(dist格)|无> 位移:<m>m goal:<cls>`
  微决策词表 `{继续, 换目标:<cls>, 重规划}`
GT 规则(确定性,程序生成):
  · 达成当前 goal 里程碑 → 换目标:<下一里程碑>(推进下一步);
  · 意外获得计划外物品(新里程碑非当前 goal)→ 重规划;
  · 僵局:连续 N 次心跳库存无Δ且窗口净位移<阈 → 重规划;
  · 其余 → 继续。
N 作 sweep {10,20,40} 各训一份小样,取留出准确率最高者定 N(窗口=N+4 行)。
合成轨迹为主(覆盖里程碑/换目标),混入真实 R2 rollout 记录(pose/vis/goal_log,
多为无进展→僵局)。LoRA 照 reason_delta 配方(r16 qkvo),1.5B 新 adapter。

对外接口:main(CLI)。用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/heartbeat_sft.py --sweep 10 20 40
"""
import argparse
import glob
import json
import random

import numpy as np
import torch

MS = ["木头", "木板", "木镐", "圆石", "石镐", "生铁"]   # 里程碑链(与 goal_log 中文一致)
OFFPLAN = ["泥土", "沙子", "煤炭", "种子"]             # 计划外物品(触发重规划)
VIS_CLS = ["原木", "铁矿", "煤矿", "无"]
DISP_THRESH = 3.0                                     # 窗口净位移<此(m)=卡住
DECISIONS_FIX = ["继续", "重规划"]


def state_line(t, inv, vis, disp, goal):
    inv_s = "、".join(f"{k}×{v}" for k, v in inv.items()) if inv else "空"
    vis_s = "无" if (vis is None or vis[0] == "无") else f"{vis[0]}({vis[1]}格)"
    return f"t={t} 库存:{inv_s} 可见:{vis_s} 位移:{disp:.0f}m goal:{goal}"


def gt_decision(hist, N):
    """hist=[(inv_frozen, disp, gained_item, is_goal_ms)];最新一条的 GT 决策。"""
    inv, disp, gained, is_goal = hist[-1]
    if gained is not None:
        if is_goal:
            gi = MS.index(gained)
            nxt = MS[gi + 1] if gi + 1 < len(MS) else gained
            return f"换目标:{nxt}"
        return "重规划"                                    # 计划外里程碑
    if len(hist) >= N:
        win = hist[-N:]
        inv_flat = all(w[0] == win[0][0] for w in win)
        net_disp = sum(w[1] for w in win)
        if inv_flat and net_disp < DISP_THRESH:
            return "重规划"                                # 僵局
    return "继续"


def synth_traj(rng, length=60):
    """一条合成心跳轨迹 → [(state_line_str, hist_tuple)]。"""
    inv, gi, out, hist = {}, 0, [], []
    t = 0
    stuck = 0
    for _ in range(length):
        t += rng.randint(12, 18)
        goal = MS[gi]
        gained, is_goal = None, False
        r = rng.random()
        if stuck > 0:                                      # 卡住段:低位移无收获
            disp = rng.uniform(0, 1.0)
            stuck -= 1
        elif r < 0.12:                                     # 达成当前里程碑
            gained, is_goal = MS[gi], True
            inv = dict(inv)
            inv[MS[gi]] = inv.get(MS[gi], 0) + 1
            if gi + 1 < len(MS):
                gi += 1
            disp = rng.uniform(1, 4)
        elif r < 0.18:                                     # 计划外物品
            it = rng.choice(OFFPLAN)
            gained = it
            inv = dict(inv)
            inv[it] = inv.get(it, 0) + 1
            disp = rng.uniform(1, 4)
        elif r < 0.30:                                     # 进入卡住段
            stuck = rng.randint(10, 42)
            disp = rng.uniform(0, 1.0)
        else:                                              # 正常移动
            disp = rng.uniform(2, 6)
        vis = None if rng.random() < 0.4 else (rng.choice(VIS_CLS[:3]), rng.randint(2, 12))
        hist.append((frozenset(inv.items()), disp, gained, is_goal))
        out.append((state_line(t, inv, vis, disp, MS[gi if gained and is_goal else gi]),
                    tuple(hist)))
    return out


def real_trajs(real_dir, rng, max_files=20):
    """真实 R2 rollout npz → 心跳轨迹(pose→位移,vis,goal_log;多为无进展)。"""
    trajs = []
    files = sorted(glob.glob(f"{real_dir}/*.npz"))[:max_files]
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            recs = json.loads(str(z["recs"]))
        except Exception:
            continue
        for i, r in enumerate(recs):
            if f"pose{i}" not in z.files:
                continue
            pose = z[f"pose{i}"]
            vis = z[f"vis{i}"] if f"vis{i}" in z.files else None
            gl = r.get("goal_log", [[0, MS[2]]])
            inv_steps = r.get("inv_steps", {}) or {}
            steps = int(r.get("steps", len(pose)))
            hb, out, hist = 15, [], []
            for t in range(hb, steps, hb):
                inv = {k: 1 for k, s in inv_steps.items() if s <= t}
                disp = float(np.linalg.norm(pose[t] - pose[max(0, t - hb)])) if t < len(pose) else 0.0
                goal = next((g for tt, g in reversed(gl) if tt <= t), gl[0][1])
                v = None
                if vis is not None and t < len(vis) and vis[t]:
                    v = (goal if goal in VIS_CLS else "原木", 5)
                hist.append((frozenset(inv.items()), disp, None, False))
                out.append((state_line(t, inv, v, disp, goal), tuple(hist)))
            if len(out) >= 6:
                trajs.append(out)
    return trajs


def make_samples(trajs, N, rng, window_pad=4):
    """轨迹 → (prompt 状态行窗口, GT 决策) 样本。窗口=N+pad 行。"""
    W = N + window_pad
    samples = []
    for tr in trajs:
        for j in range(len(tr)):
            hist = tr[j][1]
            dec = gt_decision(hist, N)
            lines = [tr[k][0] for k in range(max(0, j - W + 1), j + 1)]
            samples.append(("\n".join(lines), dec))
    return samples


CHAIN_CTX = "里程碑链:" + "→".join(MS) + "。达成当前 goal(库存新增该物品)即换到链上下一个。"


def build_prompt(window):
    """教师须是学生观测的函数:里程碑链进 prompt(否则"换目标:<下一个>"不可观测)。"""
    return ("你是 Minecraft 智能体的慢脑心跳决策器。" + CHAIN_CTX + "\n"
            "规则:达成当前 goal 里程碑→换目标:<下一个>;获得计划外物品→重规划;"
            "最近连续多次心跳库存无变化且几乎没位移(卡住)→重规划;否则→继续。\n"
            "下面是最近若干次心跳的状态行,对最新一行给出微决策,"
            "只输出以下之一:继续 / 重规划 / 换目标:<物品>。\n\n"
            f"{window}\n\n决策:")


def run_variant(model_name, N, n_train, steps, lr, seed, trajs_all, dev):
    """训一个 N 变体的 LoRA,返回留出决策准确率 + 格式合规率。"""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import defaultdict
    rng = random.Random(seed)
    samples = make_samples(trajs_all, N, rng)
    rng.shuffle(samples)
    groups = defaultdict(list)                              # 按决策精确串分层
    for w, d in samples:
        groups[d].append((w, d))
    hold, train = [], []
    for d, g in groups.items():
        nh = min(max(4, len(g) // 5), 12)                  # 每类留出 4~12(均衡口径)
        hold += g[:nh]
        train += g[nh:]
    rng.shuffle(hold)
    cont = [s for s in train if s[1] == "继续"]             # 继续 下采样 ≈ 其余总数,防坍塌到多数类
    rest = [s for s in train if s[1] != "继续"]
    rng.shuffle(cont)
    train = rest + cont[:max(len(rest), 60)]
    rng.shuffle(train)
    train = train[:n_train]

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(dev)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    def enc(window, dec):
        msgs = [{"role": "user", "content": build_prompt(window)},
                {"role": "assistant", "content": dec}]
        ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                      return_dict=True)["input_ids"][0]
        pre = tok.apply_chat_template([msgs[0]], tokenize=True,
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"][0]
        lab = ids.clone()
        lab[:len(pre)] = -100
        return ids, lab

    data = [enc(w, d) for w, d in train]
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr)
    model.train()
    for st in range(steps):
        ids, lab = data[rng.randrange(len(data))]
        loss = model(input_ids=ids[None].to(dev), labels=lab[None].to(dev)).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % 100 == 0:
            print(f"  [N={N}] step {st} loss {loss.item():.4f}", flush=True)
    model.eval()

    @torch.no_grad()
    def gen(window):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(window)}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
            return_dict=True)["input_ids"].to(dev)
        out = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    ok = fmt = 0
    for w, d in hold:
        o = gen(w)
        first = o.splitlines()[0].strip() if o else ""
        is_fmt = first == "继续" or first == "重规划" or first.startswith("换目标:")
        fmt += is_fmt
        ok += (first == d)
    return dict(N=N, acc=ok / len(hold), fmt=fmt / len(hold), n_hold=len(hold),
                n_train=len(data), model=model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sweep", type=int, nargs="+", default=[10, 20, 40])
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_synth", type=int, default=120)
    p.add_argument("--real_dir", default="runs/grpo_r2")
    p.add_argument("--out", default="runs/heartbeat_lora")
    p.add_argument("--out_json", default="runs/heartbeat_gates.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda"
    rng = random.Random(args.seed)

    trajs = [synth_traj(rng) for _ in range(args.n_synth)]
    if args.real_dir:
        rt = real_trajs(args.real_dir, rng)
        trajs += rt
        print(f"[hb] 合成 {args.n_synth} 轨迹 + 真实 {len(rt)} 轨迹", flush=True)

    results, best = [], None
    for N in args.sweep:
        print(f"=== N={N} sweep ===", flush=True)
        r = run_variant(args.model, N, args.n_train, args.steps, args.lr,
                        args.seed, trajs, dev)
        print(f"  [N={N}] 留出决策acc={r['acc']:.3f} 格式合规={r['fmt']:.3f} "
              f"(hold={r['n_hold']}/train={r['n_train']})", flush=True)
        if best is None or r["acc"] > best["acc"]:
            best = r
        results.append({k: r[k] for k in ("N", "acc", "fmt", "n_hold", "n_train")})

    best["model"].save_pretrained(args.out)
    gate = dict(best_N=best["N"], best_acc=round(best["acc"], 3),
                best_fmt=round(best["fmt"], 3), sweep=results,
                gate_acc_0p95=bool(best["acc"] >= 0.95),
                gate_fmt_1p0=bool(best["fmt"] >= 0.999),
                verdict="PASS" if (best["acc"] >= 0.95 and best["fmt"] >= 0.999) else "FAIL")
    json.dump(gate, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    print(f"[hb] 定 N={best['N']} acc={best['acc']:.3f} fmt={best['fmt']:.3f} → {args.out}")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
