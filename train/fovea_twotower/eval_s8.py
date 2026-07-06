#!/usr/bin/env python3
"""S8a/S8b —— 轨迹质量的"能力位"与"口径位"冻结塔探针(Step5,判据先于结果登记)。

设计见 docs/architectures/fovea-twotower-step5.md §2:
  S8a 能力位:同起点/近邻配对成对轨迹,冻结塔 STATE + Bradley-Terry 线性判优 vs 真值。
             过 = 判优 acc CI 下界 > 0.55。败 → 质量信号不在塔内,裁判降级为显式计分器。
  S8b 口径位(存亡命题,§0.6):对抗配对(强策略×难起点 vs 弱策略×易起点),
             return 口径 vs regret 口径各测。过 = regret 口径 acc CI 下界 > 0.5
             (预期 return 口径塌 < 0.5,一次实验同证"可以但须换口径")。

两部分解耦:
  ① BT 探针核心(数据无关):reps[N,D] + value[N] + 分组/难度元数据 → 判优 acc + bootstrap CI。
     `--smoke` 走植入信号的合成集,端到端自检(不需塔/真数据)。
  ② 塔编码前端(需真轨迹数据):逐条轨迹过冻结 W4 塔 → pool_ssm(21 层 ssm_state)得 STATE。

────────────────────────────────────────────────────────────────────────
轨迹数据契约(`--traj-dir` 下每条一个 .npz,由 CraftGround rollout 采集器产出,待建):
  lat   float32 [T, ...]   逐帧视觉 latent(DINO,与训练流同格式)
  act   float32 [T, ...]   逐帧动作(流内 act token)
  msg   float32 [T, ...]   逐帧消息(流内 msg token)
  score float32 标量        轨迹价值 = 视界 H 内成就/进度计分(CraftGround obs["full"] 计)
  start_id  int            起点态簇 id(S8a 同起点配对用)
  policy_strong int {0,1}  采样策略强弱(S8b 对抗配对用)
  start_hard    int {0,1}  起点难度(S8b:regret = score − 该难度基线)
价值标签来自 CraftGround 成就计分,不来自 VPT/g500(那是快头 BC 数据)。
────────────────────────────────────────────────────────────────────────

用法:
  # 合成自检(无需 GPU/数据):
  python -m train.fovea_twotower.eval_s8 --smoke --out runs/ftt_s8_smoke.json
  # 真数据(塔 ckpt + 采集好的轨迹):
  python -m train.fovea_twotower.eval_s8 --ckpt runs/ftt_w4c/ckpt.pt \
      --traj-dir runs/data/s8_traj --out runs/ftt_s8.json
"""
import argparse
import glob
import json
import os

import numpy as np


# ───────────────────────── BT 探针核心(数据无关) ─────────────────────────
def fit_bt(X, y, l2=1.0, iters=300, lr=0.5):
    """L2 正则逻辑回归(Bradley-Terry:X=rep_a−rep_b,y=1 若 a 优)。返回权重 w。
    无截距(纯 BT:偏好只由分差 w·(rep_a−rep_b) 定)。梯度下降,标准化步长。"""
    n, d = X.shape
    w = np.zeros(d)
    Xs = X / (np.linalg.norm(X, axis=1, keepdims=True).mean() + 1e-8)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w)))
        grad = Xs.T @ (p - y) / n + l2 * w / n
        w -= lr * grad
    return w


def pair_acc(w, Xte, yte):
    """判优准确率:sign(w·(rep_a−rep_b)) 是否与 y(a 是否真优)一致。"""
    s = Xte @ w
    pred = (s > 0).astype(float)
    return float((pred == yte).mean())


def boot_acc_ci(w, Xte, yte, boot=1000, seed=0):
    """对测试配对做 bootstrap,返回 (acc, ci_lo, ci_hi)(95%)。"""
    rng = np.random.default_rng(seed)
    n = len(yte)
    accs = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        accs.append(pair_acc(w, Xte[idx], yte[idx]))
    accs = np.sort(accs)
    return pair_acc(w, Xte, yte), float(accs[int(0.025 * boot)]), float(accs[int(0.975 * boot)])


def make_pairs(reps, vals, idx, margin, rng, max_pairs=20000):
    """在给定轨迹子集 idx 内,对所有满足 |Δval|>margin 的配对建 (feat=rep_a−rep_b, y)。
    随机翻转正负序使标签 {0,1} 均衡。"""
    feats, ys = [], []
    m = len(idx)
    all_pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
    rng.shuffle(all_pairs)
    for i, j in all_pairs:
        if abs(vals[i] - vals[j]) <= margin:
            continue
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        feats.append(reps[a] - reps[b])
        ys.append(1.0 if vals[a] > vals[b] else 0.0)
        if len(ys) >= max_pairs:
            break
    if not ys:
        return np.zeros((0, reps.shape[1])), np.zeros(0)
    return np.stack(feats), np.array(ys)


# ───────────────────────── S8a 能力位 ─────────────────────────
def run_s8a(reps, score, start_id, margin, seed=0):
    """同起点配对 BT 判优;按 start_id 划 train/test(起点不重叠防泄漏)。"""
    rng = np.random.default_rng(seed)
    starts = np.unique(start_id)
    rng.shuffle(starts)
    n_te = max(1, len(starts) // 3)
    te_starts = set(starts[:n_te].tolist())
    tr_idx, te_idx = {}, {}
    for s in starts:
        members = np.nonzero(start_id == s)[0]
        (te_idx if s in te_starts else tr_idx)[s] = members
    # 同起点内配对(train 各起点桶内、test 各起点桶内)
    Xtr, ytr = [], []
    for s, members in tr_idx.items():
        f, y = make_pairs(reps, score, list(members), margin, rng)
        if len(y):
            Xtr.append(f); ytr.append(y)
    Xte, yte = [], []
    for s, members in te_idx.items():
        f, y = make_pairs(reps, score, list(members), margin, rng)
        if len(y):
            Xte.append(f); yte.append(y)
    if not Xtr or not Xte:
        return {"error": "同起点配对不足(检查 start_id 分组与 margin)"}
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    Xte, yte = np.concatenate(Xte), np.concatenate(yte)
    w = fit_bt(Xtr, ytr)
    acc, lo, hi = boot_acc_ci(w, Xte, yte, seed=seed)
    verdict = "PASS" if lo > 0.55 else "FAIL"
    return {"n_pairs_tr": int(len(ytr)), "n_pairs_te": int(len(yte)),
            "acc": round(acc, 4), "ci": [round(lo, 4), round(hi, 4)],
            "verdict_s8a": f"{verdict} (acc={acc:.4f}, ci_lo={lo:.4f}, 门 0.55)"}


# ───────────────────────── S8b 口径位(对抗配对) ─────────────────────────
def run_s8b(reps, score, policy_strong, start_hard, seed=0):
    """对抗配对:a=强策略×难起点 vs b=弱策略×易起点,真值=强策略更优(过程能力)。
    return 口径(值=raw score)预期被易起点抬分骗过(acc<0.5);
    regret 口径(值=score−该难度基线)剥掉起点难度→复原能力序(acc>0.5=过)。
    两口径 BT 头各在**非对抗随机配对**上训练,再在对抗配对上测(考验口径可迁移性)。"""
    rng = np.random.default_rng(seed)
    N = len(score)
    # regret 值:减去按起点难度分组的基线(易/难各自均分)
    base = np.zeros(N)
    for h in (0, 1):
        m = start_hard == h
        if m.any():
            base[m] = score[m].mean()
    regret = score - base

    # 训练配对:非对抗随机对(避开正好是对抗结构的对,考验迁移)
    idx_all = list(range(N))
    def train_head(vals):
        f, y = make_pairs(reps, vals, idx_all, margin=1e-6, rng=np.random.default_rng(seed), max_pairs=20000)
        return fit_bt(f, y) if len(y) else None

    w_ret = train_head(score)
    w_reg = train_head(regret)

    # 对抗测试对:a∈{强&难},b∈{弱&易},真值 a 优
    A = np.nonzero((policy_strong == 1) & (start_hard == 1))[0]
    B = np.nonzero((policy_strong == 0) & (start_hard == 0))[0]
    if len(A) == 0 or len(B) == 0 or w_ret is None or w_reg is None:
        return {"error": f"对抗桶为空(强难={len(A)} 弱易={len(B)})或训练对不足"}
    rng.shuffle(A); rng.shuffle(B)
    npair = min(len(A), len(B), 4000)
    Xadv = np.stack([reps[A[k % len(A)]] - reps[B[k % len(B)]] for k in range(npair)])
    yadv = np.ones(npair)  # 真值恒为 a(强&难)优

    acc_ret, lo_ret, hi_ret = boot_acc_ci(w_ret, Xadv, yadv, seed=seed)
    acc_reg, lo_reg, hi_reg = boot_acc_ci(w_reg, Xadv, yadv, seed=seed)
    verdict = "PASS" if lo_reg > 0.5 else "FAIL"
    return {"n_adv_pairs": int(npair),
            "return_caliber": {"acc": round(acc_ret, 4), "ci": [round(lo_ret, 4), round(hi_ret, 4)]},
            "regret_caliber": {"acc": round(acc_reg, 4), "ci": [round(lo_reg, 4), round(hi_reg, 4)]},
            "verdict_s8b": f"{verdict} (regret acc={acc_reg:.4f} ci_lo={lo_reg:.4f} 门 0.5; "
                           f"return acc={acc_ret:.4f} 预期<0.5)"}


# ───────────────────────── 合成自检(植入信号) ─────────────────────────
def synth_data(n=1200, d=64, n_start=40, seed=0):
    """植入:轨迹质量 q(线性可读)+ 起点难度效应 e。
    score=q−e+噪声(难起点压分);rep 同时线性含 q 与 e 两方向 → 口径可分辨。"""
    rng = np.random.default_rng(seed)
    start_id = rng.integers(0, n_start, n)
    start_hard = (start_id % 2).astype(int)                 # 一半起点为难
    e = start_hard * 1.5                                    # 难起点压分 1.5
    policy_strong = rng.integers(0, 2, n)
    q = policy_strong * 1.2 + rng.normal(0, 0.4, n)         # 强策略质量更高
    score = (q - e + rng.normal(0, 0.15, n)).astype(np.float32)
    # rep:前两坐标=q、e 方向(加噪),其余纯噪声 → 线性头能各自读出
    rep = rng.normal(0, 1.0, (n, d)).astype(np.float32)
    rep[:, 0] = q + rng.normal(0, 0.3, n)
    rep[:, 1] = e + rng.normal(0, 0.3, n)
    return rep, score, start_id, policy_strong, start_hard


# ───────────────────────── 塔编码前端(真数据) ─────────────────────────
def encode_traj_dir(ckpt, traj_dir, dev):
    """逐条轨迹过冻结 W4 塔 → STATE(pool_ssm)。返回 reps[N,D] 及元数据数组。"""
    import torch
    from train.fovea_twotower.model_utils import build_eval_model, pool_ssm
    model, ck = build_eval_model(ckpt, dev)
    files = sorted(glob.glob(os.path.join(traj_dir, "*.npz")))
    assert files, f"{traj_dir} 下无 .npz 轨迹"
    reps, score, start_id, policy_strong, start_hard = [], [], [], [], []
    with torch.no_grad():
        mdtype = next(model.parameters()).dtype   # 塔为 bf16;latify 存 fp16/fp32,须对齐免 LayerNorm 类型炸
        for fp in files:
            z = np.load(fp)
            lat = torch.tensor(z["lat"]).unsqueeze(0).to(dev, mdtype)
            act = torch.tensor(z["act"]).unsqueeze(0).to(dev, mdtype)
            msg = torch.tensor(z["msg"]).unsqueeze(0).to(dev, mdtype)
            _, states = model.encode(lat, act, msg, want_states=True)
            reps.append(pool_ssm(states).float().cpu().numpy()[0])
            score.append(float(z["score"]))
            start_id.append(int(z["start_id"]))
            policy_strong.append(int(z["policy_strong"]))
            start_hard.append(int(z["start_hard"]))
    return (np.stack(reps), np.array(score, np.float32),
            np.array(start_id), np.array(policy_strong), np.array(start_hard), ck)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="合成植入信号自检(无需塔/数据)")
    p.add_argument("--ckpt", default="runs/ftt_w4c/ckpt.pt")
    p.add_argument("--traj-dir", default="runs/data/s8_traj")
    p.add_argument("--margin", type=float, default=0.2,
                   help="S8a 配对最小 |Δscore|(滤平局)")
    p.add_argument("--out", default="runs/ftt_s8.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        rep, score, start_id, pol, hard = synth_data(seed=args.seed)
        src = "synthetic"
        ckstep = None
    else:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        rep, score, start_id, pol, hard, ck = encode_traj_dir(args.ckpt, args.traj_dir, dev)
        src = args.traj_dir
        ckstep = ck.get("step")

    res = {"source": src, "ckpt_step": ckstep, "n_traj": int(len(score)),
           "dim_rep": int(rep.shape[1]), "margin": args.margin}
    res["s8a"] = run_s8a(rep, score, start_id, args.margin, seed=args.seed)
    res["s8b"] = run_s8b(rep, score, pol, hard, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
