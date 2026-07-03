"""R2a 探针:采样发散度 ↔ 真实开环误差的相关性 (tests/probe_r2a_divergence.py)。

对应 knowledge/design_rollout_research_program.md §R2:验证"同一起点、同一真实动作序列下,
N 个随机采样 open-loop rollout 的逐步发散度"能否作模型不确定性信号——
与像素空间真实开环误差在固定 k 下跨起点做 Spearman 相关。

过闸判据:mean per-k Spearman ≥ 0.5 采纳为想象截断/降权信号;< 0.3 弃用,回退 K 头集成。
注意:发散度与误差都随 k 单调增长,pooled 相关会被 k 趋势虚高——主指标必须是逐 k 相关的均值。

使用方法(从仓库根目录):
    PYTHONPATH=. python tests/probe_r2a_divergence.py \
        --run-dir runs/crafter_r2a --size small --collect-steps 512 --horizon 15
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.dreamerv3 import build_dreamerv3
from train.crafter.env import VecCrafterEnv
from train.crafter.train_dreamerv3 import SIZE_PRESETS

N_ACTIONS = 17


def parse_args():
    p = argparse.ArgumentParser(description="R2a: rollout 采样发散度 vs 真实开环误差")
    p.add_argument("--run-dir", default="runs/crafter_r2a",
                   help="训练 run 目录(取 checkpoints/ 下最新)或直接给 .pt 路径(--ckpt)")
    p.add_argument("--ckpt", default="", help="检查点路径;空 = run-dir 下最新")
    p.add_argument("--size", choices=list(SIZE_PRESETS), default="small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--collect-steps", type=int, default=512, help="每 env 采集步数")
    p.add_argument("--horizon", type=int, default=15, help="开环步数 K")
    p.add_argument("--n-samples", type=int, default=8, help="每起点采样 rollout 数 N")
    p.add_argument("--stride", type=int, default=8, help="起点间隔")
    p.add_argument("--chunk", type=int, default=128, help="起点批处理块大小(显存控制)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="结果 JSON;空 = {run-dir}/probe_divergence.json")
    return p.parse_args()


def latest_ckpt(run_dir):
    cands = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.pt")),
                   key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(f"{run_dir}/checkpoints 下无检查点")
    return cands[-1]


@torch.no_grad()
def collect(agent, args):
    """用当前策略(采样)采集轨迹。

    Returns:
        obs:      [E, T+1, 3, 64, 64] float ∈[0,1](含末观测)。
        actions:  [E, T, A] one-hot(actions[t] 在 obs[t] 处执行)。
        is_first: [E, T+1] float。
    """
    device = args.device
    envs = VecCrafterEnv(n_envs=args.n_envs, device=device, seed=args.seed + 7)
    obs = envs.reset()
    E, T = args.n_envs, args.collect_steps
    obs_buf = torch.empty(E, T + 1, 3, 64, 64, device=device)
    act_buf = torch.empty(E, T, N_ACTIONS, device=device)
    first_buf = torch.zeros(E, T + 1, device=device)
    first_buf[:, 0] = 1.0
    state, is_first = None, torch.ones(E, device=device)
    for t in range(T):
        obs_buf[:, t] = obs
        idx, onehot, state = agent.policy(obs, state, is_first, training=True)
        obs, _, done, _, _ = envs.step(idx.cpu().numpy())
        act_buf[:, t] = onehot
        is_first = done.to(device=device, dtype=torch.float32)
        first_buf[:, t + 1] = is_first
    obs_buf[:, T] = obs
    return obs_buf, act_buf, first_buf


def spearman(x, y):
    """逐列无平局近似的 Spearman ρ。x, y: [n] np.ndarray。"""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / max(denom, 1e-12))


@torch.no_grad()
def main():
    args = parse_args()
    ckpt_path = args.ckpt or latest_ckpt(args.run_dir)
    agent = build_dreamerv3(device=args.device, num_actions=N_ACTIONS,
                            obs_shape=(3, 64, 64), **SIZE_PRESETS[args.size])
    ckpt = torch.load(ckpt_path, map_location=args.device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    wm, dyn = agent.world_model, agent.world_model.dynamics
    print(f"✅ 检查点 {ckpt_path} (total_steps={ckpt.get('total_steps', '?')})")

    obs, actions, is_first = collect(agent, args)
    E, T1 = obs.shape[:2]
    T, K, N = T1 - 1, args.horizon, args.n_samples

    # 全序列后验(与 wm.loss 同因果对齐:observe 吃右移一位的 prev_action)
    embed = wm.encoder(wm.preprocess_image(obs[:, :-1]))
    prev_action = torch.cat(
        [torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)
    post, _ = dyn.observe(embed, prev_action, is_first[:, :-1])

    # 有效起点:窗口 (t0, t0+K] 内无 episode 重置
    windows = []
    first_np = is_first.cpu().numpy()
    for e in range(E):
        for t0 in range(1, T - K, args.stride):
            if first_np[e, t0 + 1:t0 + K + 1].sum() == 0:
                windows.append((e, t0))
    if not windows:
        raise RuntimeError("无有效窗口(全被重置切断)")
    print(f"起点窗口 {len(windows)} 个 | K={K} N={N}")

    div_all, err_all = [], []                      # 各 [B_total, K]
    for c0 in range(0, len(windows), args.chunk):
        chunk = windows[c0:c0 + args.chunk]
        es = torch.tensor([w[0] for w in chunk], device=args.device)
        ts = torch.tensor([w[1] for w in chunk], device=args.device)
        state0 = {k: v[es, ts] for k, v in post.items()}            # [B, ...]
        act_seq = torch.stack(
            [actions[e, t0:t0 + K] for e, t0 in chunk])             # [B, K, A]
        real = wm.preprocess_image(torch.stack(
            [obs[e, t0 + 1:t0 + K + 1] for e, t0 in chunk]))        # [B, K, 3, H, W]

        feats_n, err_n = [], []
        for _ in range(N):
            prior = dyn.imagine_with_action(act_seq, dict(state0))
            feat = dyn.get_feat(prior)                              # [B, K, F]
            pred = wm.decoder(feat)                                 # [B, K, 3, H, W]
            feats_n.append(feat)
            err_n.append(((pred - real) ** 2).mean(dim=(-3, -2, -1)))   # [B, K]
        feats_n = torch.stack(feats_n)                              # [N, B, K, F]
        div = feats_n.var(dim=0, unbiased=False).mean(-1)           # [B, K]
        err = torch.stack(err_n).mean(0)                            # [B, K]
        div_all.append(div.cpu())
        err_all.append(err.cpu())

    div = torch.cat(div_all).numpy()                                # [B, K]
    err = torch.cat(err_all).numpy()
    per_k = [spearman(div[:, k], err[:, k]) for k in range(K)]
    mean_rho = float(np.mean(per_k))
    pooled = spearman(div.flatten(), err.flatten())                 # 参考值(被 k 趋势虚高)

    verdict = ("PASS(采纳为截断信号)" if mean_rho >= 0.5
               else "FAIL(弃用,回退 K 头集成)" if mean_rho < 0.3
               else "GRAY(0.3-0.5,加样本复测)")
    result = {
        "ckpt": ckpt_path, "total_steps": int(ckpt.get("total_steps", -1)),
        "n_windows": len(windows), "horizon": K, "n_samples": N,
        "spearman_per_k": [round(r, 4) for r in per_k],
        "spearman_mean_per_k": round(mean_rho, 4),
        "spearman_pooled_ref": round(pooled, 4),
        "verdict": verdict,
    }
    out = args.out or os.path.join(args.run_dir, "probe_divergence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"📊 mean per-k Spearman = {mean_rho:.3f} (pooled 参考 {pooled:.3f})")
    print(f"   per-k: {[round(r, 2) for r in per_k]}")
    print(f"🏁 {verdict} → {out}")


if __name__ == "__main__":
    main()
