#!/usr/bin/env python3
"""快塔目标追踪/导航 BC:YOLO 解析 token + goal 指导 → 示范者动作(train/fovea_twotower)。

命题(见 knowledge/design_fovea_yolo_fasttower.md §4):慢塔指导下,快塔(YOLO 解析头 +
TrackNavTower)能否学会"追踪目标 + 朝目标导航"。数据 = S8 C2 强策略轨迹:强策略靠
raycast 对准并逼近铁矿,其相机 (dx,dy) 即天然的追踪校正、按键含挖掘 → BC 即"追踪+挖"。

流程:
  1) 逐帧 YOLO(E) 解析 → 目标 token [T,K,7](首跑算好缓存到 --token_cache,再训直读)。
  2) 窗口采样 (tokens[L,K,7], prev_action[L,22], target[L,22]) + 固定 goal 向量(单任务)。
  3) TrackNavTower 前向 → BC(相机分箱 CE + 按键 BCE,复用 train.minecraft.train_bc.bc_losses)。
  4) holdout:相机 bin top-1 acc(vs 持续性基线)+ 追踪代理(选中目标 token 是否居中)。

YOLO 冻结不可导(感知前端),只训 TrackNavTower(几 M 参数)。多目标可控(追 A vs B)为后续。

用法(GPU;首跑解析较慢,缓存后快):
  PYTHONPATH=. ./.venv/bin/python train/fovea_twotower/train_tracknav.py \
      --raw runs/data/s8_full --token_cache runs/data/s8_tracknav_tok \
      --weights runs/checkpoints/yoloe-11l-seg.pt --total_steps 3000 --run_dir runs/tracknav
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from net.fovea_twotower.yolo_parse import (PARSE_DIM, TrackNavConfig, YoloParseHead,
                                           build_tracknav)
from train.minecraft.train_bc import bc_losses
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_SCALE, N_MOUSE, camera_to_bin

MC_CLASSES = ["iron ore", "stone", "wall", "dirt", "cave floor"]


def clip_action(z):
    """s8 npz → target action [T,22];末帧置 0(同 encode_feats c2 契约)。"""
    T = z["frames"].shape[0]
    act = np.zeros((T, ACTION_DIM), np.float32)
    act[:-1, 0] = np.clip(z["dx"] / CAMERA_SCALE, -1.0, 1.0)
    act[:-1, 1] = np.clip(z["dy"] / CAMERA_SCALE, -1.0, 1.0)
    act[:-1, 2:] = z["keys"]
    return act


def build_token_cache(raw, cache, weights, K, imgsz, strong_only, device, limit=0):
    """对每条 s8 轨迹逐帧 YOLO 解析 → tokens[T,K,7],与 action 一起缓存 npz。增量跳过。"""
    os.makedirs(cache, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw, "*.npz")))
    head = None
    done = skip = 0
    for fp in files:
        if limit and done >= limit:
            break
        z = np.load(fp, allow_pickle=True)
        if strong_only and int(z["policy_strong"]) != 1:
            continue
        outp = os.path.join(cache, os.path.basename(fp))
        if os.path.exists(outp):
            skip += 1
            continue
        if head is None:                                   # 懒建(避免无待解析时也载 YOLO)
            head = YoloParseHead(weights, K=K, imgsz=imgsz,
                                 text_classes=MC_CLASSES, device=device)
        frames = z["frames"]                               # [T,3,126,126] u8
        toks = np.zeros((len(frames), K, PARSE_DIM), np.float32)
        for s in range(0, len(frames), 64):
            batch = torch.from_numpy(frames[s:s + 64])     # [b,3,126,126] u8
            toks[s:s + batch.shape[0]] = head(batch).numpy()
        np.savez(outp, tokens=toks.astype(np.float16), action=clip_action(z),
                 score=z["score"])
        done += 1
        print(f"[parse] {os.path.basename(fp)} T={len(frames)} [{done} done/{skip} skip]", flush=True)
    print(f"[parse] cache DONE {done} 新/{skip} 跳过 → {cache}", flush=True)


class TokWindowDataset(IterableDataset):
    """无限采样等长 token 窗口。split=train 用前 len-holdout_n,holdout 用末段。"""

    def __init__(self, cache, seq_len, split="train", holdout_n=4, seed=0):
        files = sorted(glob.glob(os.path.join(cache, "*.npz")))
        assert files, f"{cache} 无 token 缓存(先跑解析)"
        self.files = files[:-holdout_n] if split == "train" else files[-holdout_n:]
        self.seq_len, self.seed = seq_len, seed
        self.usable = []
        for f in self.files:
            with np.load(f) as z:
                if z["tokens"].shape[0] >= seq_len + 1:
                    self.usable.append(f)
        assert self.usable, f"无片长 ≥ {seq_len + 1}"

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        cache = {}
        while True:
            f = self.usable[rng.integers(len(self.usable))]
            if f not in cache:
                z = np.load(f)
                cache[f] = (z["tokens"], z["action"])
            tok, act = cache[f]
            s = int(rng.integers(0, tok.shape[0] - self.seq_len))
            L = self.seq_len
            yield {"tokens": torch.from_numpy(tok[s:s + L].astype(np.float32)),
                   "action": torch.from_numpy(act[s:s + L].astype(np.float32))}


def make_batch(sample, goal, device):
    tokens = sample["tokens"].to(device)                   # [B,L,K,7]
    target = sample["action"].to(device).float()           # [B,L,22]
    prev = torch.zeros_like(target); prev[:, 1:] = target[:, :-1]
    g = goal.expand(tokens.shape[0], -1)                   # [B,goal_dim]
    return tokens, g, prev, target


@torch.no_grad()
def evaluate(tower, goal, hold_iter, n, device):
    tower.eval()
    cc = ct = cp = 0
    for _ in range(n):
        tokens, g, prev, target = make_batch(next(hold_iter), goal, device)
        cam, _ = tower(tokens, g, prev)
        pred = cam.float().argmax(-1)
        tgt = camera_to_bin(target[..., :N_MOUSE])
        cc += (pred == tgt).sum().item(); ct += tgt.numel()
        cp += (camera_to_bin(prev[..., :N_MOUSE]) == tgt).sum().item()
    tower.train()
    return {"cam_acc": cc / max(ct, 1), "cam_acc_persist": cp / max(ct, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="runs/data/s8_full")
    p.add_argument("--token_cache", default="runs/data/s8_tracknav_tok")
    p.add_argument("--weights", default="runs/checkpoints/yoloe-11l-seg.pt")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--strong_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--parse_limit", type=int, default=0, help=">0 只解析前 N 段(冒烟)")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--goal_dim", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--holdout_n", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--parse_only", action="store_true", default=False)
    p.add_argument("--run_dir", default="runs/tracknav")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    build_token_cache(args.raw, args.token_cache, args.weights, args.K, args.imgsz,
                      args.strong_only, device, args.parse_limit)
    if args.parse_only:
        return

    goal = torch.zeros(1, args.goal_dim, device=device)     # 单任务固定 goal(多目标可控为后续)
    goal[0, 0] = 1.0
    cfg = TrackNavConfig(d=args.d, layers=args.layers, heads=args.heads,
                         goal_dim=args.goal_dim, max_len=max(128, args.seq_len))
    tower = build_tracknav(cfg).to(device)
    n_par = sum(x.numel() for x in tower.parameters()) / 1e6

    ds_kw = dict(seq_len=args.seq_len, holdout_n=args.holdout_n, seed=args.seed)
    tr = DataLoader(TokWindowDataset(args.token_cache, split="train", **ds_kw),
                    batch_size=args.batch_size, num_workers=args.workers,
                    persistent_workers=args.workers > 0)
    hd = DataLoader(TokWindowDataset(args.token_cache, split="holdout", **ds_kw),
                    batch_size=args.batch_size, num_workers=1, persistent_workers=True)
    tr_it, hd_it = iter(tr), iter(hd)
    opt = torch.optim.AdamW(tower.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"[tracknav] {n_par:.2f}M 可训 | goal_dim={args.goal_dim} seq={args.seq_len}", flush=True)

    tower.train(); best = float("inf"); t0 = time.time()
    for step in range(args.total_steps):
        tokens, g, prev, target = make_batch(next(tr_it), goal, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            cam, key = tower(tokens, g, prev)
        cam_ce, key_bce, _ = bc_losses(cam, key, target)
        loss = cam_ce + key_bce
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(tower.parameters(), 1.0); opt.step(); sched.step()
        if step % args.log_interval == 0:
            print(f"[{step:5d}/{args.total_steps}] loss={loss.item():.4f} "
                  f"cam_ce={cam_ce.item():.4f} key_bce={key_bce.item():.4f}", flush=True)
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.total_steps:
            m = evaluate(tower, goal, hd_it, 16, device)
            print(f"    holdout@{step+1}: cam_acc={m['cam_acc']:.3f}"
                  f"(持续 {m['cam_acc_persist']:.3f})", flush=True)
            if cam_ce.item() < best:
                best = cam_ce.item()
                torch.save({"tower": tower.state_dict(), "cfg": vars(cfg),
                            "step": step + 1, "args": vars(args)},
                           os.path.join(args.run_dir, "best.pt"))
    torch.save({"tower": tower.state_dict(), "cfg": vars(cfg), "step": args.total_steps,
                "args": vars(args)}, os.path.join(args.run_dir, "final.pt"))
    print(f"[tracknav] done {(time.time()-t0)/60:.1f}min → {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
