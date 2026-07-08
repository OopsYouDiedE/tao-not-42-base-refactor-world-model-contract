#!/usr/bin/env python3
"""E1 训练端:goal 相对 token → 指令条件快头 BC(train/fovea_twotower)。

与 train_tracknav 的差异(那版=单任务固定 goal,token 类列未校准):
  · token 折成 **goal 相对视图** [K,8] = [几何6, p_goal, p_other_max]——
    p_goal = 指令类 softmax 概率,p_other_max = 其余类(含背景)最大概率。
    快头结构上类无关:"听指挥"不靠 goal 向量语义,靠输入契约本身;切换指令
    = token 的 p_goal 列换列 → 策略应立即重定向(闭环判据见 eval_track_cmd)。
  · 示范含局中切换段落(采集器 switch_t),BC 直接学到"重瞄准"行为。

用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/train_track_cmd.py \
      --data runs/data/trackcmd --run_dir runs/trackcmd_bc --total_steps 3000
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import torch.nn.functional as F

from net.fovea_twotower.token_stream import (PARSE_DIM_REL,  # noqa: F401
                                             goal_relative)
from net.fovea_twotower.yolo_parse import TrackNavConfig, build_tracknav
from train.minecraft.vpt_action import (ACTION_DIM, CAMERA_BINS, N_MOUSE,
                                        camera_to_bin)

CAM_NORM_PX = 120.0            # 相机归一化(px):教师单步 ±18°=±120px 满量程。
                               # 勿用 vpt_action.CAMERA_SCALE=10(人类小步口径):
                               # 会把教师大转角截断到 ±1.5°/步,学生重瞄准慢 12 倍,切换判据必死


def load_traj(fp):
    z = np.load(fp, allow_pickle=True)
    toks = goal_relative(z["tokens"].astype(np.float32), z["goal_idx"])
    T = toks.shape[0]
    act = np.zeros((T, ACTION_DIM), np.float32)
    act[:, 0] = np.clip(z["dx"] / CAM_NORM_PX, -1, 1)
    act[:, 1] = np.clip(z["dy"] / CAM_NORM_PX, -1, 1)
    act[:, 2:] = z["keys"]
    switches = np.where(np.diff(z["goal_idx"]) != 0)[0] + 1
    return toks, act, switches


class RelTokDataset(IterableDataset):
    """等长窗口无限采样;按局切 train/holdout(不跨局)。"""

    def __init__(self, data, seq_len, split="train", holdout_n=6, seed=0,
                 switch_os=0.5, max_train_files=0, chunk_k=1):
        files = sorted(f for d in data.split(",")                  # 逗号=多目录聚合
                       for f in glob.glob(os.path.join(d, "*.npz")))
        assert len(files) > holdout_n, f"{data} 轨迹不足"
        self.files = files[:-holdout_n] if split == "train" else files[-holdout_n:]
        if split == "train" and max_train_files > 0:
            self.files = self.files[:max_train_files]              # C2 示范量曲线
        self.seq_len, self.seed, self.switch_os = seq_len, seed, switch_os
        self.chunk_k = chunk_k     # 分块:动作窗延伸 k-1 步,target[t]=act[t:t+k]

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        cache = {}
        while True:
            f = self.files[rng.integers(len(self.files))]
            if f not in cache:
                cache[f] = load_traj(f)
            toks, act, sw = cache[f]
            L, k = self.seq_len, self.chunk_k
            smax = toks.shape[0] - L - k + 1          # 起点上限(动作延伸 k-1 步)
            if smax <= 0:                             # 轨迹短于窗+块:换一条
                continue
            if len(sw) and rng.random() < self.switch_os:   # 切换窗口过采样:重定向行为
                t0 = int(sw[rng.integers(len(sw))])     # 是稀疏事件(v9 切换 0.23 病因)
                s = int(np.clip(t0 - rng.integers(8, L - 8), 0, smax))
            else:
                s = int(rng.integers(0, smax + 1))
            yield {"tokens": torch.from_numpy(toks[s:s + L]),
                   "action": torch.from_numpy(act[s:s + L + k - 1])}


def make_batch(sample, device, prev_dropout=0.0, rng=None, chunk_k=1):
    """→ tokens[B,L,K,8], g[B,1], prev[B,L,A], target[B,L,k,A]。

    action 窗长 L+k-1:cur=前 L 步(逐步执行/构造 prev),target[t]=act[t:t+k](块)。"""
    tokens = sample["tokens"].to(device)
    act_full = sample["action"].to(device).float()          # [B, L+k-1, A]
    L = tokens.shape[1]
    cur = act_full[:, :L]                                   # [B,L,A] 逐步执行动作
    target = torch.stack([act_full[:, j:j + L] for j in range(chunk_k)], dim=2)  # [B,L,k,A]
    prev = torch.zeros_like(cur)
    prev[:, 1:] = cur[:, :-1]
    if prev_dropout > 0 and rng is not None:                # 因果混淆修法:随机切断
        keep = torch.rand(tokens.shape[0], 1, 1, device=device,
                          generator=rng) >= prev_dropout    # "抄自己惯性"捷径(v2 闭环
        prev = prev * keep                                  # 学生退化成 frozen 的病根)
    g = torch.zeros(tokens.shape[0], 1, device=device)      # goal 已折进 token,占位
    return tokens, g, prev, target


def bin_weights(files, device):
    """相机 bin 逆频权重:锁定期小箱压倒多数 → argmax 坍塌"永远微调",
    大箱(重瞄准)欠表达 → 闭环无法发起转身。sqrt 逆频折中。"""
    cnt = torch.zeros(CAMERA_BINS)
    for f in files:
        _, act, _sw = load_traj(f)
        b = camera_to_bin(torch.from_numpy(act[:, :N_MOUSE]))
        cnt += torch.bincount(b.flatten(), minlength=CAMERA_BINS).float()
    w = (cnt.sum() / (cnt + 1.0)).sqrt()
    w = (w / w.mean()).clamp(max=3.0)     # 加帽:无帽时大箱权重过猛→闭环过度转身(v3 教训)
    return (w / w.mean()).to(device)


KEY_POS_W = torch.ones(ACTION_DIM - N_MOUSE)
KEY_POS_W[0] = 4.0             # forward 正例仅 21%,普通 BCE 学成站桩(v9 到达 0.0 病因)


def weighted_bc_losses(cam_logits, key_logits, target, w):
    """target [B,T,k,A];cam/key logits 允许旧口径(k=1 无块维)或 [B,T,k,...]。
    块内均匀权重(cross_entropy 默认对 k 维一并求均值,不引折扣)。"""
    B, T, k, A = target.shape
    cbins = cam_logits.shape[-1]
    cam = cam_logits.reshape(B, T, k, N_MOUSE, cbins)
    key = key_logits.reshape(B, T, k, A - N_MOUSE)
    cam_tgt = camera_to_bin(target[..., :N_MOUSE])           # [B,T,k,N_MOUSE]
    cam_ce = F.cross_entropy(cam.reshape(-1, cbins).float(),
                             cam_tgt.reshape(-1), weight=w)
    key_bce = F.binary_cross_entropy_with_logits(
        key.float(), target[..., N_MOUSE:],
        pos_weight=KEY_POS_W.to(key_logits.device))
    return cam_ce, key_bce


@torch.no_grad()
def evaluate(tower, hold_iter, n, device, chunk_k=1):
    """留出首步(部署执行步)相机准确率 + 键 F1,与逐 tick(v17)口径可比。"""
    tower.eval()
    cc = ct = cp = 0
    tp = fp = fn = 0
    for _ in range(n):
        tokens, g, prev, target = make_batch(next(hold_iter), device, chunk_k=chunk_k)
        cam, key = tower(tokens, g, prev)
        B, T, k, A = target.shape
        cbins = cam.shape[-1]
        cam = cam.reshape(B, T, k, N_MOUSE, cbins)
        key = key.reshape(B, T, k, A - N_MOUSE)
        tgt = camera_to_bin(target[:, :, 0, :N_MOUSE])       # 首步
        cc += (cam[:, :, 0].float().argmax(-1) == tgt).sum().item()
        ct += tgt.numel()
        cp += (camera_to_bin(prev[..., :N_MOUSE]) == tgt).sum().item()
        kp = key[:, :, 0].float().sigmoid() > 0.5
        kt = target[:, :, 0, N_MOUSE:] > 0.5
        tp += (kp & kt).sum().item()
        fp += (kp & ~kt).sum().item()
        fn += (~kp & kt).sum().item()
    tower.train()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"cam_acc": cc / max(ct, 1), "cam_acc_persist": cp / max(ct, 1),
            "key_f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/trackcmd")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout_n", type=int, default=6)
    p.add_argument("--prev_dropout", type=float, default=0.5)
    p.add_argument("--switch_os", type=float, default=0.5)
    p.add_argument("--d", type=int, default=256, help="C2 规模曲线:256/384/512")
    p.add_argument("--layers", type=int, default=3, help="C2 规模曲线:3/5/7")
    p.add_argument("--max_train_files", type=int, default=0,
                   help=">0 截断训练轨迹数(C2 示范量曲线)")
    p.add_argument("--chunk_k", type=int, default=1,
                   help="动作分块步数(k=1 逐 tick=v17 口径;R-B 定标 k=4/8)")
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--run_dir", default="runs/trackcmd_bc")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = TrackNavConfig(parse_dim=PARSE_DIM_REL, goal_dim=1, d=args.d,
                         layers=args.layers, heads=4,
                         max_len=max(128, args.seq_len), chunk_k=args.chunk_k)
    tower = build_tracknav(cfg).to(device)
    print(f"[trackcmd_bc] {sum(x.numel() for x in tower.parameters())/1e6:.2f}M", flush=True)

    kw = dict(seq_len=args.seq_len, holdout_n=args.holdout_n, seed=args.seed,
              switch_os=args.switch_os, max_train_files=args.max_train_files,
              chunk_k=args.chunk_k)
    tr = iter(DataLoader(RelTokDataset(args.data, split="train", **kw),
                         batch_size=args.batch_size, num_workers=2,
                         persistent_workers=True))
    hd = iter(DataLoader(RelTokDataset(args.data, split="holdout", **kw),
                         batch_size=args.batch_size, num_workers=1,
                         persistent_workers=True))
    opt = torch.optim.AdamW(tower.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    os.makedirs(args.run_dir, exist_ok=True)

    files_tr = sorted(f for d in args.data.split(",")
                      for f in glob.glob(os.path.join(d, "*.npz")))[:-args.holdout_n]
    w_cam = bin_weights(files_tr, device)
    print(f"[trackcmd_bc] bin 权重 {[round(float(x),2) for x in w_cam]}", flush=True)
    grng = torch.Generator(device=device).manual_seed(args.seed)

    best = -1.0
    t0 = time.time()
    for step in range(args.total_steps):
        tokens, g, prev, target = make_batch(next(tr), device,
                                             args.prev_dropout, grng,
                                             chunk_k=args.chunk_k)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            cam, key = tower(tokens, g, prev)
        cam_ce, key_bce = weighted_bc_losses(cam, key, target, w_cam)
        loss = cam_ce + key_bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tower.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"[{step:5d}] loss={loss.item():.4f}", flush=True)
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.total_steps:
            m = evaluate(tower, hd, 16, device, chunk_k=args.chunk_k)
            print(f"    holdout@{step+1}: cam_acc={m['cam_acc']:.3f} "
                  f"(持续 {m['cam_acc_persist']:.3f}) key_f1={m['key_f1']:.3f}", flush=True)
            if m["cam_acc"] > best:
                best = m["cam_acc"]
                torch.save({"tower": tower.state_dict(), "cfg": vars(cfg),
                            "cam_acc": best, "args": vars(args)},
                           os.path.join(args.run_dir, "best.pt"))
    print(f"[trackcmd_bc] done {(time.time()-t0)/60:.1f}min best={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
