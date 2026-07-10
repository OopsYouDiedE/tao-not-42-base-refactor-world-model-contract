#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VPT 人类视频 → PixelTower 的 BC 暖启动(设计文档 §7 E1;GRPO 只做精修)。

对外接口:
    VPT_TO_V2 — VPT 键序 → V2 键序的置换索引
    encode_targets(act_agg) — 数据契约动作 → (cam bins, keys_v2, prev) 训练目标
    main() — CLI 训练入口(python -m train.craftground.bc_vpt_warmstart)

依据(受控实证,conclusion_fasttower_skill_ceiling.md):GRPO 两次起效都从 BC 暖启动
起跑(可见目标 0.50→0.81);从随机初始化直接 GRPO 的监督带宽差 4~5 个数量级
(design_bitter_lesson §1.1)。本训练器把 VPT 承包商数据边缘化到 CraftGround V2 契约:

  相机:jsonl mouse dx/dy(px)× 0.15 deg/px → /CAM_MAX_DEG 归一 → mu-law 11 bins。
    0.15 = 上游 openai/Video-Pre-Training run 代码的 CAMERA_SCALER=360/2400,是该
    数据集的**格式常量**(schema 知识,非注入物理先验;光流自标定在人类录像上被
    静态覆盖层污染,见 lessons_do_not_retry"感知先验与表征"节)。
  键位:VPT_KEYS → V2_KEYS 逐名置换(w→forward / s→back / a→left / d→right,余同序)。
  口径:T=1 + 帧堆叠 S=4、goal=零向量(无慢塔指导;hindsight relabel 是后续工作)、
    prev=上一 tick 的 [bin中心, keys](与 grpo_pixel.rollout:423 同构造);
    prev 以 --prev-drop 概率整体置零,防 copy-prev 捷径(v17 配方先例,悬置项)。

评估(holdout 独立目录):cam top-1 acc(全 tick / 非零相机 tick)、每键 P/R/F1,
对照"恒预测零相机 bin + 全键不按"的多数类基线——不显著超过即未学到东西。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.craftground.action_contract import (CAM_BINS, CAM_MAX_DEG, CAM_MU,  # noqa: E402
                                               V2_KEYS, stack_frames)
from train.minecraft.vpt_dataset import VPT_KEYS, VPTStreamDataset, _pair_list  # noqa: E402

IMG_HW = (90, 160)                      # 与 grpo_pixel.IMG_HW 一致
DEG_PER_MOUSE_PX = 0.15                 # VPT 数据集格式常量(见文件头)
CAMERA_SCALE = CAM_MAX_DEG / DEG_PER_MOUSE_PX   # =120 px:_action_vec 归一后恰为 deg/18

# VPT 键名 → V2 键名(语义同名,仅 wasd 命名不同)
_V2_OF_VPT = {"key_w": "forward", "key_s": "back", "key_a": "left", "key_d": "right",
              "key_space": "jump", "key_sneak": "sneak", "key_sprint": "sprint",
              "key_attack": "attack", "key_use": "use", "key_drop": "drop",
              "key_inventory": "inventory",
              **{f"key_hotbar.{i}": f"hotbar.{i}" for i in range(1, 10)}}
# 置换索引:keys_v2 = keys_vpt[..., VPT_TO_V2]
VPT_TO_V2 = [VPT_KEYS.index(vk) for v2 in V2_KEYS
             for vk, tgt in _V2_OF_VPT.items() if tgt == v2]
assert len(VPT_TO_V2) == len(V2_KEYS) == 20


def deg_to_bins_t(v: torch.Tensor) -> torch.Tensor:
    """归一相机值 [-1,1] → mu-law bin。[...]float → [...]long。与 numpy 契约同式(单测锚定)。"""
    v = v.float().clamp(-1.0, 1.0)                               # I4:危险算子 fp32
    x = torch.sign(v) * torch.log1p(CAM_MU * v.abs()) / math.log1p(CAM_MU)
    return torch.round((x + 1) / 2 * (CAM_BINS - 1)).long()


def bin_center_t(b: torch.Tensor) -> torch.Tensor:
    """bin → 归一相机值(bin 中心,即 bins_to_deg/CAM_MAX_DEG)。[...]long → [...]float32。"""
    x = b.float() / (CAM_BINS - 1) * 2 - 1
    return torch.sign(x) * (torch.pow(1 + CAM_MU, x.abs()) - 1) / CAM_MU


def encode_targets(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """数据契约动作 → 训练目标。

    Parameters
    ----------
    act : [*, 22] float32(vpt_dataset._action_vec 布局:归一 dx,dy ⊕ VPT 键序 20)

    Returns
    -------
    bins : [*, 2] long — mu-law 相机目标
    keys : [*, 20] float32 — V2 键序目标
    prev : [*, 22] float32 — 作 prev 输入用的 [bin中心 2 ⊕ V2 键 20]
        (与 grpo_pixel.rollout 的 prev=concat([deg/CAM_MAX_DEG, kp]) 同口径:
        相机分量过量化-反量化,与采样端"prev 来自已采样 bin"分布一致)
    """
    bins = deg_to_bins_t(act[..., :2])
    keys = act[..., 2:][..., VPT_TO_V2].float()
    prev = torch.cat([bin_center_t(bins), keys], dim=-1)
    return bins, keys, prev


def _window_batch(img_u8: torch.Tensor, act: torch.Tensor, s: int, device,
                  prev_drop: float, train: bool, rng: torch.Generator | None):
    """窗口批 → 展平的 tick 批(T=1 口径)。

    img_u8 [B,T,3,H,W] uint8;act [B,T-1,22](act[t] = 帧 t 的动作,frame_skip=1)。
    监督 tick 范围 t∈[s-1, T-2]:帧堆叠取窗口内真实历史(不做首帧填充——窗口在
    episode 中段,真实历史存在;与 stack_frames 的 episode 开局填充仅在 rollout
    首 s-1 tick 不同,占比可忽略),prev 取 act[t-1](t≥1 恒成立)。
    返回 img [N,1,3s,H,W] float、goal [N,386] 零、prev [N,1,22]、bins [N,2]、keys [N,20]。
    """
    b, t_n = img_u8.shape[:2]
    ts = torch.arange(s - 1, t_n - 1)                            # 监督 tick
    idx = (ts[:, None] + torch.arange(-(s - 1), 1)[None, :]).to(device)  # [Nt,s] 旧→新
    # 先传原始帧再在 GPU 上做堆叠索引:CPU 侧展开会把批膨胀 s 倍(实测数据管线饿死 GPU)
    img = img_u8.to(device, non_blocking=True)[:, idx].float() / 255.0
    img = img.reshape(b, len(ts), s * 3, *img.shape[-2:])        # [B,Nt,3s,H,W]
    img = img.reshape(-1, 1, *img.shape[2:])                     # [N,1,3s,H,W]
    bins, keys, prev_all = encode_targets(act.to(device, non_blocking=True))
    bins = bins[:, ts].reshape(-1, 2)
    keys = keys[:, ts].reshape(-1, keys.shape[-1])
    prev = prev_all[:, ts - 1].reshape(-1, 1, prev_all.shape[-1])
    if train and prev_drop > 0:
        keep = (torch.rand(prev.shape[0], 1, 1, device=device, generator=rng)
                >= prev_drop).float()
        prev = prev * keep
    goal = torch.zeros(img.shape[0], 384 + 2, device=device)
    return img, goal, prev, bins, keys


def load_holdout(holdout_dir: str, s: int) -> list[dict]:
    """holdout clips 一次性解码+帧堆叠,常驻内存供反复评估(uint8,2 clips ≈ 2.5GB)。"""
    ds = VPTStreamDataset(holdout_dir, seq_len=8, img_size=IMG_HW,
                          camera_scale=CAMERA_SCALE, split=None)
    clips = []
    for mp4, jsonl in ds.pairs:
        clip = ds._load_clip(mp4, jsonl)
        if clip is None:
            continue
        imgs = clip["img"].numpy().transpose(0, 2, 3, 1)         # [T,H,W,3] uint8
        stacked = stack_frames(imgs, s)                          # [T,3s,H,W] 与采样端同序
        bins, keys, prev_all = encode_targets(clip["action"])
        prev = torch.cat([torch.zeros(1, 22), prev_all[:-1]], 0)  # t=0 无前动作
        gui = clip["gui"]
        ok = np.ones(clip["n"], bool) if gui is None else (gui.numpy() < 0.5)
        ok[:s - 1] = False                                       # 开局填充帧不计分
        clips.append(dict(stacked=stacked, bins=bins, keys=keys, prev=prev,
                          ok=ok, n=clip["n"]))
    if not clips:
        raise RuntimeError(f"{holdout_dir} 无可解码 holdout clip")
    return clips


@torch.no_grad()
def evaluate(tower, clips: list[dict], device, chunk: int = 512) -> dict:
    """holdout 全 tick 评估(GUI tick 剔除——GUI 动作是标签噪声,与训练口径一致)。"""
    tower.eval()
    n_cam = n_cam_hit = n_nz = n_nz_hit = n_base = 0
    tp = np.zeros(len(V2_KEYS)); fp = np.zeros(len(V2_KEYS)); fn = np.zeros(len(V2_KEYS))
    pos = np.zeros(len(V2_KEYS)); ce_sum = bce_sum = n_tick = 0.0
    for clip in clips:
        stacked, bins, keys, prev, ok = (clip["stacked"], clip["bins"], clip["keys"],
                                         clip["prev"], clip["ok"])
        for i0 in range(0, clip["n"], chunk):
            sl = slice(i0, min(i0 + chunk, clip["n"]))
            m = ok[sl]
            if not m.any():
                continue
            img = torch.from_numpy(stacked[sl][m]).float().div_(255.0)
            img = img.unsqueeze(1).to(device)
            pv = prev[sl][m].unsqueeze(1).to(device)
            goal = torch.zeros(img.shape[0], 386, device=device)
            cam_l, key_l = tower(img, goal, pv)
            cam_l = cam_l[:, 0, 0].float()                       # [n,2,11]
            key_l = key_l[:, 0, 0].float()                       # [n,20]
            tb = bins[sl][m].to(device)
            tk = keys[sl][m].to(device)
            ce_sum += float(F.cross_entropy(cam_l.reshape(-1, CAM_BINS),
                                            tb.reshape(-1), reduction="sum")) / 2
            bce_sum += float(F.binary_cross_entropy_with_logits(
                key_l, tk, reduction="sum")) / key_l.shape[1]
            n_tick += img.shape[0]
            pred = cam_l.argmax(-1)
            n_cam += tb.numel(); n_cam_hit += int((pred == tb).sum())
            zero_bin = CAM_BINS // 2
            nz = tb != zero_bin
            n_nz += int(nz.sum()); n_nz_hit += int((pred[nz] == tb[nz]).sum())
            n_base += int((tb == zero_bin).sum())
            kp = (torch.sigmoid(key_l) > 0.5).float()
            tp += ((kp == 1) & (tk == 1)).sum(0).cpu().numpy()
            fp += ((kp == 1) & (tk == 0)).sum(0).cpu().numpy()
            fn += ((kp == 0) & (tk == 1)).sum(0).cpu().numpy()
            pos += tk.sum(0).cpu().numpy()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-4)           # I1:分母有界
    sup = pos > 0
    per_key = {k: dict(p=round(float(prec[i]), 3), r=round(float(rec[i]), 3),
                       f1=round(float(f1[i]), 3), pos_rate=round(float(pos[i] / max(n_tick, 1)), 4))
               for i, k in enumerate(V2_KEYS)}
    return dict(cam_acc=round(n_cam_hit / max(n_cam, 1), 4),
                cam_acc_nonzero=round(n_nz_hit / max(n_nz, 1), 4),
                cam_base_zero=round(n_base / max(n_cam, 1), 4),
                nonzero_rate=round(n_nz / max(n_cam, 1), 4),
                key_f1_mean=round(float(f1[sup].mean()) if sup.any() else 0.0, 4),
                ce=round(ce_sum / max(n_tick, 1), 4), bce=round(bce_sum / max(n_tick, 1), 4),
                n_tick=int(n_tick), per_key=per_key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/data/vpt_early")
    ap.add_argument("--holdout", default="runs/data/vpt_holdout")
    ap.add_argument("--out", default="runs/checkpoints/bc_vpt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32, help="窗口数/step(tick 批=batch×(seq-s))")
    ap.add_argument("--seq", type=int, default=20, help="窗口帧数(监督 tick=seq-frame_stack)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--prev-drop", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="300 step 数值冒烟")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.batch = 300, 150, 8

    device = "cuda"
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = PixelTowerConfig(img_hw=IMG_HW, goal_dim=384 + 2, n_keys=len(V2_KEYS),
                           camera_bins=CAM_BINS)                 # 与 grpo_pixel:515 一致
    assert cfg.n_keys == len(V2_KEYS) and cfg.camera_bins == CAM_BINS
    tower = build_pixel_tower(cfg).to(device)
    print(f"PixelTower params = {sum(p.numel() for p in tower.parameters())/1e6:.2f} M",
          flush=True)
    opt = torch.optim.AdamW(tower.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(args.warmup, 1)))
    rng = torch.Generator(device=device); rng.manual_seed(args.seed)

    ds = VPTStreamDataset(args.data, seq_len=args.seq, img_size=IMG_HW,
                          camera_scale=CAMERA_SCALE, frame_skip=1, split=None,
                          clip_cache=3, clip_refresh=192, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers else None)
    it = iter(dl)

    s = cfg.frame_stack
    hold_clips = load_holdout(args.holdout, s)
    best = float("inf"); t0 = time.time(); tick_count = 0
    for step in range(1, args.steps + 1):
        batch = next(it)
        img, goal, prev, bins, keys = _window_batch(
            batch["img"], batch["act_agg"], s, device, args.prev_drop, True, rng)
        tower.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cam_l, key_l = tower(img, goal, prev)
        cam_l = cam_l[:, 0, 0].float()                           # I4:损失 fp32
        key_l = key_l[:, 0, 0].float()
        ce = F.cross_entropy(cam_l.reshape(-1, CAM_BINS), bins.reshape(-1))
        bce = F.binary_cross_entropy_with_logits(key_l, keys)
        loss = ce + bce                                          # 与 grpo_pixel.update 同权
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tower.parameters(), 1.0)
        opt.step(); sched.step()
        tick_count += img.shape[0]

        if step % 50 == 0:
            print(f"[{step}/{args.steps}] loss={float(loss):.4f} ce={float(ce):.4f} "
                  f"bce={float(bce):.4f} ticks/s={tick_count/(time.time()-t0):.0f}",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(tower, hold_clips, device)
            m.update(step=step, train_loss=round(float(loss), 4),
                     lr=opt.param_groups[0]["lr"],
                     wall_min=round((time.time() - t0) / 60, 1))
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
            hold = m["ce"] + m["bce"]
            print(f"[eval@{step}] cam_acc={m['cam_acc']} nz={m['cam_acc_nonzero']} "
                  f"base={m['cam_base_zero']} keyF1={m['key_f1_mean']} "
                  f"ce+bce={hold:.4f}", flush=True)
            ckpt = dict(tower=tower.state_dict(), cfg=vars(cfg), step=step, metrics=m)
            torch.save(ckpt, out / "last.pt")
            if hold < best:
                best = hold
                torch.save(ckpt, out / "best.pt")


if __name__ == "__main__":
    main()
