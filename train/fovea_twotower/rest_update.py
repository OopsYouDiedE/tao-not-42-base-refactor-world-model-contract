# -*- coding: utf-8 -*-
"""ReST/RAFT 更新:朝判优优胜轨迹做 优势加权 BC(条件化在各自指令上)。

一轮闭环的收尾步。输入:
  - 组采样目录(rollout_groups.py 产)runs/rest_rK/feats/*.npz(feats/action/instr_idx);
  - 判优优势 advantages.json {traj_id: advantage}(SubAgent 组内相对排序换算,见判优编排);
  - 指令 emb(instr_emb.pt)、上一轮策略 ckpt(轮 K,首轮=ftt_c2bc)。

RAFT 口径:保留 advantage ≥ keep_thresh 的"优胜"轨迹,按 max(adv,0) 加权做 BC
(相机 CE + 键 BCE,复用 bc_losses),条件=该轨迹指令 emb。text_embed 因此从零被
逐轮灌成"指令→行为"的服从。预编码 feats → _Stub 骨干,更新期不跑 dino,快。

用法:
  PYTHONPATH=. python train/fovea_twotower/rest_update.py \
      --round_dir runs/rest_r0 --adv runs/rest_r0/advantages.json \
      --init runs/ftt_c2bc/best.pt --out runs/rest_r1 --steps 1500
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from net.bc import BCConfig
from net.config import BackboneConfig
from train.fovea_twotower.text_cond_policy import TextCondPolicy
from train.minecraft.train_bc import bc_losses
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE


class _Stub(nn.Module):
    def __init__(self, d):
        super().__init__(); self.embed_dim = d
    def forward(self, x):
        return torch.zeros(x.shape[0], self.embed_dim, device=x.device)


class WinnerWindows(torch.utils.data.IterableDataset):
    """从优胜轨迹采等长窗口 + 指令 emb + 优势权重。按权重(∝adv)对轨迹加权抽样。"""

    def __init__(self, round_dir, adv, instr_emb, seq_len, keep_thresh, seed=0):
        self.seq_len = seq_len; self.seed = seed
        self.emb = instr_emb                      # [K,384]
        files = sorted(glob.glob(os.path.join(round_dir, "feats", "*.npz")))
        self.items, self.weights = [], []
        for f in files:
            tid = os.path.basename(f)[:-4]
            a = float(adv.get(tid, 0.0))
            if a < keep_thresh:
                continue
            with np.load(f, allow_pickle=True) as z:
                if z["feats"].shape[0] < seq_len + 1:
                    continue
                self.items.append((f, int(z["instr_idx"]), max(a, 1e-3)))
                self.weights.append(max(a, 1e-3))
        assert self.items, f"无优胜轨迹(keep_thresh={keep_thresh});检查 advantages.json"
        self.weights = np.array(self.weights) / np.sum(self.weights)
        self.n_win = len(self.items)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        cache = {}
        while True:
            i = rng.choice(len(self.items), p=self.weights)
            f, iidx, w = self.items[i]
            if f not in cache:
                z = np.load(f)
                cache[f] = (z["feats"], z["action"])
            feats, action = cache[f]
            s = int(rng.integers(0, feats.shape[0] - self.seq_len))
            L = self.seq_len
            yield {"feats": torch.from_numpy(feats[s:s + L].astype(np.float32)),
                   "action": torch.from_numpy(action[s:s + L].astype(np.float32)),
                   "text": self.emb[iidx].float(),
                   "w": torch.tensor(float(w), dtype=torch.float32)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round_dir", default="runs/rest_r0")
    p.add_argument("--adv", default="runs/rest_r0/advantages.json")
    p.add_argument("--instr_emb", default="runs/ftt_instr/instr_emb.pt")
    p.add_argument("--init", default="runs/ftt_c2bc/best.pt", help="上一轮策略 ckpt")
    p.add_argument("--out", default="runs/rest_r1")
    p.add_argument("--keep_thresh", type=float, default=0.0, help="保留 adv≥此阈的轨迹")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ie = torch.load(args.instr_emb)
    adv = json.load(open(args.adv))
    if "advantages" in adv:                       # 容错:嵌套或扁平皆可
        adv = adv["advantages"]

    ds = WinnerWindows(args.round_dir, adv, ie["emb"], args.seq_len, args.keep_thresh, args.seed)
    print(f"⛏️ ReST 更新 | 优胜轨迹 {ds.n_win} 条 | init={args.init} → {args.out}")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                                         pin_memory=True, persistent_workers=args.workers > 0)
    it = iter(loader)

    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=384, heads=6, layers=4,
                   dropout=0.1, max_len=max(128, args.seq_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = TextCondPolicy(cfg, injected_backbone=_Stub(384)).to(device)
    ck = torch.load(args.init, map_location=device, weights_only=False)
    sd = ck.get("policy", ck.get("model", ck))
    if any(k.startswith("text_embed.") for k in sd):
        policy.load_state_dict(sd, strict=False)          # 续训 text-cond ckpt
    else:
        policy.load_c2bc(args.init, device)               # 首轮从纯挖起手
    trainable = [q for q in policy.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))

    os.makedirs(args.out, exist_ok=True)
    policy.train(); t0 = time.time()
    for step in range(args.steps):
        b = next(it)
        feats = b["feats"].to(device); target = b["action"].to(device)
        text = b["text"].to(device); w = b["w"].to(device)
        prev = torch.zeros_like(target); prev[:, 1:] = target[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cam_logits, key_logits = policy(feats, prev, text)
        loss = weighted_bc(cam_logits, key_logits, target, w)   # 逐样本 BC × 归一优势权重(RWR)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
        if step % args.log_interval == 0:
            print(f"[{step:5d}/{args.steps}] loss={loss.item():.4f} | "
                  f"{args.batch_size*args.seq_len*(step+1)/max(time.time()-t0,1e-6):.0f} 帧/s", flush=True)
    state = {k: v for k, v in policy.state_dict().items() if not k.startswith("backbone.")}
    torch.save({"policy": state, "step": args.steps, "round_dir": args.round_dir,
                "cfg": {"d": 384, "heads": 6, "layers": 4}}, os.path.join(args.out, "final.pt"))
    print(f"💾 {args.out}/final.pt | {(time.time()-t0)/60:.1f}min", flush=True)


def weighted_bc(cam_logits, key_logits, target, w):
    """逐样本 BC 损失 × 归一优势权重 w[B] → 标量。复用 bc_losses 的分箱/CE 约定但按样本聚合。"""
    import torch.nn.functional as F
    from train.minecraft.vpt_action import camera_to_bin
    B, T = cam_logits.shape[:2]
    cam_tgt = target[..., :N_MOUSE]
    bins = camera_to_bin(cam_tgt).long()                      # [B,T,2]
    ce = F.cross_entropy(cam_logits.reshape(-1, CAMERA_BINS),
                         bins.reshape(-1), reduction="none").view(B, T, N_MOUSE).mean((1, 2))
    key_tgt = target[..., N_MOUSE:]
    bce = F.binary_cross_entropy_with_logits(key_logits, key_tgt, reduction="none").mean((1, 2))
    wn = w / (w.sum() + 1e-8)
    return (wn * (ce + bce)).sum()


if __name__ == "__main__":
    main()
