#!/usr/bin/env python3
"""命题③最小存在性证明:return-conditioned 快头 BC —— "用信号操纵执行能力"。

Decision-Transformer 式:BC 时把每条轨迹自己的回报(C2 score)作条件喂进去训;
推理时**命令**一个目标回报,看行为是否随命令可控变化(命令高回报→更像采矿,命令低→退化)。
若可控 = "微调/条件化能调执行"这条愿景命题拿到最小正证(不依赖 RL,数据现成)。

数据:tests/encode_c2_feats.py --no-strong-only 出的 runs/data/c2_cond_feat(全 2×2 轨迹,
每条带 score)。模型=BCPolicy 子类,仅加一个 ret_embed(标量回报→d),其余(骨干/头/时序干)全复用。

用法:
  PYTHONPATH=. python train/fovea_twotower/train_fasthead_cond.py \
      --data runs/data/c2_cond_feat --total_steps 3000 --run_dir runs/ftt_c2cond
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from net.bc import BCConfig, build_bc_policy
from net.bc.policy import BCPolicy
from net.config import BackboneConfig
from train.minecraft.train_bc import bc_losses
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE

RET_SCALE = 2.0     # 回报归一化(C2 score 量级 0..~3);条件输入 = score/RET_SCALE


class CondPolicy(BCPolicy):
    """BCPolicy + 回报条件。forward(feats, prev, ret) 在输入相加处多一项 ret_embed。"""

    def __init__(self, cfg, injected_backbone=None):
        super().__init__(cfg, injected_backbone)
        self.ret_embed = nn.Linear(1, cfg.d)

    def forward(self, feats, prev_action, ret):
        B, T = feats.shape[:2]
        r = self.ret_embed(ret.view(B, 1, 1).expand(B, T, 1).to(feats.dtype))
        x = self.feat_proj(feats) + self.act_embed(prev_action) + r + self.pos[:, :T]
        for blk in self.trunk:
            x = blk(x)
        x = self.out_norm(x)
        cam = self.cam_head(x).view(B, T, self.cfg.n_mouse, self.cfg.camera_bins)
        return cam, self.key_head(x)


class _Stub(nn.Module):
    def __init__(self, d):
        super().__init__(); self.embed_dim = d
    def forward(self, x):
        return torch.zeros(x.shape[0], self.embed_dim, device=x.device)


class CondWindowDataset(IterableDataset):
    """采样等长窗口 + 该轨迹回报(score/RET_SCALE)。"""

    def __init__(self, data_dir, seq_len, split="train", holdout_n=8, seed=0):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert files, f"{data_dir} 无 npz"
        self.files = files[:-holdout_n] if split == "train" else files[-holdout_n:]
        self.seq_len = seq_len; self.seed = seed
        # 回报分层:pos=score>0(采矿成功,稀少),zero=score==0(多数)。采样时各半,
        # 保证模型充分看到"高回报→采矿"对应,否则 89% 的 0 分会淹没条件信号。
        self.pos, self.zero = [], []
        for f in self.files:
            with np.load(f) as z:
                if z["feats"].shape[0] >= seq_len + 1:
                    (self.pos if float(z["score"]) > 0 else self.zero).append(f)
        assert self.pos and self.zero, f"分层不足 pos={len(self.pos)} zero={len(self.zero)}"

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        cache = {}
        while True:
            pool = self.pos if rng.random() < 0.5 else self.zero   # 分层各半
            f = pool[rng.integers(len(pool))]
            if f not in cache:
                z = np.load(f)
                cache[f] = (z["feats"], z["action"], float(z["score"]))
            feats, action, score = cache[f]
            s = int(rng.integers(0, feats.shape[0] - self.seq_len))
            L = self.seq_len
            yield {"feats": torch.from_numpy(feats[s:s + L].astype(np.float32)),
                   "action": torch.from_numpy(action[s:s + L].astype(np.float32)),
                   "ret": torch.tensor(score / RET_SCALE, dtype=torch.float32)}


def make_batch(sample, device):
    feats = sample["feats"].to(device).float()
    target = sample["action"].to(device).float()
    ret = sample["ret"].to(device).float()
    prev = torch.zeros_like(target); prev[:, 1:] = target[:, :-1]
    return feats, prev, target, ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/c2_cond_feat")
    p.add_argument("--holdout_n", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--run_dir", default="runs/ftt_c2cond")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"⛏️ return-conditioned 快头 BC | data={args.data} seq={args.seq_len} bs={args.batch_size}")

    ds = CondWindowDataset(args.data, args.seq_len, "train", args.holdout_n, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    it = iter(loader)

    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=args.d, heads=args.heads,
                   layers=args.layers, dropout=0.1, max_len=max(128, args.seq_len),
                   action_dim=ACTION_DIM, n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = CondPolicy(cfg, injected_backbone=_Stub(384)).to(device)
    trainable = [p_ for p_ in policy.parameters() if p_.requires_grad]
    print(f"✅ 可训练 {sum(p_.numel() for p_ in trainable)/1e6:.2f}M(含 ret_embed 条件头)")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s+1)/max(args.warmup_steps,1)))

    os.makedirs(args.run_dir, exist_ok=True)
    policy.train(); t0 = time.time()
    for step in range(args.total_steps):
        feats, prev, target, ret = make_batch(next(it), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cam_logits, key_logits = policy(feats, prev, ret)
        cam_ce, key_bce, _ = bc_losses(cam_logits, key_logits, target)
        loss = cam_ce + key_bce
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
        if step % args.log_interval == 0:
            fps = args.batch_size*args.seq_len*(step+1)/max(time.time()-t0, 1e-6)
            print(f"[{step:5d}/{args.total_steps}] loss={loss.item():.4f} "
                  f"cam_ce={cam_ce.item():.4f} key_bce={key_bce.item():.4f} | {fps:.0f} 帧/s")
    state = {k: v for k, v in policy.state_dict().items() if not k.startswith("backbone.")}
    torch.save({"policy": state, "step": args.total_steps, "cfg": vars(args),
                "ret_scale": RET_SCALE}, os.path.join(args.run_dir, "final.pt"))
    print(f"💾 {args.run_dir}/final.pt | 完成 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
