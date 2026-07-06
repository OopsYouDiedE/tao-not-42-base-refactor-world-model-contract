#!/usr/bin/env python3
"""快头行为克隆(BC):吃**预编码 CLS 特征**直训,免 mp4 解码 + 免 DINO 前向。

train/minecraft/train_bc.py 的预编码版——同一 BCPolicy、同一动作契约与损失,唯一差异
是数据源:不再逐 batch 现场 encode_frames,而是读 tests/encode_g500_feats.py 产出的
runs/data/g500_mc_feat/*.npz(feats fp16 [T,384] + action fp32 [T,22]=[dx/scale,dy/scale,keys×20])。
预编码把 DINO 前向从训练热路径移除,sps 由此从"每步一次骨干前向"跃到"纯时序头前向"。

动作/时间对齐(与 encode_g500_feats 逐字段一致):action[t] = 观测 o_t→o_{t+1} 的聚合动作;
策略在 o_t(与 a_{<t})下预测 a_t。窗口 [s, s+L) 的 L 个 action 全为有效转移(采样保证
s+L-1 ≤ T-2),故直接 feats=[B,L,384] 喂 forward,prev=右移一位补零。

评估(holdout 末 N 片,窗口不与训练重叠):相机 bin top-1 acc(vs 多数 bin / 持续性基线)、
按键 micro-F1(vs 持续性基线)。持续性基线(抄上一步动作)是关键对照:相机/键位是惯性
信号,低于它=没从视觉学到东西。

用法(GPU;骨干权重已由 encode 阶段缓存):
    PYTHONPATH=. python train/fovea_twotower/train_fasthead.py \
        --data runs/data/g500_mc_feat --seq_len 128 --batch_size 16 --total_steps 4000
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from net.bc import BCConfig, CondPolicy, build_bc_policy
from net.config import BackboneConfig
from train.minecraft.train_bc import bc_losses
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE, camera_to_bin


# ── 数据:预编码特征窗口 ────────────────────────────────────────────────
class FeatWindowDataset(IterableDataset):
    """无限采样等长窗口。文件按长度加权(长会话→更多窗口,避免短片过采)。

    每片惰性加载进 worker 本地缓存(feats fp16 + action fp32,数 MB/片,72 片可全驻留)。
    split='train' 用前 len-holdout_n 片,'holdout' 用末 holdout_n 片。
    """

    def __init__(self, data_dir, seq_len, split="train", holdout_n=3, seed=0):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert files, f"{data_dir} 下无 npz"
        self.files = files[:-holdout_n] if split == "train" else files[-holdout_n:]
        assert self.files, f"split={split} 无文件(holdout_n={holdout_n} 太大?)"
        self.seq_len = seq_len
        self.seed = seed
        # 预读长度(只读 header,便宜),过滤过短片、算采样权重
        lens = []
        for f in self.files:
            with np.load(f) as z:
                lens.append(int(z["feats"].shape[0]))
        self.usable = [(f, T) for f, T in zip(self.files, lens) if T >= seq_len + 1]
        assert self.usable, f"无片长 ≥ seq_len+1={seq_len + 1}"
        w = np.array([T - seq_len for _, T in self.usable], np.float64)
        self.p = w / w.sum()

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        cache = {}
        while True:
            i = rng.choice(len(self.usable), p=self.p)
            f, T = self.usable[i]
            if f not in cache:
                z = np.load(f)
                cache[f] = (z["feats"], z["action"])
            feats, action = cache[f]
            s = int(rng.integers(0, T - self.seq_len))
            L = self.seq_len
            yield {"feats": torch.from_numpy(feats[s:s + L].astype(np.float32)),
                   "action": torch.from_numpy(action[s:s + L].astype(np.float32))}


class CondWindowDataset(IterableDataset):
    """return-conditioned 窗口:采样等长窗口 + 该轨迹回报(score/cond_scale)。

    回报分层各半采样:pos=score>0(采矿成功,稀少)/ zero=score==0(多数),
    否则多数 0 分会淹没条件信号(见 knowledge/fovea 命题③)。yield 额外带 'ret'。
    """

    def __init__(self, data_dir, seq_len, split="train", holdout_n=8, seed=0,
                 cond_scale=2.0):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert files, f"{data_dir} 无 npz"
        self.files = files[:-holdout_n] if split == "train" else files[-holdout_n:]
        self.seq_len = seq_len
        self.seed = seed
        self.cond_scale = cond_scale
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
                   "ret": torch.tensor(score / self.cond_scale, dtype=torch.float32)}


def make_batch(sample, device):
    """sample → (feats [B,L,enc], prev_act [B,L,A], target [B,L,A], ret [B]|None)。

    prev=右移补零;ret 仅 return-conditioned(--cond)数据集出现,否则 None。
    """
    feats = sample["feats"].to(device, non_blocking=True).float()
    target = sample["action"].to(device, non_blocking=True).float()
    prev = torch.zeros_like(target)
    prev[:, 1:] = target[:, :-1]
    ret = sample["ret"].to(device).float() if "ret" in sample else None
    return feats, prev, target, ret


def _forward(policy, feats, prev, ret):
    """条件/非条件策略前向统一入口:ret 非空走 CondPolicy(feats,prev,ret)。"""
    return policy(feats, prev, ret) if ret is not None else policy(feats, prev)


@torch.no_grad()
def evaluate(policy, hold_iter, n_batches, device, use_amp):
    policy.eval()
    cam_correct = cam_total = cam_persist = 0
    key_tp = key_fp = key_fn = p_tp = p_fp = p_fn = 0
    cam_ce_sum = key_bce_sum = 0.0
    all_bins = []
    for _ in range(n_batches):
        feats, prev, target, ret = make_batch(next(hold_iter), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            cam_logits, key_logits = _forward(policy, feats, prev, ret)
        cam_ce, key_bce, cam_tgt = bc_losses(cam_logits, key_logits, target)
        cam_ce_sum += cam_ce.item()
        key_bce_sum += key_bce.item()
        pred = cam_logits.float().argmax(-1)
        cam_correct += (pred == cam_tgt).sum().item()
        cam_total += cam_tgt.numel()
        all_bins.append(cam_tgt.flatten().cpu())
        cam_persist += (camera_to_bin(prev[..., :N_MOUSE]) == cam_tgt).sum().item()
        kt = target[..., N_MOUSE:] > 0.5
        kp = key_logits.float().sigmoid() > 0.5
        key_tp += (kp & kt).sum().item(); key_fp += (kp & ~kt).sum().item()
        key_fn += (~kp & kt).sum().item()
        pp = prev[..., N_MOUSE:] > 0.5
        p_tp += (pp & kt).sum().item(); p_fp += (pp & ~kt).sum().item()
        p_fn += (~pp & kt).sum().item()
    policy.train()
    bins = torch.cat(all_bins)
    maj = bins.bincount(minlength=CAMERA_BINS).max().item() / max(len(bins), 1)
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    return {"cam_ce": cam_ce_sum / n_batches, "key_bce": key_bce_sum / n_batches,
            "cam_acc": cam_correct / max(cam_total, 1), "cam_acc_majority": maj,
            "cam_acc_persist": cam_persist / max(cam_total, 1),
            "key_f1": f1(key_tp, key_fp, key_fn), "key_f1_persist": f1(p_tp, p_fp, p_fn)}


def parse_args():
    p = argparse.ArgumentParser(description="快头预编码 BC 训练")
    p.add_argument("--data", default="runs/data/g500_mc_feat")
    p.add_argument("--holdout_n", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--cond", action="store_true", default=False,
                   help="return-conditioned 快头 BC:分层采样 + CondPolicy(ret_embed 条件头)")
    p.add_argument("--cond_scale", type=float, default=2.0,
                   help="回报归一化尺度(条件输入=score/cond_scale);仅 --cond 生效")
    p.add_argument("--backbone", choices=["dinov3", "dinov2"], default="dinov3",
                   help="须与 encode 时一致(feats 契约);gate 时现场编码同款")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--key_loss_coeff", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--n_eval_batches", type=int, default=16)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--run_dir", default="runs/ftt_fasthead")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval_only", action="store_true", default=False)
    p.add_argument("--no_backbone", action="store_true", default=True,
                   help="训练不需骨干(feats 已编码);默认不加载省显存/免下载")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class _StubBackbone(torch.nn.Module):
    """占位骨干:训练期 feats 已预编码,不需真骨干。带 embed_dim 供 build_backbone 注入路径。
    gate/推理需真骨干时用 train_bc 全量策略,不走这里。"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x):  # [b,3,H,W] → [b,embed_dim];训练不调用
        return torch.zeros(x.shape[0], self.embed_dim, device=x.device)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_amp = args.amp and device.type == "cuda"

    print("=" * 78)
    print("⛏️  快头预编码 BC(fast-head, pre-encoded CLS)")
    print(f"   data={args.data} | backbone={args.backbone} | seq_len={args.seq_len} "
          f"| bs={args.batch_size} | amp={'bf16' if use_amp else 'off'}")
    print("=" * 78)

    enc_dim = 384 if args.backbone == "dinov3" else 384
    ds_kw = dict(seq_len=args.seq_len, holdout_n=args.holdout_n, seed=args.seed)
    DS = CondWindowDataset if args.cond else FeatWindowDataset
    if args.cond:
        ds_kw["cond_scale"] = args.cond_scale
    train_ds = DS(args.data, split="train", **ds_kw)
    hold_ds = DS(args.data, split="holdout", **ds_kw)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    hold_loader = DataLoader(hold_ds, batch_size=args.batch_size, num_workers=1,
                             pin_memory=True, persistent_workers=True)
    train_iter, hold_iter = iter(train_loader), iter(hold_loader)

    cfg = BCConfig(backbone=BackboneConfig(kind=args.backbone), d=args.d,
                   heads=args.heads, layers=args.layers, dropout=args.dropout,
                   max_len=max(128, args.seq_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    inj = _StubBackbone(enc_dim) if args.no_backbone else None
    policy = (CondPolicy(cfg, injected_backbone=inj) if args.cond
              else build_bc_policy(cfg, injected_backbone=inj)).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"✅ 可训练 {n_train / 1e6:.2f}M(时序头,骨干{'占位' if inj else '冻结'})")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = policy.load_state_dict(ckpt["policy"], strict=False)
        assert not unexpected, f"未知权重: {unexpected[:4]}"
        if not args.eval_only and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"♻️  恢复 {args.resume}(step={start_step})")

    os.makedirs(args.run_dir, exist_ok=True)

    def save_ckpt(tag, step, metrics=None):
        state = {k: v for k, v in policy.state_dict().items()
                 if not k.startswith("backbone.")}   # 骨干不入 ckpt(占位/HF 可复原)
        path = os.path.join(args.run_dir, f"{tag}.pt")
        torch.save({"policy": state, "optimizer": optimizer.state_dict(),
                    "step": step, "cfg": vars(args), "metrics": metrics}, path)
        print(f"💾 {path}")

    if args.eval_only:
        m = evaluate(policy, hold_iter, args.n_eval_batches, device, use_amp)
        print("📊 holdout:", {k: round(v, 4) for k, v in m.items()})
        return

    policy.train()
    best = float("inf")
    t0 = time.time()
    try:
        for step in range(start_step, args.total_steps):
            feats, prev, target, ret = make_batch(next(train_iter), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                cam_logits, key_logits = _forward(policy, feats, prev, ret)
            cam_ce, key_bce, _ = bc_losses(cam_logits, key_logits, target)
            loss = cam_ce + args.key_loss_coeff * key_bce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            sched.step()

            if step % args.log_interval == 0:
                fps = args.batch_size * args.seq_len * (step - start_step + 1) \
                    / max(time.time() - t0, 1e-6)
                print(f"[{step:5d}/{args.total_steps}] loss={loss.item():.4f} "
                      f"cam_ce={cam_ce.item():.4f} key_bce={key_bce.item():.4f} | {fps:.0f} 帧/s")
            if (step + 1) % args.eval_interval == 0 or step + 1 == args.total_steps:
                m = evaluate(policy, hold_iter, args.n_eval_batches, device, use_amp)
                print(f"    📊 holdout@{step + 1}: cam_acc={m['cam_acc']:.3f}"
                      f"(多数 {m['cam_acc_majority']:.3f}/持续 {m['cam_acc_persist']:.3f}) "
                      f"key_F1={m['key_f1']:.3f}(持续 {m['key_f1_persist']:.3f}) "
                      f"cam_ce={m['cam_ce']:.3f} key_bce={m['key_bce']:.3f}")
                hl = m["cam_ce"] + m["key_bce"]
                if hl < best:
                    best = hl
                    save_ckpt("best", step + 1, m)
    except KeyboardInterrupt:
        print("\n⏹️  中断")
    finally:
        save_ckpt("final", args.total_steps)
    print(f"✅ 完成 {(time.time() - t0) / 60:.1f} min,best holdout loss={best:.4f}")


if __name__ == "__main__":
    main()
