# -*- coding: utf-8 -*-
"""C1(step3 指令总线):文本指令 token 条件化的 Action 塔 BC。

指令 = 事后标注(hindsight):从窗口未来 0.8s 真实动作导出适用指令集,
70% 概率从适用集均匀采样,30% 强制"act freely"(空指令,保持自由行为校准)。
指令短语过 MiniLM 编成 384 维文本嵌入(runs/ftt_cmd/cmd_emb.pt),
作为首 token 钉在动作塔序列头;历史播种(W1 状态,真实消息)与 M1 臂一致。

词表(en 训练用 / zh 对照): turn left/向左转, turn right/向右转, open fire/开火,
move forward/前进, jump/跳, act freely/自由行动。

用法:
    PYTHONPATH=. python train/fovea_twotower/train_cmd.py \
        --ctx runs/ftt_w1/ckpt.pt --out runs/ftt_c1 --rng 0
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net.fovea_twotower import ActionTower, ContextTower
from train.fovea_twotower.train_w2 import H, prep
from train.gaming500.dataset import Gaming500Dataset, KEY_NAMES, N_MSG

DX_TH = 0.5                                            # 明确转向阈(featurize 域)
I_ATT = 2 + KEY_NAMES.index("key_attack")
I_FWD = 2 + KEY_NAMES.index("key_w")
I_JMP = 2 + KEY_NAMES.index("key_space")
LEFT, RIGHT, FIRE, FWD, JUMP, FREE = range(6)
P_FREE = 0.3


def applicable(z1):
    """z1 [B,H,24] → list[list[int]]:每窗适用指令(dx正=右,标注约定)。"""
    dxs = z1[..., 0].sum(1).float()
    fire = (z1[..., I_ATT] > 0.5).any(1)
    fwd = (z1[..., I_FWD] > 0.5).float().mean(1) >= 0.5
    jmp = (z1[..., I_JMP] > 0.5).any(1)
    out = []
    for i in range(z1.shape[0]):
        a = []
        if dxs[i] < -DX_TH:
            a.append(LEFT)
        if dxs[i] > DX_TH:
            a.append(RIGHT)
        if fire[i]:
            a.append(FIRE)
        if fwd[i]:
            a.append(FWD)
        if jmp[i]:
            a.append(JUMP)
        out.append(a or [FREE])
    return out


def sample_cmd(z1, gen):
    """事后标注采样 → 指令 id [B](30% 空指令)。"""
    apps = applicable(z1)
    ids = []
    for a in apps:
        if torch.rand((), generator=gen).item() < P_FREE:
            ids.append(FREE)
        else:
            ids.append(a[int(torch.randint(len(a), (), generator=gen))])
    return torch.tensor(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_360p")
    p.add_argument("--ctx", default="runs/ftt_w1/ckpt.pt")
    p.add_argument("--emb", default="runs/ftt_cmd/cmd_emb.pt")
    p.add_argument("--out", default="runs/ftt_c1")
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.rng)
    sgen = torch.Generator().manual_seed(777)          # 消息扰动(mode=1 不用,占位)
    cgen = torch.Generator().manual_seed(555)          # 指令采样专用,独立 RNG

    emb = torch.load(args.emb)["emb"].to(dev).bfloat16()   # [6,384]
    ck = torch.load(args.ctx, map_location=dev)
    crop = ck.get("args", {}).get("crop", "center")
    mk = lambda split, sh: DataLoader(
        Gaming500Dataset(args.data, seq_len=args.seq, img_size=126,
                         stride=args.seq // 2, crop_mode=crop, periph=True,
                         split=split, holdout_frac=0.1),
        batch_size=args.bs, shuffle=sh, drop_last=sh, num_workers=args.workers,
        pin_memory=True, persistent_workers=True)
    dl, dl_ev = mk("train", True), mk("holdout", False)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    ctx = ContextTower(n_msg=N_MSG).to(dev).bfloat16().eval()
    ctx.load_state_dict(ck["model"])
    for q in ctx.parameters():
        q.requires_grad_(False)
    model = ActionTower(horizon=H, n_cmd=emb.shape[1]).to(dev).bfloat16()
    model.init_from(ctx)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / args.warmup, 0.5 * (1 + torch.cos(torch.tensor(
            min(s / args.steps, 1.0) * 3.14159)).item())))
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"[C1] {len(dl.dataset)}/{len(dl_ev.dataset)} windows", flush=True)

    step, t0, it = 0, time.time(), iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        lat_now, states, z1, _ = prep(batch, dino, ctx, 1, sgen, dev)
        cmd = emb[sample_cmd(z1, cgen).to(dev)]
        B = z1.shape[0]
        eps = torch.randn_like(z1)
        tau = torch.rand(B, device=dev, dtype=z1.dtype)
        x_tau = (1 - tau[:, None, None]) * eps + tau[:, None, None] * z1
        v = model(lat_now, x_tau, tau, seed=states, cmd=cmd)
        loss = F.mse_loss(v, (z1 - eps))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "loss": round(loss.item(), 5),
                   "gnorm": round(float(gn), 3),
                   "sps": round(step / (time.time() - t0), 3)}
            print(f"[C1] {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ev, n = 0.0, 0
            gen = torch.Generator(dev).manual_seed(1234)
            egen = torch.Generator().manual_seed(555)
            with torch.no_grad():
                for b in dl_ev:
                    lat_now, states, z1, _ = prep(b, dino, ctx, 1,
                                                  torch.Generator().manual_seed(777),
                                                  dev)
                    cmd = emb[sample_cmd(z1, egen).to(dev)]
                    pred = model.sample(lat_now, seed=states, steps=4,
                                        generator=gen, cmd=cmd)
                    ev += F.mse_loss(pred.float(), z1.float()).item()
                    n += 1
                    if n >= 40:
                        break
            model.train()
            rec = {"step": step, "eval_mse": round(ev / max(n, 1), 5)}
            print(f"[C1] EVAL {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            torch.save({"model": model.state_dict(), "step": step,
                        "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))
    print(f"[C1] done → {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
