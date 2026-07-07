#!/usr/bin/env python3
"""Y1 前端对决:YOLOE 校准语义 token vs DINO-CLS vs DINO-patch(跟踪能力横向对照)。

命题(用户 2026-07-08):YOLOE 能力是否显著有效改善快塔跟踪?
已有纵向剂量-响应证据(感知每升级跟踪随动),缺横向对照——同示范、同反应式策略干,
三臂只换视觉前端:
  yoloe : conv-token [K,8] goal相对(现行,校准语义+几何)
  dinoc : DINO-CLS 全局 384d + goal onehot(历史快头前端,闭环挖铁0.25的那个)
  dinop : DINO patch 16×(224/16)² token(空间无命名——把"校准语义"从"空间结构"
          里剥出来单独计价的最锋利对照)
判据预登记:yoloe 追踪中位误差 ≤0.7×最强DINO臂 且 锁定率≥+0.2;n=20 固定课程;
  若 dinop 打平 → 如实记"价值在空间结构非语义校准"。
数据:trackcmd_v13(+_frames,stride2 对齐):token[2i]/frame[i]/act[2i]/goal[2i]。

用法: --mode train --arm yoloe|dinoc|dinop ; --mode eval --arm ...
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fovea_twotower.token_stream import CLASSES, TokenHead, as_hwc, goal_relative
from train.minecraft.vpt_action import CAMERA_BINS, bin_to_camera, camera_to_bin

CAM_NORM = 120.0
D = 192


class DinoFront:
    """冻结 DINO 前端:frame [360,640,3]u8 → CLS[384] 或 patch[196,384]。"""

    def __init__(self, dev="cuda"):
        from net.backbone import build_backbone
        from net.config import BackboneConfig
        self.m = build_backbone(BackboneConfig(kind="dinov2"))[0].to(dev).eval()
        self.dev = dev
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frames_u8, patches=False):
        # frames_u8 [B,360,640,3] → resize 224² → CLS / patch
        x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(self.dev).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.m(pixel_values=x).last_hidden_state          # [B,1+N,384]
        return out[:, 1:] if patches else out[:, 0]


class FrontPolicy(nn.Module):
    """反应式:前端表征 → (slot注意 or 投影) ⊕ prev_a ⊕ goal → MLP → bins+keys。"""

    def __init__(self, arm, n_keys=20):
        super().__init__()
        self.arm = arm
        if arm == "yoloe":
            self.slot = nn.Linear(8, D)
            self.q = nn.Parameter(torch.randn(D) * 0.02)
            cin = D
        elif arm == "dinop":
            self.slot = nn.Linear(384, D)
            self.q = nn.Parameter(torch.randn(D) * 0.02)
            cin = D
        else:                                   # dinoc
            self.proj = nn.Linear(384, D)
            cin = D
        cin += 22 + len(CLASSES)                # prev_a + goal onehot
        self.mlp = nn.Sequential(nn.Linear(cin, D), nn.GELU(),
                                 nn.Linear(D, D), nn.GELU())
        self.cam = nn.Linear(D, 2 * CAMERA_BINS)
        self.key = nn.Linear(D, n_keys)

    def forward(self, feat, prev_a, goal1h):
        # feat: yoloe [B,T,K,8] | dinop [B,T,196,384] | dinoc [B,T,384]
        if self.arm in ("yoloe", "dinop"):
            s = self.slot(feat)
            att = (s @ self.q).softmax(-1)
            rep = (att[..., None] * s).sum(-2)
        else:
            rep = self.proj(feat)
        h = self.mlp(torch.cat([rep, prev_a, goal1h], -1))
        return self.cam(h).view(*h.shape[:2], 2, CAMERA_BINS), self.key(h)


def load_pair(tok_fp, frame_dir):
    z = np.load(tok_fp, allow_pickle=True)
    fp2 = os.path.join(frame_dir, os.path.basename(tok_fp))
    if not os.path.exists(fp2):
        return None
    zf = np.load(fp2, allow_pickle=True)
    n = min(len(zf["frames"]), z["tokens"].shape[0] // 2)
    ii = np.arange(n) * 2                              # 对齐偶数步
    toks = z["tokens"][ii].astype(np.float32)
    goal = z["goal_idx"][ii]
    rel = goal_relative(toks, goal)
    act = np.zeros((n, 22), np.float32)
    act[:, 0] = np.clip(z["dx"][ii] / CAM_NORM, -1, 1)
    act[:, 1] = np.clip(z["dy"][ii] / CAM_NORM, -1, 1)
    act[:, 2:] = z["keys"][ii]
    frames = zf["frames"][:n].transpose(0, 2, 3, 1)     # [n,360,640,3]
    g1h = np.eye(len(CLASSES), dtype=np.float32)[goal]
    return rel, frames, act, g1h


def run_train(args):
    dev = "cuda"
    toks_files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    pairs = [p for p in (load_pair(f, args.data + "_frames") for f in toks_files)
             if p is not None][:args.max_eps]
    print(f"[fd:{args.arm}] {len(pairs)} eps 对齐", flush=True)
    dino = DinoFront(dev) if args.arm.startswith("dino") else None
    # 预计算 DINO 特征(冻结,缓存显存外)
    feats = []
    for rel, frames, act, g1h in pairs:
        if args.arm == "yoloe":
            feats.append(torch.from_numpy(rel))
        else:
            out = []
            for s in range(0, len(frames), 32):
                out.append(dino(frames[s:s + 32], patches=(args.arm == "dinop")).cpu())
            feats.append(torch.cat(out).float())
    pol = FrontPolicy(args.arm).to(dev)
    print(f"[fd:{args.arm}] {sum(p.numel() for p in pol.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    allb = torch.cat([camera_to_bin(torch.from_numpy(p[2][:, :2])) for p in pairs])
    cnt = torch.bincount(allb.flatten(), minlength=CAMERA_BINS).float()
    w = ((cnt.sum() / (cnt + 1)).sqrt().clamp(max=3.0)).to(dev)
    for step in range(args.steps):
        bi = rng.choice(len(pairs), 8)
        Tm = min(feats[i].shape[0] for i in bi)
        fe = torch.stack([feats[i][:Tm] for i in bi]).to(dev)
        act = torch.stack([torch.from_numpy(pairs[i][2][:Tm]) for i in bi]).to(dev)
        g1 = torch.stack([torch.from_numpy(pairs[i][3][:Tm]) for i in bi]).to(dev)
        prev = torch.zeros_like(act)
        prev[:, 1:] = act[:, :-1]
        if rng.random() < 0.5:
            prev = prev * 0
        cam, key = pol(fe, prev, g1)
        tgt = camera_to_bin(act[..., :2])
        loss = (F.cross_entropy(cam.flatten(0, 2).float(), tgt.flatten(), weight=w)
                + F.binary_cross_entropy_with_logits(key.float(), act[..., 2:]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"[fd:{args.arm}] {step} loss={loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"pol": pol.state_dict(), "arm": args.arm}, args.ckpt)
    print(f"[fd:{args.arm}] saved {args.ckpt}", flush=True)


def run_eval(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from net.fovea_twotower.token_stream import aim_solution
    from tests.integration.collect_calib640 import (WALL_Z_VARIANTS, _pose,
                                                    anchor_gt_blocks,
                                                    build_calib_course,
                                                    sample_offsets)

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    pol = FrontPolicy(ck["arm"]).to(dev).eval()
    pol.load_state_dict(ck["pol"])
    arm = ck["arm"]
    th = TokenHead(conv_head=args.conv_head) if arm == "yoloe" else None
    dino = DinoFront(dev) if arm.startswith("dino") else None
    gcls = CLASSES.index("iron_ore")
    g1h = torch.eye(len(CLASSES), device=dev)[gcls][None, None]

    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    rng = np.random.default_rng(args.seed)
    res = []
    for ep in range(args.episodes):
        offsets = sample_offsets(rng)
        obs, _ = env.reset(options={"fast_reset": True, "extra_commands":
                                    build_calib_course(WALL_Z_VARIANTS[ep % 4], offsets)})
        for _ in range(8):
            obs, *_ = env.step(noop)
        gt, obs = anchor_gt_blocks(env, noop, offsets)
        if gt is None:
            continue
        a0 = dict(noop)
        a0["camera_yaw"] = float(rng.uniform(-40, 40))
        a0["camera_pitch"] = float(rng.uniform(-15, 10))
        obs, *_ = env.step(a0)
        rgb = as_hwc(obs["rgb"])
        prev = np.zeros(22, np.float32)
        errs = []
        for t in range(args.ep_steps):
            with torch.no_grad():
                if arm == "yoloe":
                    toks = th(rgb)
                    fe = torch.from_numpy(goal_relative(toks[None],
                                                        np.array([gcls])))[None].to(dev)
                else:
                    fe = dino(rgb[None], patches=(arm == "dinop"))[None].float()
                pa = torch.from_numpy(prev)[None, None].to(dev)
                cam, key = pol(fe, pa, g1h)
            cb = cam[0, -1].float().argmax(-1).cpu()
            val = bin_to_camera(cb).numpy() * CAM_NORM
            a = dict(noop)
            a["camera_yaw"] = float(np.clip(val[0] * 0.15, -18, 18))
            a["camera_pitch"] = float(np.clip(val[1] * 0.15, -18, 18))
            if torch.sigmoid(key[0, -1, 0]) > 0.5:
                a["forward"] = True
            prev = np.zeros(22, np.float32)
            prev[:2] = np.clip(val / CAM_NORM, -1, 1)
            prev[2] = bool(a.get("forward", False))
            pose = _pose(obs["full"])
            err = min(aim_solution(pose, (b[0] + .5, b[1] + .5, b[2]))[2]
                      for b in gt["iron_ore"])
            errs.append(err)
            obs, *_ = env.step(a)
            rgb = as_hwc(obs["rgb"])
        errs = np.array(errs)
        res.append(dict(err_med=float(np.median(errs[15:])),
                        locked=bool(np.median(errs[-20:]) < 12)))
        print(f"[ep{ep}] err={res[-1]['err_med']:.1f}° lock={'✓' if res[-1]['locked'] else '✗'}",
              flush=True)
    env.close()
    em = np.array([r["err_med"] for r in res])
    lk = np.array([r["locked"] for r in res], float)
    boots = [float(np.median(rng.choice(em, len(em)))) for _ in range(2000)]
    out = dict(arm=arm, n=len(res), err_med=float(np.median(em)),
               err_ci=[float(np.percentile(boots, 2.5)),
                       float(np.percentile(boots, 97.5))],
               lock_rate=float(lk.mean()), episodes=res)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[fd:{arm}] err_med={out['err_med']:.1f}° CI{out['err_ci']} "
          f"lock={out['lock_rate']:.2f} → {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--arm", choices=["yoloe", "dinoc", "dinop"], required=True)
    p.add_argument("--data", default="runs/data/trackcmd_v13")
    p.add_argument("--ckpt", default="")
    p.add_argument("--max_eps", type=int, default=200)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--ep_steps", type=int, default=160)
    p.add_argument("--conv_head", default="runs/g1_conv_head_v4.pt")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--port", type=int, default=8680)
    p.add_argument("--out", default="")
    args = p.parse_args()
    if not args.ckpt:
        args.ckpt = f"runs/frontduel_{args.arm}/best.pt"
    if not args.out:
        args.out = f"runs/frontduel_eval_{args.arm}.json"
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
