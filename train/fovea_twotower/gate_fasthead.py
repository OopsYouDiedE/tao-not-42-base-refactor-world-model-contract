#!/usr/bin/env python3
"""快头 Phase0 gate(P4):把 train_fasthead 训好的 BC 头装回**真 dinov3 骨干**,在 CraftGround
生存世界自回归 rollout,统计成就解锁率。判据(scale-readiness P4):CraftGround 成就 ≥5%。

与训练的关系:训练吃预编码 CLS(骨干占位);gate 必须现场编码(真骨干)才闭环。best.pt 只存
时序头(骨干不入 ckpt),这里 build_bc_policy(真 dinov3)后 strict=False 载入头权重。

动作解码(训练是逆:action[:,:2]=dx/scale,:2 后 20 键):
  相机 bin → bin_to_camera(归一值[-1,1]) → ×CAMERA_SCALE=dx_px → ×0.15=deg(V2 camera_yaw/pitch);
  键 sigmoid>0.5(或采样)→ V2 布尔键(V2_KEYS 名)。
成就:train.craftground.reward 的库存 translation_key 子串规则 + 深度阈值(ALL_ACHIEVEMENTS=36)。

用法(CraftGround CPU 渲染;与 S8 采集错峰,别同时占 CPU):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. python train/fovea_twotower/gate_fasthead.py \
      --ckpt runs/ftt_fasthead/best.pt --episodes 8 --steps 600 --out runs/ftt_fasthead_gate.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from net.bc import BCConfig, build_bc_policy
from net.config import BackboneConfig
from train.craftground.achievements import ALL_ACHIEVEMENTS
from train.craftground.reward import (detect_achievements, extract_inventory_keys,
                                      DEPTH_THRESHOLDS)
from train.gaming500.dataset import KEY_NAMES
from train.minecraft.vpt_action import (ACTION_DIM, CAMERA_BINS, CAMERA_SCALE,
                                        N_MOUSE, bin_to_camera)

# 开局白送的根进度,非真成就(reward.py:story.root 游戏开始即授予)——排除出成就率口径
FREEBIE_ACHIEVEMENTS = {"minecraft.story.root"}
_K2V = {"w": "forward", "a": "left", "s": "back", "d": "right", "space": "jump"}
V2_KEYS = [_K2V.get(k[len("key_"):], k[len("key_"):]) for k in KEY_NAMES]
PX2DEG = 0.15                            # VPT 约定:1px ≈ 0.15deg(collect 的 DEG2PX 逆)


def crop128(rgb):
    """CraftGround rgb(CHW/HWC)→ [1,1,3,128,128] float01(中心裁剪+resize,同 encode 口径)。"""
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    h, w = arr.shape[:2]
    s = min(h, w)
    crop = arr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    im = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(im.transpose(2, 0, 1)).float() / 255.0
    return t.view(1, 1, 3, 128, 128)


def decode_action(cam_logits, key_logits, noop, greedy, rng):
    """策略末步 logits → V2 动作 dict。cam_logits [2,bins], key_logits [20]。"""
    a = dict(noop)
    for axis, name in enumerate(("camera_yaw", "camera_pitch")):
        logit = cam_logits[axis]
        if greedy:
            b = int(logit.argmax())
        else:
            p = torch.softmax(logit.float(), -1).cpu().numpy()
            b = int(rng.choice(CAMERA_BINS, p=p))
        norm = float(bin_to_camera(torch.tensor(b)))
        a[name] = norm * CAMERA_SCALE * PX2DEG
    prob = key_logits.float().sigmoid().cpu().numpy()
    on = prob > 0.5 if greedy else rng.random(len(prob)) < prob
    for i, name in enumerate(V2_KEYS):
        if name in a:
            a[name] = bool(on[i])
    return a, on.astype(np.float32)


@torch.no_grad()
def rollout(policy, env, noop, steps, max_len, device, greedy, rng):
    """单 episode AR rollout。返回(解锁成就集, 步数)。"""
    obs, _ = env.reset()
    feats_hist, act_hist = [], []
    unlocked = set()
    prev_vec = np.zeros(ACTION_DIM, np.float32)
    for t in range(steps):
        img = crop128(obs["rgb"]).to(device)
        f = policy.encode_frames(img)[:, 0]            # [1, enc_dim]
        feats_hist.append(f)
        act_hist.append(torch.from_numpy(prev_vec).to(device).view(1, -1))
        fseq = torch.stack(feats_hist[-max_len:], 1)   # [1,L,enc]
        aseq = torch.stack(act_hist[-max_len:], 1)     # [1,L,A]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cam_logits, key_logits = policy(fseq.float(), aseq.float())
        a, key_on = decode_action(cam_logits[0, -1], key_logits[0, -1], noop, greedy, rng)
        # 记录本步动作作下步 prev(归一相机 + 键;相机用命令值反算归一,近似即可置 0 让键主导记忆)
        prev_vec = np.zeros(ACTION_DIM, np.float32)
        prev_vec[N_MOUSE:] = key_on
        obs, *_ = env.step(a)
        full = obs["full"]
        keys = extract_inventory_keys(full)
        unlocked |= detect_achievements(keys, t)
        y = float(getattr(full, "y", 0.0))
        for name, thr in DEPTH_THRESHOLDS:
            if y < thr:
                unlocked.add(name)
    return unlocked, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ftt_fasthead/best.pt")
    p.add_argument("--backbone", choices=["dinov3", "dinov2"], default="dinov3")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--greedy", action="store_true", default=False,
                   help="贪心解码(默认按 logits 采样,利探索)")
    p.add_argument("--port", type=int, default=8560)
    p.add_argument("--out", default="runs/ftt_fasthead_gate.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_saved = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind=args.backbone),
                   d=cfg_saved.get("d", 384), heads=cfg_saved.get("heads", 6),
                   layers=cfg_saved.get("layers", 4), dropout=0.0,
                   max_len=max(128, args.max_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = build_bc_policy(cfg).to(device).eval()     # 真骨干(dinov3)
    missing, unexpected = policy.load_state_dict(ck["policy"], strict=False)
    head_missing = [m for m in missing if not m.startswith("backbone.")]
    assert not head_missing, f"头权重缺失: {head_missing[:6]}"
    assert not unexpected, f"未知权重: {unexpected[:6]}"
    print(f"✅ 载入头 {args.ckpt}(step={ck.get('step')});真 {args.backbone} 骨干")

    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    cfg_env = InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg_env,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    rng = np.random.default_rng(args.seed)

    per_ep = []           # 排除白送根进度后的真成就
    all_unlocked = set()
    for ep in range(args.episodes):
        unlocked, n = rollout(policy, env, noop, args.steps, args.max_len,
                              device, args.greedy, rng)
        real = sorted(unlocked - FREEBIE_ACHIEVEMENTS)
        per_ep.append(real)
        all_unlocked |= set(real)
        print(f"[gate] ep{ep}: {len(real)} 真成就 {real} "
              f"(+白送 {sorted(unlocked & FREEBIE_ACHIEVEMENTS)})", flush=True)
    env.close()

    # 成就率口径(排除白送 story.root):①任一真成就的 episode 占比;②平均每 episode 真成就数;③种类占比
    hit_rate = float(np.mean([len(u) > 0 for u in per_ep]))
    mean_ach = float(np.mean([len(u) for u in per_ep]))
    variety = len(all_unlocked) / len(ALL_ACHIEVEMENTS)
    verdict = "PASS" if hit_rate >= 0.05 else "FAIL"
    res = {"ckpt": args.ckpt, "ckpt_step": ck.get("step"), "episodes": args.episodes,
           "steps": args.steps, "greedy": args.greedy, "freebies_excluded": sorted(FREEBIE_ACHIEVEMENTS),
           "episode_hit_rate": round(hit_rate, 4), "mean_achievements": round(mean_ach, 3),
           "variety_unlocked": round(variety, 4), "all_unlocked": sorted(all_unlocked),
           "per_episode": per_ep,
           "verdict_p4": f"{verdict} (真成就 episode_hit_rate={hit_rate:.3f} 门 0.05;白送 story.root 已排除)"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("episode_hit_rate", "mean_achievements",
                                          "variety_unlocked", "all_unlocked", "verdict_p4")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
