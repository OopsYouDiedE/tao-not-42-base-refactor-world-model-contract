#!/usr/bin/env python3
"""同域 BC 一票否决实验·后端:C2 房间里 rollout 快头 BC,测技能表达。

预登记判据(先于结果,见 encode_c2_feats.py 头注):易起点(带镐)N 局中
score>0(挖到并捡到铁)占比 ≥15% = 执行接口 PASS;恒 0 = 接口 FAIL
(冻结 CLS/动作解码表达不了该技能,加数据无用)。
次级诊断:mined_iron>0 占比(拆"挖得断"与"捡得到")。

环境与 collect_s8 同款(superflat 定 seed + raycast + C2 参数化课程),
episode 轮换 WALL_Z_VARIANTS 的易起点;动作解码同 gate_fasthead(bin 采样→V2)。

用法(CPU 渲染 + GPU dinov3):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. python tests/integration/gate_c2.py \
      --ckpt runs/ftt_c2bc/best.pt --episodes 20 --steps 220 --out runs/ftt_c2bc_gate.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from net.bc import BCConfig, build_bc_policy
from net.config import BackboneConfig
from tests.integration.collect_s8 import (WALL_Z_VARIANTS, build_c2_course,
                                          score_c2, _mined_iron)
from train.fovea_twotower.gate_fasthead import decode_action
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE


def crop128(rgb):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    h, w = arr.shape[:2]
    s = min(h, w)
    crop = arr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    im = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im.transpose(2, 0, 1)).float().view(1, 1, 3, 128, 128) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ftt_c2bc/best.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--greedy", action="store_true", default=False)
    p.add_argument("--port", type=int, default=8565)
    p.add_argument("--out", default="runs/ftt_c2bc_gate.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_saved = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"),
                   d=cfg_saved.get("d", 384), heads=cfg_saved.get("heads", 6),
                   layers=cfg_saved.get("layers", 4), dropout=0.0,
                   max_len=max(128, args.max_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = build_bc_policy(cfg).to(device).eval()      # 真 dinov3
    missing, unexpected = policy.load_state_dict(ck["policy"], strict=False)
    assert not [m for m in missing if not m.startswith("backbone.")], missing[:6]
    assert not unexpected, unexpected[:6]
    print(f"✅ 载入 {args.ckpt}(step={ck.get('step')})", flush=True)

    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    cfg_env = InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        mined_stat_keys=["iron_ore"],
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg_env,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    rng = np.random.default_rng(args.seed)

    results = []
    with torch.no_grad():
        for ep in range(args.episodes):
            wall_z = WALL_Z_VARIANTS[ep % len(WALL_Z_VARIANTS)]
            obs, _ = env.reset(options={"fast_reset": True,
                                        "extra_commands": build_c2_course(wall_z, pickaxe=True)})
            for _ in range(args.settle):
                obs, *_ = env.step(noop)
            feats_hist, act_hist = [], []
            prev_vec = np.zeros(ACTION_DIM, np.float32)
            for t in range(args.steps):
                f = policy.encode_frames(crop128(obs["rgb"]).to(device))[:, 0]
                feats_hist.append(f)
                act_hist.append(torch.from_numpy(prev_vec).to(device).view(1, -1))
                fseq = torch.stack(feats_hist[-args.max_len:], 1)
                aseq = torch.stack(act_hist[-args.max_len:], 1)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    cam_logits, key_logits = policy(fseq.float(), aseq.float())
                a, key_on = decode_action(cam_logits[0, -1], key_logits[0, -1],
                                          noop, args.greedy, rng)
                prev_vec = np.zeros(ACTION_DIM, np.float32)
                prev_vec[N_MOUSE:] = key_on              # 相机 prev 置 0(键主导记忆,同 gate_fasthead)
                obs, *_ = env.step(a)
            sc = score_c2(obs["full"])
            mined = _mined_iron(obs["full"])
            results.append({"ep": ep, "wall_z": wall_z, "score": sc, "mined_iron": mined})
            print(f"[gate_c2] ep{ep} wall_z={wall_z} score={sc} mined={mined}", flush=True)
    env.close()

    scores = np.array([r["score"] for r in results])
    mineds = np.array([r["mined_iron"] for r in results])
    # mined_statistics 跨 episode 累计:差分出本局破坏数
    mined_per_ep = np.diff(np.concatenate([[0], mineds]))
    pickup_rate = float((scores > 0).mean())
    mine_rate = float((mined_per_ep > 0).mean())
    verdict = "PASS" if pickup_rate >= 0.15 else "FAIL"
    res = {"ckpt": args.ckpt, "ckpt_step": ck.get("step"), "episodes": args.episodes,
           "steps": args.steps, "greedy": args.greedy,
           "pickup_rate": round(pickup_rate, 4), "mine_rate": round(mine_rate, 4),
           "mean_score": round(float(scores.mean()), 3), "per_episode": results,
           "verdict_interface": f"{verdict} (score>0 率={pickup_rate:.3f} 门 0.15;"
                                f"老师 0.42;破坏率={mine_rate:.3f})"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("pickup_rate", "mine_rate", "mean_score",
                                          "verdict_interface")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
