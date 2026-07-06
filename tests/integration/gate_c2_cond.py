#!/usr/bin/env python3
"""命题③决定性测试:return-conditioned 快头,命令不同目标回报跑 C2,看行为是否可控变化。

同一个条件化策略(train_fasthead_cond 训好),分别**命令高回报(如 2)与低回报(0)**在同款 C2
房间 rollout。若命令高回报 → 采矿成功率 / 破坏率显著高于命令低回报 = **"用信号能操纵执行"**
拿到实证(命题③最小存在性证明)。判据先登记:high 命令的 score>0 率 − low 命令的 ≥ +0.15。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. python tests/integration/gate_c2_cond.py \
      --ckpt runs/ftt_c2cond/final.pt --episodes 12 --steps 220 --out runs/ftt_c2cond_gate.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from net.bc import BCConfig
from net.config import BackboneConfig
from tests.integration.collect_s8 import WALL_Z_VARIANTS, build_c2_course, score_c2, _mined_iron
from train.fovea_twotower.gate_fasthead import decode_action
from train.fovea_twotower.train_fasthead_cond import CondPolicy, RET_SCALE
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE


def crop128(rgb):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    h, w = arr.shape[:2]; s = min(h, w)
    crop = arr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    im = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im.transpose(2, 0, 1)).float().view(1, 1, 3, 128, 128) / 255.0


@torch.no_grad()
def run_cell(policy, env, noop, cmd_ret, episodes, steps, max_len, settle, device, rng):
    """给定命令回报 cmd_ret,跑 episodes 局,返回每局 (score, mined_delta)。"""
    ret_t = torch.tensor([cmd_ret / RET_SCALE], device=device).float()
    out = []
    prev_mined = _mined_iron(env.reset()[0]["full"]) if False else 0
    for ep in range(episodes):
        wall_z = WALL_Z_VARIANTS[ep % len(WALL_Z_VARIANTS)]
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_c2_course(wall_z, pickaxe=True)})
        for _ in range(settle):
            obs, *_ = env.step(noop)
        feats_hist, act_hist = [], []
        prev_vec = np.zeros(ACTION_DIM, np.float32)
        for t in range(steps):
            f = policy.encode_frames(crop128(obs["rgb"]).to(device))[:, 0]
            feats_hist.append(f)
            act_hist.append(torch.from_numpy(prev_vec).to(device).view(1, -1))
            fseq = torch.stack(feats_hist[-max_len:], 1)
            aseq = torch.stack(act_hist[-max_len:], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                cam_logits, key_logits = policy(fseq.float(), aseq.float(), ret_t)
            a, key_on = decode_action(cam_logits[0, -1], key_logits[0, -1], noop, False, rng)
            prev_vec = np.zeros(ACTION_DIM, np.float32); prev_vec[N_MOUSE:] = key_on
            obs, *_ = env.step(a)
        mined_now = _mined_iron(obs["full"])
        out.append({"ep": ep, "wall_z": wall_z, "score": score_c2(obs["full"]),
                    "mined_cum": mined_now})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ftt_c2cond/final.pt")
    p.add_argument("--episodes", type=int, default=12, help="每个命令回报的局数")
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--cmds", type=float, nargs="+", default=[2.0, 0.0], help="命令的目标回报(高在前)")
    p.add_argument("--port", type=int, default=8570)
    p.add_argument("--out", default="runs/ftt_c2cond_gate.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cs = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=cs.get("d", 384),
                   heads=cs.get("heads", 6), layers=cs.get("layers", 4), dropout=0.0,
                   max_len=max(128, args.max_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = CondPolicy(cfg).to(device).eval()      # 真 dinov3 骨干
    missing, unexpected = policy.load_state_dict(ck["policy"], strict=False)
    assert not [m for m in missing if not m.startswith("backbone.")], missing[:6]
    assert not unexpected, unexpected[:6]
    print(f"✅ 载入条件化 head {args.ckpt}(step={ck.get('step')})", flush=True)

    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        mined_stat_keys=["iron_ore"], initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=args.port, verbose=False)
    noop = no_op_v2(); env.reset(); rng = np.random.default_rng(args.seed)

    res = {"ckpt": args.ckpt, "episodes": args.episodes, "steps": args.steps, "by_command": {}}
    for cmd in args.cmds:
        rows = run_cell(policy, env, noop, cmd, args.episodes, args.steps,
                        args.max_len, args.settle, device, rng)
        scores = np.array([r["score"] for r in rows])
        mined = np.diff(np.concatenate([[rows[0]["mined_cum"] - 0], [r["mined_cum"] for r in rows]]))
        rate = float((scores > 0).mean())
        res["by_command"][f"{cmd}"] = {"pickup_rate": round(rate, 4),
                                       "mean_score": round(float(scores.mean()), 3),
                                       "per_ep_score": scores.tolist()}
        print(f"[cond] 命令回报={cmd}: score>0率={rate:.3f} 均分={scores.mean():.2f}", flush=True)
    env.close()

    hi, lo = str(args.cmds[0]), str(args.cmds[-1])
    delta = res["by_command"][hi]["pickup_rate"] - res["by_command"][lo]["pickup_rate"]
    res["steer_delta"] = round(delta, 4)
    res["verdict_cond"] = (f"{'PASS' if delta >= 0.15 else 'FAIL'} "
                           f"(命令高{hi}−低{lo} 的 score>0率差={delta:+.3f} 门 +0.15)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"steer_delta": res["steer_delta"], "verdict": res["verdict_cond"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
