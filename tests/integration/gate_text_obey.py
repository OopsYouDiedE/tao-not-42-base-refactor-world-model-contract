#!/usr/bin/env python3
"""看向指令服从 gate(S6 式,Minecraft 快头版):命令不同看向,量学到的视角偏置。

判据先登记(看向三指令 down/left/right,held-out 起点 wall_z=8,确定性读出):
  · yaw 服从   = 净yaw(right) − 净yaw(left) ≥ YAW_TH(命令右比命令左更向右转,度);
  · pitch 服从 = 净pitch(down) − ½[净pitch(left)+净pitch(right)] ≥ PITCH_TH(down 更朝下,度)。
读出 = **期望相机动作**逐步累计(Σ softmax·bin值,非采样,低方差):贪心取模会塌到
中心bin(净0)测不出偏置,故用期望值揭示学到的方向偏。同时对比 text-inert 基线应≈0。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. python tests/integration/gate_text_obey.py \
      --ckpt runs/rest_look/r1/final.pt --instrs down,left,right --wall_z 8 --episodes 4 \
      --out runs/rest_look/r1_gate.json
"""
import argparse
import json
import os

import numpy as np
import torch

from net.bc import BCConfig
from net.config import BackboneConfig
from net.bc import TextCondPolicy
from tests.integration.collect_s8 import build_c2_course
from tests.integration.test_utils import crop128
from train.fovea_twotower.gate_fasthead import bin_to_camera
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, CAMERA_SCALE, N_MOUSE

PX2DEG = 0.15
YAW_TH = 8.0
PITCH_TH = 6.0
BIN_VALS = None                                     # 惰性:每个 bin 的归一相机值


def expected_camera(cam_logits):
    """Σ softmax(logits)·bin值 → 两轴期望归一相机 [yaw,pitch](∈[-1,1])。"""
    global BIN_VALS
    if BIN_VALS is None:
        BIN_VALS = torch.tensor([float(bin_to_camera(torch.tensor(b)))
                                 for b in range(CAMERA_BINS)])
    p = torch.softmax(cam_logits.float(), -1).cpu()      # [2,bins]
    return (p * BIN_VALS).sum(-1)                        # [2]


@torch.no_grad()
def roll_expected(policy, env, noop, te, wall_z, steps, max_len, settle, device):
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": build_c2_course(wall_z, pickaxe=True)})
    for _ in range(settle):
        obs, *_ = env.step(noop)
    feats_hist, act_hist = [], []
    prev_vec = np.zeros(ACTION_DIM, np.float32)
    net_yaw = net_pitch = 0.0
    for t in range(steps):
        f = policy.encode_frames(crop128(obs["rgb"]).to(device))[:, 0]
        feats_hist.append(f); act_hist.append(torch.from_numpy(prev_vec).to(device).view(1, -1))
        fseq = torch.stack(feats_hist[-max_len:], 1); aseq = torch.stack(act_hist[-max_len:], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cam_logits, key_logits = policy(fseq.float(), aseq.float(), te)
        ecam = expected_camera(cam_logits[0, -1])        # [2] 归一
        yaw_deg = float(ecam[0]) * CAMERA_SCALE * PX2DEG
        pitch_deg = float(ecam[1]) * CAMERA_SCALE * PX2DEG
        net_yaw += yaw_deg; net_pitch += pitch_deg
        a = dict(noop)
        a["camera_yaw"] = yaw_deg; a["camera_pitch"] = pitch_deg
        on = (key_logits[0, -1].float().sigmoid() > 0.5).cpu().numpy()
        from tests.integration.collect_s8 import V2_KEYS
        for i, name in enumerate(V2_KEYS):
            if name in a:
                a[name] = bool(on[i])
        prev_vec = np.zeros(ACTION_DIM, np.float32); prev_vec[N_MOUSE:] = on.astype(np.float32)
        obs, *_ = env.step(a)
    return net_yaw, net_pitch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--instr_emb", default="runs/ftt_instr/instr_emb.pt")
    p.add_argument("--instrs", default="down,left,right")
    p.add_argument("--wall_z", type=int, default=8, help="held-out 起点")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--port", type=int, default=8581)
    p.add_argument("--out", default="runs/rest_look/gate.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ie = torch.load(args.instr_emb)
    id2idx = {i: k for k, i in enumerate(ie["ids"])}
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cs = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=cs.get("d", 384),
                   heads=cs.get("heads", 6), layers=cs.get("layers", 4), dropout=0.0,
                   max_len=max(128, args.max_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = TextCondPolicy(cfg).to(device).eval()
    sd = ck.get("policy", ck.get("model", ck))
    if any(k.startswith("text_embed.") for k in sd):
        policy.load_state_dict(sd, strict=False)
    else:
        policy.load_c2bc(args.ckpt, device)
    print(f"✅ 载入 {args.ckpt}", flush=True)

    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        mined_stat_keys=["iron_ore"], initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=args.port, verbose=False)
    noop = no_op_v2(); env.reset()

    res = {"ckpt": args.ckpt, "wall_z": args.wall_z, "by_instr": {}}
    for instr in args.instrs.split(","):
        te = ie["emb"][id2idx[instr]].view(1, -1).to(device)
        ys, ps = [], []
        for ep in range(args.episodes):
            ny, npi = roll_expected(policy, env, noop, te, args.wall_z, args.steps,
                                    args.max_len, args.settle, device)
            ys.append(ny); ps.append(npi)
        res["by_instr"][instr] = {"net_yaw_deg": round(float(np.mean(ys)), 2),
                                  "net_pitch_deg": round(float(np.mean(ps)), 2)}
        print(f"[obey] {instr}: net_yaw={np.mean(ys):+.2f} net_pitch={np.mean(ps):+.2f}", flush=True)
    env.close()

    bi = res["by_instr"]
    yaw_delta = bi["right"]["net_yaw_deg"] - bi["left"]["net_yaw_deg"]
    pitch_delta = bi["down"]["net_pitch_deg"] - 0.5 * (bi["left"]["net_pitch_deg"] + bi["right"]["net_pitch_deg"])
    res["yaw_delta"] = round(yaw_delta, 2); res["pitch_delta"] = round(pitch_delta, 2)
    res["verdict_yaw"] = f"{'PASS' if yaw_delta >= YAW_TH else 'FAIL'} (右−左 net_yaw={yaw_delta:+.2f} 门 {YAW_TH})"
    res["verdict_pitch"] = f"{'PASS' if pitch_delta >= PITCH_TH else 'FAIL'} (down−均 net_pitch={pitch_delta:+.2f} 门 {PITCH_TH})"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"yaw_delta": res["yaw_delta"], "pitch_delta": res["pitch_delta"],
                      "verdict_yaw": res["verdict_yaw"], "verdict_pitch": res["verdict_pitch"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
