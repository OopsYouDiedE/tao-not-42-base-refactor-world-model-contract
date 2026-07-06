#!/usr/bin/env python3
"""ReST/RAFT 组采样:文字指示条件下,同起点温度采样 N 条轨迹 → 存判优素材。

每个 group = (指令, 起点 wall_z),组内 N 条轨迹从**同一房间**采出(superflat 定 seed +
fast_reset,起点逐帧一致,组内差异全来自策略温度采样)。这正是"同时走多条轨迹、用
组内相对优势避免全垃圾起步"(GRPO/ReST 冷启动)的素材层。

每条轨迹落三样:
  ① feats/*.npz  —— DINOv3 CLS feats[T,384] + 动作 act[T,22] + score/mined(ReST BC 回灌用);
  ② frames/<traj>/f*.png —— **灵活关键帧**(事件驱动:开挖/破坏/拾取/矿进准星/大转视 +
     首尾 + 均匀补足,预算封顶);判优 SubAgent 看这些;
  ③ groups.json —— 清单:每 group 下 N 条轨迹的 摘要(数值特征)+ 关键帧路径 + score。

判优在环外(我 spawn SubAgent 读 groups.json + 关键帧排序);本脚本只产素材,不含奖励。

用法(CPU 渲染 + GPU dinov3):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. python tests/integration/rollout_groups.py \
      --ckpt runs/ftt_c2bc/best.pt --instrs mine,still,left,right --wall_z 7,9 \
      --n 6 --steps 200 --temp 1.3 --out runs/rest_r0
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from net.bc import BCConfig
from net.config import BackboneConfig
from tests.integration.collect_s8 import (WALL_Z_VARIANTS, V2_KEYS, build_c2_course,
                                          score_c2, _mined_iron, _raycast)
from train.fovea_twotower.gate_fasthead import bin_to_camera
from train.fovea_twotower.text_cond_policy import TextCondPolicy
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, CAMERA_SCALE, N_MOUSE

PX2DEG = 0.15
I_ATTACK = V2_KEYS.index("attack")      # 7:挥击(挖)
I_FWD = V2_KEYS.index("forward")        # 0:前进键


def crop_square(rgb, size):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    h, w = arr.shape[:2]; s = min(h, w)
    crop = arr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def crop128(rgb):
    im = crop_square(rgb, 128)
    return torch.from_numpy(im.transpose(2, 0, 1)).float().view(1, 1, 3, 128, 128) / 255.0


def decode_temp(cam_logits, key_logits, noop, cam_temp, key_temp, rng):
    """温度采样末步 logits → (V2 动作 dict, act_vec[22])。act_vec=[norm_yaw,norm_pitch,key×20]
    与 encode_c2_feats 的 BC 目标契约一致(相机存归一 bin 值,非度)。
    相机/按键温度解耦:看向指令需相机高温多样(视角随机游走出各方向),挖矿指令需相机
    低温保准星稳(否则漂移无法连破)——单一温度两头难顾,故分开。"""
    a = dict(noop)
    act_vec = np.zeros(ACTION_DIM, np.float32)
    for axis, name in enumerate(("camera_yaw", "camera_pitch")):
        p = torch.softmax(cam_logits[axis].float() / cam_temp, -1).cpu().numpy()
        b = int(rng.choice(CAMERA_BINS, p=p))
        norm = float(bin_to_camera(torch.tensor(b)))
        act_vec[axis] = norm
        a[name] = norm * CAMERA_SCALE * PX2DEG
    prob = (key_logits.float() / key_temp).sigmoid().cpu().numpy()
    on = rng.random(len(prob)) < prob
    for i, name in enumerate(V2_KEYS):
        if name in a:
            a[name] = bool(on[i])
    act_vec[N_MOUSE:] = on.astype(np.float32)
    return a, act_vec, on.astype(np.float32)


def pick_keyframes(trace, budget=7):
    """事件驱动的灵活关键帧选择 → 排序去重的帧索引(≤budget)。
    事件:开挖(attack 上升沿)/破坏(mined↑)/拾取(score↑)/矿进准星(raycast→iron)/
    大转视(净 yaw 累计过 ±25°);首尾恒选;不足预算用均匀补齐;超预算按优先级截。"""
    T = len(trace)
    if T <= budget:
        return list(range(T))
    att = np.array([r["attack"] for r in trace])
    mined = np.array([r["mined"] for r in trace])
    score = np.array([r["score"] for r in trace])
    ore = np.array([1.0 if "iron" in r["ray_key"] else 0.0 for r in trace])
    cyaw = np.cumsum([r["dyaw_deg"] for r in trace])
    pri = []                                    # (优先级, 帧idx)
    pri.append((100, 0)); pri.append((100, T - 1))          # 首尾
    for t in range(1, T):
        if score[t] > score[t - 1]:
            pri.append((90, t))                 # 拾取铁(最强证据)
        if mined[t] > mined[t - 1]:
            pri.append((80, t))                 # 破坏矿
        if ore[t] > ore[t - 1]:
            pri.append((70, t))                 # 矿进准星
        if att[t] > att[t - 1]:
            pri.append((50, t))                 # 开挖
        if abs(cyaw[t]) >= 25 and abs(cyaw[t - 1]) < 25:
            pri.append((60, t))                 # 大转视里程碑
    seen, chosen = set(), []
    for _, t in sorted(pri, key=lambda x: -x[0]):
        if t not in seen:
            seen.add(t); chosen.append(t)
        if len(chosen) >= budget:
            break
    if len(chosen) < budget:                    # 均匀补齐覆盖空档
        for t in np.linspace(0, T - 1, budget).round().astype(int):
            if int(t) not in seen:
                seen.add(int(t)); chosen.append(int(t))
            if len(chosen) >= budget:
                break
    return sorted(chosen)


def summarize(trace, instr_text):
    """数值摘要(判优 SubAgent 的文本判据侧)。"""
    T = len(trace)
    yaw = float(sum(r["dyaw_deg"] for r in trace))
    pitch = float(sum(r["dpitch_deg"] for r in trace))
    swings = int(sum(r["attack"] for r in trace))
    fwd = int(sum(r["fwd"] for r in trace))
    ore_frac = float(np.mean([1.0 if "iron" in r["ray_key"] else 0.0 for r in trace]))
    min_ore = min([r["ray_dist"] for r in trace if "iron" in r["ray_key"]] or [None]) \
        if any("iron" in r["ray_key"] for r in trace) else None
    return {
        "steps": T,
        "net_yaw_deg": round(yaw, 1), "net_pitch_deg": round(pitch, 1),
        "swings": swings, "forward_steps": fwd,
        "ore_in_crosshair_frac": round(ore_frac, 3),
        "min_dist_to_ore": round(min_ore, 2) if min_ore is not None else None,
        "mined_iron": int(trace[-1]["mined"] - trace[0]["mined"]),
        "iron_collected": round(float(trace[-1]["score"] - trace[0]["score"]), 1),
        "instruction": instr_text,
    }


@torch.no_grad()
def roll_one(policy, env, noop, instr_emb, instr_text, wall_z, steps, max_len,
             settle, cam_temp, key_temp, device, rng, frame_size):
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": build_c2_course(wall_z, pickaxe=True)})
    for _ in range(settle):
        obs, *_ = env.step(noop)
    te = instr_emb.view(1, -1).to(device)
    feats_hist, act_hist = [], []
    feats_store, act_store, rgb_store, trace = [], [], [], []
    prev_vec = np.zeros(ACTION_DIM, np.float32)
    prev_yaw = float(getattr(obs["full"], "yaw", 0.0))
    prev_pitch = float(getattr(obs["full"], "pitch", 0.0))
    for t in range(steps):
        rgb = obs["rgb"]
        f = policy.encode_frames(crop128(rgb).to(device))[:, 0]
        feats_hist.append(f); act_hist.append(torch.from_numpy(prev_vec).to(device).view(1, -1))
        fseq = torch.stack(feats_hist[-max_len:], 1); aseq = torch.stack(act_hist[-max_len:], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cam_logits, key_logits = policy(fseq.float(), aseq.float(), te)
        a, act_vec, on = decode_temp(cam_logits[0, -1], key_logits[0, -1], noop,
                                     cam_temp, key_temp, rng)
        feats_store.append(f.float().cpu().numpy()[0].astype(np.float16))
        act_store.append(act_vec.copy())
        rgb_store.append(crop_square(rgb, frame_size))
        full = obs["full"]
        ray_key, ray_dist = _raycast(full)
        yaw_now = float(getattr(full, "yaw", prev_yaw)); pit_now = float(getattr(full, "pitch", prev_pitch))
        trace.append({"attack": float(act_vec[N_MOUSE + I_ATTACK]),
                      "fwd": float(act_vec[N_MOUSE + I_FWD]),
                      "dyaw_deg": ((yaw_now - prev_yaw + 180) % 360) - 180,
                      "dpitch_deg": pit_now - prev_pitch,
                      "ray_key": ray_key, "ray_dist": min(ray_dist, 99.0),
                      "mined": _mined_iron(full), "score": score_c2(full)})
        prev_yaw, prev_pitch = yaw_now, pit_now
        prev_vec = np.zeros(ACTION_DIM, np.float32); prev_vec[N_MOUSE:] = on
        obs, *_ = env.step(a)
    feats = np.stack(feats_store); acts = np.stack(act_store)
    return feats, acts, rgb_store, trace, summarize(trace, instr_text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ftt_c2bc/best.pt")
    p.add_argument("--instr_emb", default="runs/ftt_instr/instr_emb.pt")
    p.add_argument("--instrs", default="mine,still,left,right", help="逗号分隔的指令 id")
    p.add_argument("--wall_z", default="7,9", help="逗号分隔的起点 wall_z")
    p.add_argument("--n", type=int, default=6, help="每 group 轨迹数")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--temp", type=float, default=1.3, help="缺省温度(cam/key 未单设时用)")
    p.add_argument("--cam_temp", type=float, default=None, help="相机采样温度(看向指令调高)")
    p.add_argument("--key_temp", type=float, default=None, help="按键采样温度(挖矿指令保低)")
    p.add_argument("--frame_size", type=int, default=384, help="关键帧 PNG 边长")
    p.add_argument("--kf_budget", type=int, default=7)
    p.add_argument("--port", type=int, default=8575)
    p.add_argument("--out", default="runs/rest_r0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cs = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=cs.get("d", 384),
                   heads=cs.get("heads", 6), layers=cs.get("layers", 4), dropout=0.0,
                   max_len=max(128, args.max_len), action_dim=ACTION_DIM,
                   n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = TextCondPolicy(cfg).to(device).eval()
    # 若 ckpt 已是 text-cond(ReST 后续轮)则直接载;否则从纯挖 c2bc 起手(text_embed 保持零)
    if any(k.startswith("text_embed.") for k in ck.get("policy", {})):
        policy.load_state_dict(ck["policy"], strict=False)
    else:
        policy.load_c2bc(args.ckpt, device)
    print(f"✅ policy 载入 {args.ckpt}", flush=True)

    ie = torch.load(args.instr_emb)
    id2idx = {i: k for k, i in enumerate(ie["ids"])}
    instrs = args.instrs.split(",")
    wall_zs = [int(z) for z in args.wall_z.split(",")]
    for i in instrs:
        assert i in id2idx, f"未知指令 {i};可选 {ie['ids']}"

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

    os.makedirs(os.path.join(args.out, "feats"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    groups = []
    for instr in instrs:
        emb = ie["emb"][id2idx[instr]]
        itext = ie["texts"][id2idx[instr]]
        for wz in wall_zs:
            trajs = []
            for k in range(args.n):
                rng = np.random.default_rng(args.seed * 100000 + hash((instr, wz, k)) % 99991)
                feats, acts, rgbs, trace, summ = roll_one(
                    policy, env, noop, emb, itext, wz, args.steps, args.max_len,
                    args.settle, args.cam_temp or args.temp, args.key_temp or args.temp,
                    device, rng, args.frame_size)
                tid = f"{instr}__wz{wz}__s{k}"
                np.savez_compressed(os.path.join(args.out, "feats", tid + ".npz"),
                                    feats=feats, action=acts.astype(np.float32),
                                    score=np.float32(summ["iron_collected"]),
                                    mined=np.int32(summ["mined_iron"]),
                                    instr=instr, instr_idx=np.int64(id2idx[instr]))
                kf = pick_keyframes(trace, args.kf_budget)
                fdir = os.path.join(args.out, "frames", tid); os.makedirs(fdir, exist_ok=True)
                kf_paths = []
                for idx in kf:
                    fp = os.path.join(fdir, f"f{idx:03d}.png")
                    cv2.imwrite(fp, cv2.cvtColor(rgbs[idx], cv2.COLOR_RGB2BGR))
                    kf_paths.append(fp)
                trajs.append({"traj_id": tid, "instr": instr, "instr_text": itext,
                              "wall_z": wz, "seed_k": k, "npz": tid + ".npz",
                              "keyframes": kf_paths, "keyframe_steps": kf, "summary": summ})
                print(f"  [{tid}] swings={summ['swings']} yaw={summ['net_yaw_deg']} "
                      f"ore_frac={summ['ore_in_crosshair_frac']} mined={summ['mined_iron']} "
                      f"iron={summ['iron_collected']} kf={len(kf)}", flush=True)
            groups.append({"group_id": f"{instr}__wz{wz}", "instr": instr,
                           "instr_text": itext, "wall_z": wz, "trajectories": trajs})
    env.close()
    json.dump({"args": vars(args), "instr_ids": instrs, "wall_zs": wall_zs,
               "groups": groups}, open(os.path.join(args.out, "groups.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"💾 {args.out}/groups.json | {len(groups)} groups × {args.n} 轨", flush=True)


if __name__ == "__main__":
    main()
