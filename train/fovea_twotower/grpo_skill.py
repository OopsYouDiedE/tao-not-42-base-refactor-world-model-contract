#!/usr/bin/env python3
"""GRPO 微调快塔:从 BC 暖启动,用闭环任务奖励(组内相对优势)把技能成功率往教师推。

承教训:奖励=**闭环任务产物**(挖到的目标数),非代理。策略=BCPolicy(冻结 dinov3 CLS +
因果 Transformer,随机采样 cam 分箱 + 键 bernoulli)。GRPO:每组 G 局同技能 rollout,
优势 A=(R−mean)/std(组内),策略梯度 loss=−Σ logπ(a|s)·A(整局同 A)。只训时序头,骨干冻结。
带 checkpoint/resume/日志/周期 greedy 闭环评测(承"模型没保存"教训)。

用法(ZEROCOPY :1):
  DISPLAY=:1 PYTHONPATH=. ./.venv/bin/python train/fovea_twotower/grpo_skill.py \
      --skill mine_iron --init runs/fh_mine_iron/best.pt --iters 40 --group 8 --run_dir runs/grpo_iron
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from net.bc import BCConfig, build_bc_policy
from net.config import BackboneConfig
from tests.integration.collect_s8 import V2_KEYS
from tests.integration.skill_ceiling import (SKILLS, WALL_Z, _np_rgb, build_course, inv_count)
from tests.integration.test_utils import crop128
from train.fovea_twotower.gate_fasthead import PX2DEG, bin_to_camera
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, CAMERA_SCALE, N_MOUSE


@torch.no_grad()
def rollout(policy, env, noop, sk, wall_z, steps, settle, device, greedy, rng):
    """单局 rollout。返回 dict(feats[T,384], prev[T,22], cam_b[T,2], key_on[T,20], reward)。"""
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": build_course(sk["block"], sk["tool"], wall_z)})
    for _ in range(settle):
        obs, *_ = env.step(noop)
    base = inv_count(obs["full"], sk["success"]) if sk["success"] else 0
    feats, prevs, cam_bs, key_ons = [], [], [], []
    prev = np.zeros(ACTION_DIM, np.float32)
    for t in range(steps):
        f = policy.encode_frames(crop128(_np_rgb(obs["rgb"])).to(device))[:, 0]  # [1,384]
        feats.append(f[0].float().cpu())
        prevs.append(torch.from_numpy(prev.copy()))
        fseq = torch.stack(feats, 0)[None].to(device)          # [1,t+1,384]
        aseq = torch.stack(prevs, 0)[None].to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            cam_logits, key_logits = policy(fseq[:, -64:], aseq[:, -64:])
        cl, kl = cam_logits[0, -1].float(), key_logits[0, -1].float()            # [2,bins],[20]
        a = dict(noop)
        cam_b = []
        for ax, name in enumerate(("camera_yaw", "camera_pitch")):
            p = torch.softmax(cl[ax], -1)
            b = int(torch.argmax(p)) if greedy else int(torch.multinomial(p, 1))
            cam_b.append(b)
            a[name] = float(bin_to_camera(torch.tensor(b))) * CAMERA_SCALE * PX2DEG
        kp = torch.sigmoid(kl)
        on = (kp > 0.5) if greedy else (torch.rand_like(kp) < kp)
        for i, name in enumerate(V2_KEYS):
            if name in a:
                a[name] = bool(on[i])
        cam_bs.append(torch.tensor(cam_b)); key_ons.append(on.cpu().float())
        prev = np.zeros(ACTION_DIM, np.float32); prev[N_MOUSE:] = on.cpu().numpy()
        obs, *_ = env.step(a)
        if bool(getattr(obs["full"], "is_dead", False)):
            break
    got = (inv_count(obs["full"], sk["success"]) - base) if sk["success"] else 0
    return {"feats": torch.stack(feats), "prev": torch.stack(prevs).float(),
            "cam_b": torch.stack(cam_bs), "key_on": torch.stack(key_ons),
            "reward": float(got), "success": got > 0}


def grpo_step(policy, opt, group, device):
    """组内相对优势策略梯度。group=list of rollout dict。返回 (loss, mean_R, succ_rate)。"""
    R = np.array([g["reward"] for g in group], np.float32)
    adv = (R - R.mean()) / (R.std() + 1e-4)
    if R.std() < 1e-6:                                         # 全同 → 无梯度,跳过
        return 0.0, float(R.mean()), float(np.mean([g["success"] for g in group]))
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for g, A in zip(group, adv):
        feats = g["feats"][None].to(device)                    # [1,T,384]
        prev = g["prev"][None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            cam_logits, key_logits = policy(feats, prev)        # [1,T,2,bins],[1,T,20]
        T = feats.shape[1]
        camb = g["cam_b"].to(device)                            # [T,2]
        logp_cam = F.log_softmax(cam_logits[0].float(), -1)     # [T,2,bins]
        lp = logp_cam.gather(-1, camb[..., None]).squeeze(-1).sum(-1)   # [T]
        kon = g["key_on"].to(device)                            # [T,20]
        logp_key = -F.binary_cross_entropy_with_logits(
            key_logits[0].float(), kon, reduction="none").sum(-1)      # [T]
        loss = -(lp + logp_key).mean() * float(A) / len(group)
        loss.backward()
        total += float(loss)
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    opt.step()
    return total, float(R.mean()), float(np.mean([g["success"] for g in group]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill", default="mine_iron", choices=list(SKILLS))
    p.add_argument("--init", required=True, help="BC 暖启动 ckpt(BCPolicy)")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval_every", type=int, default=8)
    p.add_argument("--eval_n", type=int, default=12)
    p.add_argument("--port", type=int, default=8990)
    p.add_argument("--run_dir", default="runs/grpo_iron")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    sk = SKILLS[args.skill]

    ck = torch.load(args.init, map_location=dev, weights_only=False)
    cs = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=cs.get("d", 384),
                   heads=cs.get("heads", 6), layers=cs.get("layers", 4), dropout=0.0,
                   max_len=128, action_dim=ACTION_DIM, n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
    policy = build_bc_policy(cfg).to(dev)
    policy.load_state_dict(ck["policy"], strict=False)
    policy.eval()                                              # 骨干冻结;trunk 训练(dropout=0)
    trainable = [q for q in policy.parameters() if q.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)

    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    zc = os.environ.get("DISPLAY", "") == ":1"
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.ZEROCOPY if zc else ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="ceiling", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=args.port, verbose=False)
    noop = no_op_v2(); env.reset()
    logf = open(os.path.join(args.run_dir, "log.jsonl"), "a")
    print(f"[grpo] {args.skill} warmstart={args.init} group={args.group} iters={args.iters}", flush=True)

    def evaluate(n):
        s = [rollout(policy, env, noop, sk, WALL_Z[i % len(WALL_Z)], args.steps,
                     args.settle, dev, True, rng)["success"] for i in range(n)]
        return float(np.mean(s))

    best = evaluate(args.eval_n)
    print(f"[grpo] init greedy success={best:.3f}", flush=True)
    torch.save({"policy": policy.state_dict(), "cfg": vars(cfg), "iter": 0}, f"{args.run_dir}/best.pt")
    t0 = time.time()
    for it in range(1, args.iters + 1):
        group = [rollout(policy, env, noop, sk, WALL_Z[i % len(WALL_Z)], args.steps,
                         args.settle, dev, False, rng) for i in range(args.group)]
        loss, mR, sr = grpo_step(policy, opt, group, dev)
        rec = {"iter": it, "loss": round(loss, 4), "mean_R": round(mR, 3),
               "sample_succ": round(sr, 3), "sps": round(it / (time.time() - t0), 3)}
        print(f"[grpo] {rec}", flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if it % args.eval_every == 0 or it == args.iters:
            g = evaluate(args.eval_n)
            print(f"[grpo] iter{it} greedy_success={g:.3f} (best {best:.3f})", flush=True)
            logf.write(json.dumps({"iter": it, "greedy_success": g}) + "\n"); logf.flush()
            if g >= best:
                best = g
                torch.save({"policy": policy.state_dict(), "cfg": vars(cfg), "iter": it,
                            "greedy_success": g}, f"{args.run_dir}/best.pt")
    env.close()
    print(f"[grpo] done best greedy_success={best:.3f} → {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
