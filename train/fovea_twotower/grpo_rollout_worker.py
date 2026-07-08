#!/usr/bin/env python3
"""GRPO-R1 rollout worker:M-IRON 长程任务单环境采样(同 seed 组,温度采样)。

记录:逐步 goal 相对 token [T,K,8] / prev [T,22] / 动作(cam bins[T,2],keys[T,20])
/ macro 掩码(闩锁步不进策略损失) + 过程事件(scorer schema)。
奖励侧可用 GT(位置格覆盖=探索度;RL 惯例,策略侧仍零特权)。
"""
import argparse
import json
import time

import numpy as np
import torch

from net.fovea_twotower.token_stream import CLASSES, TokenHead, as_hwc, goal_relative
from tests.integration.collect_calib640 import _pose, _ray
from tests.integration.fullloop_chain import ITEM2CLS, SlowBrain, env_inventory
from train.fovea_twotower.eval_track_cmd import StudentPolicy, CAM_NORM_PX, DEG_PER_PX
from train.minecraft.vpt_action import bin_to_camera
from tests.integration.collect_s8 import V2_KEYS

IRON_ITEMS = {"raw_iron", "iron_ore", "iron_ingot", "deepslate_iron_ore"}
MS_ITEM = [("log", "log"), ("planks", "planks"), ("wooden_pickaxe", "wooden_pickaxe"),
           ("cobblestone", "cobblestone"), ("stone_pickaxe", "stone_pickaxe")]


def sample_step(student, tok_rel, noop, temp, rng):
    """温度采样版 StudentPolicy.__call__(共享上下文窗口)。"""
    student.toks.append(tok_rel)
    if not student.prevs:
        student.prevs.append(np.zeros(22, np.float32))
    toks = torch.from_numpy(np.stack(student.toks[-student.L:]))[None].cuda()
    prev = torch.from_numpy(np.stack(student.prevs[-student.L:]))[None].cuda()
    g = torch.zeros(1, 1, device="cuda")
    with torch.no_grad():
        cam, key = student.tower(toks.float(), g, prev.float())
    lg = cam[0, -1].float() / max(temp, 1e-3)                  # [2,B]
    cb = torch.multinomial(lg.softmax(-1), 1)[:, 0].cpu()
    kp = (torch.rand_like(key[0, -1].float()) <
          (key[0, -1].float() / max(temp, 1e-3)).sigmoid()).cpu().numpy()
    val = bin_to_camera(cb).numpy()
    dx_px, dy_px = val * CAM_NORM_PX
    a = dict(noop)
    a["camera_yaw"] = float(np.clip(dx_px * DEG_PER_PX, -18, 18))
    a["camera_pitch"] = float(np.clip(dy_px * DEG_PER_PX, -18, 18))
    for i, k in enumerate(V2_KEYS):
        if kp[i]:
            a[k] = True
    row = np.zeros(22, np.float32)
    row[0] = np.clip(dx_px / CAM_NORM_PX, -1, 1)
    row[1] = np.clip(dy_px / CAM_NORM_PX, -1, 1)
    row[2:] = kp
    student.prevs.append(row)
    return a, cb.numpy(), kp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world_seed", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--ckpt", default="runs/trackcmd_bc_v17/best.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head_v5b.pt")
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--adapter", default="runs/reason_delta_lora_v4")
    p.add_argument("--temp", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    tok_head = TokenHead(args.vectors, conv_head=args.conv_head)
    student = StudentPolicy(args.ckpt)
    brain = SlowBrain(args.adapter)
    rng = np.random.default_rng(args.seed)
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.DEFAULT, seed=args.world_seed, request_raycast=True,
        initial_extra_commands=["gamemode survival @p", "difficulty peaceful"])

    # 环境启动看门狗:JVM 5min 拉不起端口→杀本进程 java 子进程重试一次,
    # 再败写空 npz 退出(g0_w2 教训:19min 僵启动烧掉组预算的自愈上限)
    import threading
    import subprocess as _sp
    def _boot():
        h = {}
        def _mk():
            try:
                h["env"] = make(initial_env_config=cfg,
                                action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                                port=args.port, verbose=False)
            except Exception as e:  # noqa
                h["err"] = str(e)
        for att in range(2):
            th = threading.Thread(target=_mk, daemon=True)
            th.start()
            th.join(300)
            if "env" in h:
                return h["env"]
            print(f"[w{args.port}] 启动看门狗触发(att{att}),清理 java 子进程",
                  flush=True)
            _sp.run(["bash", "-c",
                     "for p in $(pgrep -P %d); do grep -aq java /proc/$p/cmdline "
                     "2>/dev/null && kill -9 $p; done" % os.getpid()])
            time.sleep(5)
        return None
    import os
    env = _boot()
    if env is None:
        np.savez_compressed(args.out, n=0, recs=json.dumps([]))
        print(f"[w{args.port}] 环境两次启动失败,空产出退出", flush=True)
        return
    noop = no_op_v2()
    env.reset()
    rollouts = []
    for ep in range(args.episodes):
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": ["gamemode survival @p",
                                                       "difficulty peaceful",
                                                       "clear @p"]})
        for _ in range(10):
            obs, *_ = env.step(noop)
        student.reset()
        inv0 = env_inventory(obs["full"])
        steps_plan, _, _ = brain.plan("raw_iron", inv0 & {"stone_pickaxe", "raw_iron"})
        first = steps_plan[0] if steps_plan else ""
        goal_cls = ITEM2CLS.get(first, "")
        goal_cls = goal_cls if goal_cls in CLASSES else "iron_ore"
        gcls = CLASSES.index(goal_cls)
        rgb = as_hwc(obs["rgb"])
        T = dict(toks=[], prev0=[], cam=[], keys=[], macro=[], vis=[], pose=[],
                 frames=[])
        frame_every = max(args.max_steps // 8, 1)
        ev = dict(inv_events=set(), inv_steps={}, iron_lock_steps=0,
                  declared_goal=first, goal_consistent_steps=0, explored=set(),
                  success=False)
        streak = 0
        for t in range(args.max_steps):
            pose = _pose(obs["full"])
            T["pose"].append([float(pose[0]), float(pose[1]), float(pose[2])])
            ev["explored"].add((int(pose[0]) // 4, int(pose[2]) // 4))
            _xyz, key, dist = _ray(obs["full"])
            if "iron_ore" in key and 0 < dist <= 5.5:      # 挖掘闩锁(不进损失)
                a = dict(noop)
                a["attack"] = True
                if t % 10 == 0:
                    a["forward"] = True
                cb = np.array([5, 5])
                kp = np.zeros(20, bool)
                mac = True
            else:
                toks = tok_head(rgb)
                vis = len(toks) and float(toks[:, 6 + gcls].max()) > 0.4
                streak = streak + 1 if vis else 0
                ev["iron_lock_steps"] = max(ev["iron_lock_steps"], streak)
                if vis:
                    ev["goal_consistent_steps"] += 1
                rel = goal_relative(toks[None], np.array([gcls]))[0]
                a, cb, kp = sample_step(student, rel, noop, args.temp, rng)
                T["toks"].append(rel)
                mac = False
            if not mac:
                T["cam"].append(cb)
                T["keys"].append(kp)
                T["vis"].append(bool(vis))
            T["macro"].append(mac)
            obs, *_ = env.step(a)
            rgb = as_hwc(obs["rgb"])
            if t % frame_every == 0 and len(T["frames"]) < 8:
                T["frames"].append(rgb[::4, ::4].copy())   # 低清联络表帧给判官
            if t % 10 == 0:
                inv = env_inventory(obs["full"])
                for pat, name in MS_ITEM:
                    if any(pat in i for i in inv):
                        if name not in ev["inv_events"]:
                            ev["inv_steps"][name] = t
                        ev["inv_events"].add(name)
                if inv & IRON_ITEMS:
                    ev["success"] = True
                    break
        rollouts.append(dict(
            toks=np.array(T["toks"], np.float32),
            cam=np.array(T["cam"], np.int64),
            keys=np.array(T["keys"], bool),
            vis=np.array(T["vis"], bool),
            pose=np.array(T["pose"], np.float32),
            frames=np.array(T["frames"], np.uint8),
            rec=dict(seed=args.world_seed, inv_events=sorted(ev["inv_events"]),
                     inv_steps=ev["inv_steps"],
                     iron_lock_steps=int(ev["iron_lock_steps"]),
                     declared_goal=ev["declared_goal"],
                     goal_consistent_steps=int(ev["goal_consistent_steps"]),
                     explored_delta=len(ev["explored"]),
                     steps=t + 1, success=bool(ev["success"]))))
        print(f"[w{args.port}] ep{ep} steps={t+1} ms={sorted(ev['inv_events'])} "
              f"lock={ev['iron_lock_steps']} exp={len(ev['explored'])}", flush=True)
    env.close()
    np.savez_compressed(args.out,
                        n=len(rollouts),
                        recs=json.dumps([r["rec"] for r in rollouts]),
                        **{f"toks{i}": r["toks"] for i, r in enumerate(rollouts)},
                        **{f"cam{i}": r["cam"] for i, r in enumerate(rollouts)},
                        **{f"keys{i}": r["keys"] for i, r in enumerate(rollouts)},
                        **{f"vis{i}": r["vis"] for i, r in enumerate(rollouts)},
                        **{f"pose{i}": r["pose"] for i, r in enumerate(rollouts)},
                        **{f"frames{i}": r["frames"] for i, r in enumerate(rollouts)})
    print(f"[w{args.port}] DONE → {args.out}", flush=True)


if __name__ == "__main__":
    main()
