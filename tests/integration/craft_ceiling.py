#!/usr/bin/env python3
"""合成技能终审:学生(视觉BC快塔)能否学会教师的 GUI 合成动作(闭环成功=背包出木板)。

teacher:脚本化 GUI 合成(craft_skill),录 frames+动作为 demo。learned:载 BCPolicy 快塔逐帧驱动。
成功检测=结束时 oak_planks>0。给 5 oak_log(inventory.0),固定布局(先测能否复现固定动作序列)。

用法(ZEROCOPY :1):
  ...craft_ceiling.py --policy teacher --episodes 30 --save_demo runs/data/demo_craft
  ...craft_ceiling.py --policy learned --ckpt runs/fh_craft/best.pt --episodes 16
"""
import argparse
import json
import os

import numpy as np

from tests.integration.collect_s8 import DEG2PX, V2_KEYS, frame_pair
from tests.integration.craft_skill import GuiCursor, craft_from_grid, SLOT_INV0
from tests.integration.skill_ceiling import LearnedFastHead, _np_rgb

N_STEP = 70                            # 一局步数(open + craft ~50 + 余量)


def planks(full):
    return sum(it.count for it in full.inventory
               if it.count > 0 and "plank" in (it.translation_key or ""))


def run(policy_name, episodes, port, seed, out, ckpt, max_len, save_demo):
    import torch
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    zc = os.environ.get("DISPLAY", "") == ":1"
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.ZEROCOPY if zc else ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="craft",
        initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=port, verbose=False)
    noop = no_op_v2()
    obs = {}
    rng = np.random.default_rng(seed)
    learned = LearnedFastHead(ckpt, dev, max_len, False, rng) if policy_name == "learned" else None
    rows = []

    def step(a=None):
        nonlocal obs
        obs = env.step(a or dict(noop))[0]
        return obs

    for ep in range(episodes):
        env.reset(options={"fast_reset": True, "extra_commands": ["clear @p"]})
        for _ in range(3):
            step()
        env.reset(options={"fast_reset": True,
                           "extra_commands": ["item replace entity @p inventory.0 with minecraft:oak_log 5"]})
        for _ in range(10):
            step()
        frames, dxs, dys, keys_l = [], [], [], []
        rec = save_demo is not None

        def rstep(a=None):                                # 录制包装
            a = a or dict(noop)
            if rec:
                frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])
                dxs.append(float(a.get("camera_yaw", 0.0)) * DEG2PX)
                dys.append(float(a.get("camera_pitch", 0.0)) * DEG2PX)
                keys_l.append([float(bool(a.get(k, False))) for k in V2_KEYS])
            return step(a)

        if policy_name == "teacher":
            rstep(dict(noop, inventory=True)); rstep(); rstep()   # 开背包
            craft_from_grid(GuiCursor(env, noop, rstep), SLOT_INV0)
            for _ in range(6):
                rstep()
        else:                                             # learned:快塔逐帧驱动
            for t in range(N_STEP):
                a = learned(t, noop, obs)
                step(a)
        got = planks(obs["full"])
        ok = got > 0
        if rec and ok:
            frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])
            os.makedirs(save_demo, exist_ok=True)
            np.savez_compressed(os.path.join(save_demo, f"craft_ep{ep:03d}.npz"),
                                frames=np.stack(frames).astype(np.uint8),
                                dx=np.array(dxs, np.float32), dy=np.array(dys, np.float32),
                                keys=np.array(keys_l, np.uint8), gui=np.ones(len(dxs), np.uint8),
                                score=np.float32(got), policy_strong=np.int64(1),
                                start_hard=np.int64(0))
        rows.append({"ep": ep, "planks": int(got), "success": bool(ok)})
        print(f"[craft/{policy_name}] ep{ep} planks={got} success={ok}", flush=True)
    env.close()
    rate = float(np.mean([r["success"] for r in rows]))
    res = {"policy": policy_name, "episodes": episodes, "success_rate": round(rate, 4),
           "mean_planks": round(float(np.mean([r["planks"] for r in rows])), 2), "rows": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[craft/{policy_name}] SUCCESS_RATE={rate:.3f} → {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="teacher")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--port", type=int, default=9270)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_demo", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or f"runs/ceiling/craft_{args.policy}.json"
    run(args.policy, args.episodes, args.port, args.seed, out, args.ckpt, args.max_len, args.save_demo)


if __name__ == "__main__":
    main()
