#!/usr/bin/env python3
"""多步复合技能终审(同存档):教师在固定 seed 世界做"挖铁→合木板"多步任务,学生从同存档学 + eval。

同存档 = 固定 seed + 确定性命令(craftground 禁存盘,可复现靠重新生成,见 knowledge)。教师+学生用**同一**
SCENE_CMDS,世界状态逐块相同。多动作类型:移动(forward)/瞄准(camera)/挖掘(attack raycast)/GUI(背包+光标+
shift-click)。子任务成功各自判定:iron(挖到) + planks(合成)。

复用已验证原语:MineForwardPolicy(挖矿)、craft_from_grid(GUI合成)、LearnedFastHead(学生)。

用法(ZEROCOPY :1):
  ...multiskill_ceiling.py --policy teacher --episodes 30 --save_demo runs/data/demo_multiskill
  ...multiskill_ceiling.py --policy learned --ckpt runs/fh_multiskill/best.pt --episodes 16
"""
import argparse
import json
import os

import numpy as np

from tests.integration.collect_s8 import DEG2PX, V2_KEYS, MineForwardPolicy, frame_pair
from tests.integration.craft_skill import GuiCursor, craft_from_grid, SLOT_INV0
from tests.integration.skill_ceiling import LearnedFastHead, _np_rgb

# ── 同存档:固定 seed + 确定性场景(铁矿墙 + 石镐 + 主格 oak_log) ──
SCENE = ["gamemode survival @p", "difficulty peaceful", "gamerule doMobSpawning false",
         "weather clear", "gamerule doWeatherCycle false", "time set 6000",
         "gamerule doDaylightCycle false", "kill @e[type=!player]", "tp @p ~ ~ ~ 0 0",
         "fill ~-3 ~-1 ~-2 ~3 ~4 ~5 minecraft:air",
         "fill ~-3 ~-2 ~-2 ~3 ~-2 ~5 minecraft:stone",
         "fill ~-3 ~-1 ~5 ~3 ~4 ~5 minecraft:stone",
         "setblock ~0 ~ ~5 minecraft:iron_ore", "setblock ~1 ~ ~5 minecraft:iron_ore",
         "setblock ~-1 ~ ~5 minecraft:iron_ore", "clear @p",
         "item replace entity @p weapon.mainhand with minecraft:stone_pickaxe 1",
         "item replace entity @p inventory.0 with minecraft:oak_log 5"]
MINE_STEPS = 110                          # 挖矿阶段步数


def inv_count(full, kw):
    return sum(it.count for it in full.inventory
              if it.count > 0 and any(k in (it.translation_key or "") for k in kw))


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
        world_type=WorldType.SUPERFLAT, seed="multiskill", request_raycast=True,
        render_distance=4, initial_extra_commands=["gamemode survival @p"]),
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

    env.reset()
    for ep in range(episodes):
        obs = env.reset(options={"fast_reset": True, "extra_commands": SCENE})[0]
        for _ in range(10):
            step()
        frames, dxs, dys, keys_l = [], [], [], []
        rec = save_demo is not None

        def rstep(a=None):
            a = a or dict(noop)
            if rec:
                frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])
                dxs.append(float(a.get("camera_yaw", 0.0)) * DEG2PX)
                dys.append(float(a.get("camera_pitch", 0.0)) * DEG2PX)
                keys_l.append([float(bool(a.get(k, False))) for k in V2_KEYS])
            return step(a)

        if policy_name == "teacher":
            miner = MineForwardPolicy(rng, epsilon=0.0)              # 步骤1:挖铁
            for t in range(MINE_STEPS):
                rstep(miner(t, noop, obs))
            rstep(dict(noop, inventory=True)); rstep(); rstep()      # 步骤2:开背包合木板
            craft_from_grid(GuiCursor(env, noop, rstep), SLOT_INV0)
            for _ in range(4):
                rstep()
        else:
            for t in range(MINE_STEPS + 40):
                step(learned(t, noop, obs))

        iron = inv_count(obs["full"], ["raw_iron", "iron_ore"])
        plk = inv_count(obs["full"], ["plank"])
        row = {"ep": ep, "iron": int(iron), "planks": int(plk),
               "mine_ok": iron > 0, "craft_ok": plk > 0, "both": iron > 0 and plk > 0}
        if rec and row["both"]:                                     # 只存两步都成的干净示范
            frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])
            os.makedirs(save_demo, exist_ok=True)
            np.savez_compressed(os.path.join(save_demo, f"ms_ep{ep:03d}.npz"),
                                frames=np.stack(frames).astype(np.uint8),
                                dx=np.array(dxs, np.float32), dy=np.array(dys, np.float32),
                                keys=np.array(keys_l, np.uint8), gui=np.ones(len(dxs), np.uint8),
                                score=np.float32(iron + plk), policy_strong=np.int64(1),
                                start_hard=np.int64(0))
        rows.append(row)
        print(f"[multi/{policy_name}] ep{ep} iron={iron} planks={plk} "
              f"mine={row['mine_ok']} craft={row['craft_ok']}", flush=True)
    env.close()
    agg = lambda k: round(float(np.mean([r[k] for r in rows])), 3)
    res = {"policy": policy_name, "episodes": episodes, "mine_rate": agg("mine_ok"),
           "craft_rate": agg("craft_ok"), "both_rate": agg("both"), "rows": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[multi/{policy_name}] mine={res['mine_rate']} craft={res['craft_rate']} "
          f"both={res['both_rate']} → {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="teacher")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--port", type=int, default=9330)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_demo", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or f"runs/ceiling/multiskill_{args.policy}.json"
    run(args.policy, args.episodes, args.port, args.seed, out, args.ckpt, args.max_len, args.save_demo)


if __name__ == "__main__":
    main()
