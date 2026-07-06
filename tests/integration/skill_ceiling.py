#!/usr/bin/env python3
"""技能上限终审:Minecraft 技能课程 × 闭环成功率(键鼠闭环,非代理指标)。

设计原则(承过去教训):**终审只认闭环技能成功率**,不看潜空间/探针/loss。每个技能有
确定性成功检测(obs['full'] 库存/统计/存活),先量教师(脚本示范)与基线(noop/random)的
成功率 → 环境天花板;再拿学到的快塔跑同一 harness → 快塔的学习上限。

技能课程(从易到难,均 setblock 摆确定目标 + 给对应工具,成功=闭环产物进背包):
  survive       空平坦生存,成功=N 步后未死
  chop_wood     正前方 oak_log 墙,徒手,成功=背包出现 log
  mine_stone    stone 墙 + 木镐,成功=cobblestone
  mine_iron     iron_ore 墙 + 石镐,成功=raw_iron(= 旧 C2)
  mine_diamond  diamond_ore 墙 + 铁镐,成功=diamond(感知更难:矿纹稀有 OOD)

策略:teacher(collect_s8 的 raycast 闭环采矿器,块无关=瞄准+挖)/ noop / random /
      learned(载 TrackNav/BC ckpt,后续接)。

用法(RAW 渲染):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. ./.venv/bin/python \
      tests/integration/skill_ceiling.py --skill mine_iron --policy teacher \
      --episodes 20 --steps 220 --out runs/ceiling/mine_iron_teacher.json
"""
import argparse
import json
import os
import time

import numpy as np

from tests.integration.collect_s8 import MineForwardPolicy, RandomPolicy, _raycast

# 技能 = (目标方块 | None=生存, 工具 | None, 成功库存关键词子串)
SKILLS = {
    "survive":      dict(block=None,          tool=None,             success=None),
    "chop_wood":    dict(block="oak_log",     tool=None,             success=["log", "plank"]),
    "mine_stone":   dict(block="stone",       tool="wooden_pickaxe", success=["cobblestone"]),
    "mine_iron":    dict(block="iron_ore",    tool="stone_pickaxe",  success=["iron"]),
    "mine_diamond": dict(block="diamond_ore", tool="iron_pickaxe",   success=["diamond"]),
}
WALL_Z = (5, 6, 7, 8, 9, 10)


def build_course(block, tool, wall_z):
    """通用课程:清空 + 正前方 z+wall_z 摆一面 block 墙(眼平中心一块正对准星)+ 给 tool。"""
    cmds = ["gamemode survival @p", "difficulty peaceful", "tp @p ~ ~ ~ 0 0",
            f"fill ~-3 ~-1 ~-2 ~3 ~4 ~{wall_z} minecraft:air",
            f"fill ~-3 ~-2 ~-2 ~3 ~-2 ~{wall_z} minecraft:stone", "clear @p"]
    if block:
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (-1, 1)]:
            cmds.append(f"setblock ~{dx} ~{dy} ~{wall_z} minecraft:{block}")
        cmds.insert(4, f"fill ~-3 ~-1 ~{wall_z} ~3 ~4 ~{wall_z} minecraft:stone")  # 后墙托底
    if tool:
        cmds.append(f"item replace entity @p weapon.mainhand with minecraft:{tool} 1")
    return cmds


def inv_count(full, keys):
    """背包里 translation_key 含任一子串的物品计数之和。"""
    n = 0
    try:
        for it in full.inventory:
            if getattr(it, "count", 0) > 0 and any(k in (it.translation_key or "") for k in keys):
                n += it.count
    except Exception:  # noqa
        pass
    return n


def make_policy(name, rng, ckpt=None):
    if name == "teacher":
        return MineForwardPolicy(rng, epsilon=0.0)          # 纯闭环(无探索噪声)
    if name == "random":
        return RandomPolicy(rng)
    if name == "noop":
        return lambda t, noop, obs=None: dict(noop)
    raise ValueError(f"未知 policy {name}(learned ckpt 载入后续接)")


def run(skill, policy_name, episodes, steps, settle, port, seed, out):
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    sk = SKILLS[skill]
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="ceiling", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=port, verbose=False)
    noop = no_op_v2()
    env.reset()
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        wall_z = WALL_Z[ep % len(WALL_Z)]
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_course(sk["block"], sk["tool"], wall_z)})
        for _ in range(settle):
            obs, *_ = env.step(noop)
        policy = make_policy(policy_name, rng)
        base = inv_count(obs["full"], sk["success"]) if sk["success"] else 0
        died = False
        for t in range(steps):
            a = policy(t, noop, obs)
            obs, *_ = env.step(a)
            if bool(getattr(obs["full"], "is_dead", False)):
                died = True
                break
        if sk["success"] is None:
            ok = not died                                    # 生存技能
            got = 0
        else:
            got = inv_count(obs["full"], sk["success"]) - base
            ok = got > 0
        rows.append({"ep": ep, "wall_z": wall_z, "success": bool(ok), "got": int(got), "died": died})
        print(f"[{skill}/{policy_name}] ep{ep} wall_z={wall_z} success={ok} got={got}", flush=True)
    env.close()
    rate = float(np.mean([r["success"] for r in rows]))
    res = {"skill": skill, "policy": policy_name, "episodes": episodes, "steps": steps,
           "success_rate": round(rate, 4),
           "mean_got": round(float(np.mean([r["got"] for r in rows])), 3),
           "per_episode": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[{skill}/{policy_name}] SUCCESS_RATE={rate:.3f} → {out}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill", choices=list(SKILLS), required=True)
    p.add_argument("--policy", default="teacher", help="teacher/noop/random")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or f"runs/ceiling/{args.skill}_{args.policy}.json"
    run(args.skill, args.policy, args.episodes, args.steps, args.settle,
        args.port, args.seed, out)


if __name__ == "__main__":
    main()
