#!/usr/bin/env python3
"""M-IRON Step0:当前系统在自然生成世界拿铁的裸基线(判据预登记 66d518c)。

判据(结果制):WorldType.DEFAULT 随机种子,出生点冷启动,时限内背包出现铁
(iron_ore/raw_iron/iron_ingot 任一,不问来源)=成功;无教师/无特权/无 setblock。
系统臂=当前最强可跑形态(C1b 全回路的自然世界移植):
  慢脑(Qwen1.5B+E2 LoRA)差额计划 → goal → 快头(22M 学生)追踪/搜索
  + C1b 挖掘宏(raycast 闩锁;脚本占位技能,继承的诚实边界,待学习技能替代)。
如实标注的简化:①慢脑只在局首规划一次(无中途复核/重规划);②计划首项若
无感知类映射,回退 goal=iron_ore 并记 fallback;③difficulty peaceful(与
C1b 一致,世界设置非 agent 特权)。对照臂=random。
预期:成功率≈0(裸手挖铁无掉落/铁在地下/无石镐链)——失败模式分类才是产出。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 CRAFTGROUND_JVM_MAX_MEMORY=2G PYTHONPATH=. \
      .venv/bin/python tests/integration/m_iron.py --worlds 4 --max_steps 2000
"""
import argparse
import json
import time

import numpy as np

from net.fovea_twotower.token_stream import CLASSES, TokenHead, as_hwc, goal_relative
from tests.integration.collect_calib640 import _pose, _ray
from tests.integration.fullloop_chain import ITEM2CLS, SlowBrain, env_inventory

IRON_ITEMS = {"raw_iron", "iron_ore", "iron_ingot", "deepslate_iron_ore"}
RESET_CMDS = ["gamemode survival @p", "difficulty peaceful", "clear @p"]


def episode(env, noop, arm, tok_head, student, brain, rng, max_steps):
    obs, _ = env.reset(options={"fast_reset": True, "extra_commands": RESET_CMDS})
    for _ in range(10):
        obs, *_ = env.step(noop)
    log = dict(arm=arm)
    goal_cls = "iron_ore"
    if arm == "system":
        inv0 = env_inventory(obs["full"])
        steps, _, _ = brain.plan("raw_iron", inv0 & {"stone_pickaxe", "raw_iron"})
        first = steps[0] if steps else ""
        mapped = ITEM2CLS.get(first, "")
        log.update(plan=steps[:4], plan_first=first,
                   fallback=bool(mapped != "iron_ore"))
        goal_cls = mapped if mapped in CLASSES else "iron_ore"
        student.reset()
    gcls = CLASSES.index(goal_cls)
    rgb = as_hwc(obs["rgb"])
    saw_iron_tok = 0          # 感知层:token 流里出现过铁类
    macro_latched = 0         # raycast 层:挖掘宏接管过
    ok, died, t = False, False, 0
    for t in range(max_steps):
        if arm == "random":
            a = dict(noop)
            a["camera_yaw"] = float(rng.normal(0, 10))
            a["camera_pitch"] = float(rng.normal(0, 6))
            a["forward"] = bool(rng.random() < 0.3)
            a["attack"] = bool(rng.random() < 0.5)
            a["jump"] = bool(rng.random() < 0.1)
        else:
            _xyz, key, dist = _ray(obs["full"])
            if "iron_ore" in key and 0 < dist <= 5.5:   # C1b 挖掘宏原样继承
                macro_latched += 1
                a = dict(noop)
                if dist > 3.2:
                    a["forward"] = True
                else:
                    a["attack"] = True
                    if t % 10 == 0:
                        a["forward"] = True
            else:
                toks = tok_head(rgb)
                if len(toks) and float(toks[:, 6 + gcls].max()) > 0.4:
                    saw_iron_tok += 1
                rel = goal_relative(toks[None], np.array([gcls]))[0]
                a = student(rel, noop)
        obs, *_ = env.step(a)
        rgb = as_hwc(obs["rgb"])
        full = obs["full"]
        if getattr(full, "is_dead", False):
            died = True
            break
        if t % 20 == 0 and env_inventory(full) & IRON_ITEMS:
            ok = True
            break
    inv_end = sorted(env_inventory(obs["full"]))
    log.update(ok=bool(ok or bool(set(inv_end) & IRON_ITEMS)), died=died,
               steps=t + 1, saw_iron_tok=saw_iron_tok,
               macro_latched=macro_latched, inv_end=inv_end[:12])
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--arms", nargs="+", default=["system", "random"])
    p.add_argument("--ckpt", default="runs/trackcmd_bc_v17/best.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head_v4.pt")
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--adapter", default="runs/reason_delta_lora_v4")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--port", type=int, default=8760)
    p.add_argument("--out", default="runs/m_iron_step0.json")
    args = p.parse_args()

    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from train.fovea_twotower.eval_track_cmd import StudentPolicy

    tok_head = TokenHead(args.vectors, conv_head=args.conv_head)
    student = StudentPolicy(args.ckpt)
    brain = SlowBrain(args.adapter) if "system" in args.arms else None
    rng = np.random.default_rng(args.seed)
    results = []
    for w in range(args.worlds):
        wseed = str(int(rng.integers(1, 2**31)))
        cfg = InitialEnvironmentConfig(
            image_width=640, image_height=360,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            world_type=WorldType.DEFAULT, seed=wseed, request_raycast=True,
            initial_extra_commands=list(RESET_CMDS))
        t0 = time.time()
        env = make(initial_env_config=cfg,
                   action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                   port=args.port + w, verbose=False)
        noop = no_op_v2()
        env.reset()
        print(f"[world {w}] seed={wseed} up in {time.time()-t0:.0f}s", flush=True)
        for arm in args.arms:
            r = episode(env, noop, arm, tok_head, student, brain, rng,
                        args.max_steps)
            r["world_seed"] = wseed
            results.append(r)
            print(f"[world {w}][{arm}] ok={r['ok']} died={r['died']} "
                  f"steps={r['steps']} saw_tok={r.get('saw_iron_tok')} "
                  f"macro={r.get('macro_latched')} inv={r['inv_end'][:6]}", flush=True)
        env.close()
        time.sleep(3)
    out = dict(
        success_rate={a: float(np.mean([r["ok"] for r in results if r["arm"] == a]))
                      for a in args.arms},
        n_per_arm={a: sum(r["arm"] == a for r in results) for a in args.arms},
        episodes=results, max_steps=args.max_steps,
        note="Step0 裸基线;挖掘宏=C1b 继承脚本占位;慢脑局首单次规划")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["success_rate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
