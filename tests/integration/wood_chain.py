#!/usr/bin/env python3
"""W-CHAIN 段(2) 采木技能闭环终审:22M 学生零重训插拔契约实测。

协议:自然 DEFAULT 世界 spreadplayers 落点,/place feature minecraft:oak 在
8-15 格随机方位生成 2-3 棵**真 MC 橡树**(真树几何/叶冠遮挡;比赌森林落点
可控,比 setblock 柱真实;登记为课程辅助口径)。
臂:student=v17 学生(未见过 log 目标,goal_relative 类数无关→goal=log 即插)
   + WoodTokenHead(v6 4类头) + log raycast≤4.5 挖掘闩锁宏(attack 持续);
   random=同 m_iron random 臂。
成功=timeout 内背包出现 *log。门(预登记):student ≥0.5(n≥8);
GRPO 启动门对照口径:≥0.25 或组内里程碑分层。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 CRAFTGROUND_JVM_MAX_MEMORY=2G PYTHONPATH=. \
    .venv/bin/python tests/integration/wood_chain.py --episodes 8 --port 8860
"""
import argparse
import json
import time

import numpy as np

from net.fovea_twotower.token_stream import TokenHead, as_hwc, goal_relative
from net.fovea_twotower.wood import MINE_HOLD, WOOD_CLASSES
from tests.integration.collect_calib640 import _pose, _ray
from tests.integration.collect_calib_natural import relocate_cmds
from tests.integration.fullloop_chain import env_inventory


def place_trees(env, noop, rng, n=3):
    cmds = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(8, 15)
        x, z = int(d * np.sin(ang)), int(d * np.cos(ang))
        cmds.append(f"place feature minecraft:oak ~{x} ~ ~{z}")
        cmds.append(f"place feature minecraft:oak ~{x} ~1 ~{z}")
    env.add_commands(cmds)
    for _ in range(8):
        obs, *_ = env.step(noop)
    return obs


def episode(env, noop, arm, tok_head, student, rng, max_steps):
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": relocate_cmds(rng)})
    for _ in range(20):
        obs, *_ = env.step(noop)
    time.sleep(2.0)
    obs = place_trees(env, noop, rng)
    if arm == "student":
        student.reset()
    gcls = WOOD_CLASSES.index("log")
    rgb = as_hwc(obs["rgb"])
    saw = latch = 0
    mine_hold = 0
    ok = False
    t = 0
    for t in range(max_steps):
        if arm == "random":
            a = dict(noop)
            a["camera_yaw"] = float(rng.normal(0, 10))
            a["camera_pitch"] = float(rng.normal(0, 6))
            a["forward"] = bool(rng.random() < 0.3)
            a["attack"] = bool(rng.random() < 0.5)
        else:
            _xyz, key, dist = _ray(obs["full"])
            if "log" in key and 0 < dist <= 4.5:      # 命中木头→充能(持续命中=持续充能)
                mine_hold = MINE_HOLD
            if mine_hold > 0:                          # 粘性挖掘:相机锁死、一直砍到破
                mine_hold -= 1
                latch += 1
                a = dict(noop)                        # noop 相机=0→准星锁死树干,破坏不重置
                a["attack"] = True
                if t % 6 == 0:
                    a["forward"] = True               # 轻推贴住树干
            else:
                toks = tok_head(rgb)
                if len(toks) and float(toks[:, 6 + gcls].max()) > 0.4:
                    saw += 1
                rel = goal_relative(toks[None], np.array([gcls]))[0]
                a = student(rel, noop)
        obs, *_ = env.step(a)
        rgb = as_hwc(obs["rgb"])
        if t % 10 == 0 and any("log" in i for i in env_inventory(obs["full"])):
            ok = True
            break
    inv = sorted(env_inventory(obs["full"]))
    return dict(arm=arm, ok=bool(ok or any("log" in i for i in inv)),
                steps=t + 1, saw_log_tok=saw, latch=latch, inv_end=inv[:8])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--arms", nargs="+", default=["student", "random"])
    p.add_argument("--ckpt", default="runs/trackcmd_bc_v17/best.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head_v7b_wood.pt")
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--port", type=int, default=8860)
    p.add_argument("--out", default="runs/wood_chain_cl.json")
    args = p.parse_args()

    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from train.fovea_twotower.eval_track_cmd import StudentPolicy

    tok_head = TokenHead(args.vectors, conv_head=args.conv_head,
                         classes=WOOD_CLASSES)
    student = StudentPolicy(args.ckpt)
    rng = np.random.default_rng(args.seed)
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.DEFAULT, seed="woodchain1", request_raycast=True,
        initial_extra_commands=["gamemode survival @p", "difficulty peaceful"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    import atexit
    _done = []                     # JVM 收尾兜底:异常/中途退出也 close(),防 gradle 子树泄漏

    def _shutdown():
        if _done:
            return
        _done.append(1)
        try:
            env.close()
        except Exception:
            pass
    atexit.register(_shutdown)
    noop = no_op_v2()
    env.reset()
    res = []
    for ep in range(args.episodes):
        for arm in args.arms:
            r = episode(env, noop, arm, tok_head, student, rng, args.max_steps)
            res.append(r)
            print(f"[wc] ep{ep}[{arm}] ok={r['ok']} steps={r['steps']} "
                  f"saw={r['saw_log_tok']} latch={r['latch']} inv={r['inv_end'][:4]}",
                  flush=True)
    _shutdown()
    out = dict(
        rate={a: float(np.mean([r["ok"] for r in res if r["arm"] == a]))
              for a in args.arms},
        episodes=res,
        gates={"W2-closedloop(student>=0.5)":
               bool(np.mean([r["ok"] for r in res if r["arm"] == "student"]) >= 0.5),
               "GRPO-door(>=0.25)":
               bool(np.mean([r["ok"] for r in res if r["arm"] == "student"]) >= 0.25)},
        note="student=v17零重训插拔+v6头+log挖掘宏;trees=/place feature 真橡树")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["rate"], ensure_ascii=False), out["gates"], flush=True)


if __name__ == "__main__":
    main()
