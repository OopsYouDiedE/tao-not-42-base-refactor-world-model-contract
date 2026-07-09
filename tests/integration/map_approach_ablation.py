#!/usr/bin/env python3
"""地图接入快塔的效果消融:目标离屏后靠自我中心地图记忆重获取(无重训,首轮判据)。

问题(用户):快塔纯反应式(只吃当帧 token),看得见树但树一离屏就丢、乱转找不回,
approach 只~1/5。假设:接入 EgoMap 存"看过的木头在哪",离屏时朝记忆位置转回去,
approach 稳。本脚本无重训——用地图作外部记忆驱动重获取行为,先测"地图有没有用",
有用再谈让塔学着吃地图(需 GRPO 侧带位姿重训)。

两臂(同 seed 课程,唯一差异=离屏时是否用地图记忆):
  nomap  离屏→学生盲搜(现状)
  map    离屏→查地图最近木头方位→转回去+前进(记忆重获取)
地图写入=raycast 命中 log 的世界坐标(精确,免 IPM);北锚定 step 用位姿位移。
指标:wood_rate / 挖掘闩锁触发 / 首次拿到木头步数。
"""
import argparse
import json
import time

import numpy as np

from net.fovea_twotower.ego_map import EgoMapNorthLoc, MapQuery
from net.fovea_twotower.token_stream import as_hwc, goal_relative, wrap180, TokenHead
from net.fovea_twotower.wood import MINE_HOLD, WOOD_CLASSES
from tests.integration.collect_calib640 import _pose, _ray
from tests.integration.collect_calib_natural import relocate_cmds
from tests.integration.fullloop_chain import env_inventory
from tests.integration.wood_chain import place_trees


def episode(env, noop, arm, tok_head, student, rng, max_steps):
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": relocate_cmds(rng)})
    for _ in range(20):
        obs, *_ = env.step(noop)
    time.sleep(2.0)
    obs = place_trees(env, noop, rng)
    student.reset()
    gcls = WOOD_CLASSES.index("log")
    emap = EgoMapNorthLoc(1, 64, 32.0)          # 单类=log 的自我中心地图
    mq = MapQuery(emap, ["log"])
    prev_xz = None
    rgb = as_hwc(obs["rgb"])
    saw = latch = mine_hold = mapsteer = 0
    ok = False
    got_step = -1
    t = 0
    for t in range(max_steps):
        pose = _pose(obs["full"])                # (x,y,z,yaw,pitch)
        xz = np.array([pose[0], pose[2]])
        if prev_xz is not None:                  # 北锚定:按世界位移滚动地图
            emap.step(xz - prev_xz)
        prev_xz = xz
        _xyz, key, dist = _ray(obs["full"])
        if "log" in key and 0 < dist <= 15:      # raycast 命中木头→世界坐标写入地图
            import torch
            emap.write(torch.tensor([[_xyz[0] - pose[0], _xyz[2] - pose[2]]],
                                    dtype=torch.float32), torch.ones(1, 1))
        if "log" in key and 0 < dist <= 4.5:
            mine_hold = MINE_HOLD
        if mine_hold > 0:                         # 粘性挖掘(两臂同款)
            mine_hold -= 1
            latch += 1
            a = dict(noop)
            a["attack"] = True
            if t % 6 == 0:
                a["forward"] = True
        else:
            toks = tok_head(rgb)
            vis = len(toks) and float(toks[:, 6 + gcls].max()) > 0.4
            saw += vis
            if vis:
                rel = goal_relative(toks[None], np.array([gcls]))[0]
                a = student(rel, noop)
            elif arm == "map":                   # 离屏:查地图记忆→转回去
                v, _txt = mq.nearest("log")
                if v is not None:
                    mapsteer += 1
                    des_yaw = float(np.degrees(np.arctan2(-v[0], v[1])))
                    dyaw = wrap180(des_yaw - pose[3])
                    a = dict(noop)
                    a["camera_yaw"] = float(np.clip(0.6 * dyaw, -18, 18))
                    if abs(dyaw) < 30:
                        a["forward"] = True
                else:                            # 地图也没记忆→退回盲搜
                    rel = goal_relative(toks[None], np.array([gcls]))[0]
                    a = student(rel, noop)
            else:                                # nomap:盲搜(现状)
                rel = goal_relative(toks[None], np.array([gcls]))[0]
                a = student(rel, noop)
        obs, *_ = env.step(a)
        rgb = as_hwc(obs["rgb"])
        if t % 10 == 0 and any("log" in i for i in env_inventory(obs["full"])):
            ok = True
            got_step = t
            break
    return dict(arm=arm, ok=ok, steps=t + 1, saw=saw, latch=latch,
                mapsteer=mapsteer, got_step=got_step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--arms", nargs="+", default=["nomap", "map"])
    p.add_argument("--ckpt", default="runs/trackcmd_bc_v17/best.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head_v7b_wood.pt")
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--port", type=int, default=8871)
    p.add_argument("--out", default="runs/map_approach_ablation.json")
    args = p.parse_args()

    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from train.fovea_twotower.eval_track_cmd import StudentPolicy

    tok_head = TokenHead(args.vectors, conv_head=args.conv_head, classes=WOOD_CLASSES)
    student = StudentPolicy(args.ckpt)
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.DEFAULT, seed="mapablation", request_raycast=True,
        initial_extra_commands=["gamemode survival @p", "difficulty peaceful"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    import atexit
    _done = []

    def _sd():
        if _done:
            return
        _done.append(1)
        try:
            env.close()
        except Exception:
            pass
    atexit.register(_sd)
    noop = no_op_v2()
    env.reset()
    res = []
    for ep in range(args.episodes):
        for arm in args.arms:
            rng = np.random.default_rng(args.seed + ep)   # 同 ep 两臂同课程
            r = episode(env, noop, arm, tok_head, student, rng, args.max_steps)
            res.append(r)
            print(f"[abl] ep{ep}[{arm}] ok={r['ok']} steps={r['steps']} saw={r['saw']} "
                  f"latch={r['latch']} mapsteer={r['mapsteer']} got@{r['got_step']}",
                  flush=True)
    _sd()
    rate = {a: float(np.mean([r["ok"] for r in res if r["arm"] == a]))
            for a in args.arms}
    lat = {a: float(np.mean([r["latch"] for r in res if r["arm"] == a]))
           for a in args.arms}
    out = dict(wood_rate=rate, mean_latch=lat, episodes=res,
               note="无重训消融:地图作外部记忆驱动离屏重获取;有增益则值得让塔学吃地图")
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(dict(wood_rate=rate, mean_latch=lat), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
