#!/usr/bin/env python3
"""指令条件追踪/导航示范采集器(E1:快脑受慢脑指挥的最小闭环验证,数据端)。

设计(knowledge/design_fovea_yolo_fasttower.md §4 + §4.5 接线):
  · 课程 = calib 随机布局房间(铁/煤/泥土上墙,位置随机 → 反位置先验);
  · 指令 = 每局两段:前半指令类 g1,t=T/2 切换为 g2(≠g1)——示范天然含"重瞄准"段落,
    这是"指令切换可控性"判据的训练素材;
  · 教师 = 位姿投影瞄准:已知指令类方块世界坐标 + 逐帧位姿 → 期望 yaw/pitch,
    相机增量=比例控制;对准且未到则 forward。动作是几何量的函数,在 token 空间完全
    可表达(目标 token 居中/变大)→ 解掉"raycast 特权教师不可 BC"死结(承 mamba_seed §4.2);
  · token = 统一头在线计算:[K, 6几何 + 4类softmax概率](G1 校准向量 runs/g1_vectors.pt),
    训练侧再折成 goal 相对视图 → 快头结构上类无关,"听指挥"内建在输入契约里。

存盘(每局 npz):tokens[T,K,10] fp16 / goal_idx[T] / dx,dy,keys,gui,dt(s8 动作契约,
train_tracknav.clip_action 可直读) / pose[T,5] / ang_err[T](教师角误差,闭环天花板) /
dist[T] / gt_blocks / meta。

用法(GPU token + CPU 渲染并行):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. .venv/bin/python \
      tests/integration/collect_track_cmd.py --episodes 2 --steps 40 --out runs/data/trackcmd_smoke
"""
import argparse
import json
import os
import time

import numpy as np

from net.fovea_twotower.token_stream import (CLASSES, AimTeacher,  # noqa: F401
                                             TokenHead, TokenTeacher,
                                             aim_solution, as_hwc)
from tests.integration.collect_calib640 import (WALL_Z_VARIANTS, _pose,
                                                anchor_gt_blocks,
                                                build_calib_course,
                                                sample_offsets)
from tests.integration.collect_s8 import DEG2PX, V2_KEYS


def act_fields(a):
    keys = np.array([bool(a.get(k, False)) for k in V2_KEYS], np.uint8)
    return (float(a.get("camera_yaw", 0.0)) * DEG2PX,
            float(a.get("camera_pitch", 0.0)) * DEG2PX, keys)


def run(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    os.makedirs(args.out, exist_ok=True)
    tok_head = TokenHead(args.vectors, K=args.K, conv_head=args.conv_head)
    student = None
    if args.dagger_ckpt:                                # DAgger 环:学生策略驱动环境,
        from train.fovea_twotower.eval_track_cmd import StudentPolicy
        student = StudentPolicy(args.dagger_ckpt)       # 教师在学生访问的状态上打标签
        print(f"[trackcmd] DAgger 模式:{args.dagger_ckpt} (beta={args.beta})", flush=True)
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea",
        request_raycast=True,
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    print(f"[trackcmd] env up (port {args.port}) → {args.out}", flush=True)

    rng = np.random.default_rng(args.seed)
    n_done = 0
    for ep in range(args.episodes):
        wall_z = WALL_Z_VARIANTS[ep % len(WALL_Z_VARIANTS)]
        name = f"tc_v{wall_z}_s{args.seed}_e{ep}"
        outp = os.path.join(args.out, name + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        offsets = sample_offsets(rng)
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_calib_course(wall_z, offsets)})
        for _ in range(args.settle):
            obs, *_ = env.step(noop)
        gt, obs = anchor_gt_blocks(env, noop, offsets)
        if gt is None:
            print(f"[trackcmd] ✗ {name} 锚定失败,跳过", flush=True)
            continue
        # 开局扰乱:随机甩开相机(否则准星白送在中心矿上,示范学不到"从任意姿态搜目标")
        if args.scramble:
            a = dict(noop)
            a["camera_yaw"] = float(rng.uniform(-40, 40))
            a["camera_pitch"] = float(rng.uniform(-15, 10))
            obs, *_ = env.step(a)
        # 指令时间表:每 switch_every±10 步换一个(≠前),v1 单切换学不会重定向(闭环 0.25)
        sched, t_next, cur = [], 0, int(rng.integers(len(CLASSES)))
        while t_next < args.steps:
            sched.append((t_next, cur))
            t_next += int(args.switch_every + rng.integers(-10, 11))
            cur = int(rng.choice([c for c in range(len(CLASSES)) if c != cur]))
        goal_of = lambda t: [g for s, g in sched if s <= t][-1]
        teacher = (TokenTeacher(rng, args.epsilon) if args.teacher == "token"
                   else AimTeacher(rng, args.epsilon if not student else 0.0))
        if student:
            student.reset()
        rec = {k: [] for k in ("tokens", "goal_idx", "dx", "dy", "keys",
                               "pose", "ang_err", "dist")}
        rgb = as_hwc(obs["rgb"])
        prev_goal = -1
        cal_frames, cal_pose = [], []           # --store_frames:运动帧+位姿(投影GT喂conv头,
        for t in range(args.steps):             # 感知的DAgger:在闭环访问分布上训感知)
            goal = goal_of(t)
            pose = _pose(obs["full"])
            toks = tok_head(rgb)
            if isinstance(teacher, TokenTeacher):
                if goal != prev_goal:
                    teacher.new_segment()
                a = teacher(noop, toks, goal, pose[4])
                _, _, err, d = min((aim_solution(pose, (b[0] + .5, b[1] + .5, b[2]))
                                    for b in gt[CLASSES[goal]]), key=lambda c: c[3])
            else:
                a, err, d = teacher(noop, pose, gt, CLASSES[goal])
            prev_goal = goal
            if args.store_frames and t % 2 == 0:
                cal_frames.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
                cal_pose.append(pose)
            rec["tokens"].append(toks)
            rec["goal_idx"].append(goal)
            dx, dy, keys = act_fields(a)
            rec["dx"].append(dx)
            rec["dy"].append(dy)
            rec["keys"].append(keys)
            rec["pose"].append(pose)
            rec["ang_err"].append(err)
            rec["dist"].append(d)
            if student and rng.random() >= args.beta:            # DAgger:学生开车访态,
                from net.fovea_twotower.token_stream import goal_relative
                rel = goal_relative(toks[None], np.array([goal]))[0]
                exec_a = student(rel, noop)                      # 教师只出标签
            else:
                exec_a = a
            obs, *_ = env.step(exec_a)
            rgb = as_hwc(obs["rgb"])
        T = args.steps
        np.savez_compressed(
            outp, tokens=np.stack(rec["tokens"]).astype(np.float16),
            goal_idx=np.array(rec["goal_idx"], np.int64),
            dx=np.array(rec["dx"], np.float32), dy=np.array(rec["dy"], np.float32),
            keys=np.stack(rec["keys"]), gui=np.zeros(T, np.uint8),
            dt=np.full(T, 2.0, np.float32),
            pose=np.array(rec["pose"], np.float32),
            ang_err=np.array(rec["ang_err"], np.float32),
            dist=np.array(rec["dist"], np.float32),
            gt_blocks=json.dumps(gt),
            meta=json.dumps({"wall_z": wall_z, "schedule": sched,
                             "epsilon": args.epsilon, "scramble": bool(args.scramble)}))
        if args.store_frames and cal_frames:    # calib 契约(load_eps 可直读)
            fd = args.out + "_frames"
            os.makedirs(fd, exist_ok=True)
            np.savez_compressed(
                os.path.join(fd, name + ".npz"),
                frames=np.stack(cal_frames).astype(np.uint8),
                pose=np.array(cal_pose, np.float32),
                ray_xyz=np.zeros((len(cal_frames), 3), np.int64),
                ray_key=np.array([""] * len(cal_frames)),
                ray_dist=np.full(len(cal_frames), -1.0, np.float32),
                gt_blocks=json.dumps(gt),
                meta=json.dumps({"wall_z": wall_z, "motion_frames": True}))
        n_done += 1
        ae = np.array(rec["ang_err"])
        print(f"[trackcmd] ✓ {name} {len(sched)}段 "
              f"err中位={np.median(ae[10:]):.1f}° "
              f"末距={rec['dist'][-1]:.1f} {time.time()-t0:.0f}s [{n_done}]", flush=True)
    env.close()
    print(f"[trackcmd] DONE {n_done} → {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/data/trackcmd")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.08)
    p.add_argument("--switch_every", type=int, default=40)
    p.add_argument("--scramble", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--teacher", choices=["oracle", "token"], default="token",
                   help="token=观测一致教师(v6 默认);oracle=位姿投影(有特权,只作对照)")
    p.add_argument("--dagger_ckpt", default="", help="学生 ckpt;设置即 DAgger 采集")
    p.add_argument("--store_frames", action="store_true",
                   help="另存运动帧+位姿(calib契约,投影GT喂conv头)")
    p.add_argument("--beta", type=float, default=0.15, help="DAgger 混入教师执行的概率")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8536)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
