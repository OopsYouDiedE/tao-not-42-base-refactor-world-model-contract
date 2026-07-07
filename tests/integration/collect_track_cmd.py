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
import torch

from tests.integration.collect_calib640 import (WALL_Z_VARIANTS, _pose, _ray,
                                                anchor_gt_blocks,
                                                build_calib_course,
                                                sample_offsets)
from tests.integration.collect_s8 import DEG2PX, V2_KEYS

CLASSES = ["iron_ore", "coal_ore", "dirt"]
MAX_CAM = 18.0                 # 单步相机增量上限(deg)
REACH_STOP = 2.8               # 到达即停的距离
EYE_H = 1.62


def aim_solution(pose, tgt):
    """位姿 + 目标点(前脸中心) → (期望yaw, 期望pitch, 角误差deg, 距离)。

    MC 约定:yaw=0 朝 +z,forward=(-sin y·cos p, -sin p, cos y·cos p)。"""
    x, y, z, yaw, pitch = pose
    eye = np.array([x, y + EYE_H, z])
    v = np.asarray(tgt, float) - eye
    d = float(np.linalg.norm(v))
    vy = v / (d + 1e-9)
    des_yaw = float(np.degrees(np.arctan2(-vy[0], vy[2])))
    des_pitch = float(np.degrees(-np.arcsin(np.clip(vy[1], -1, 1))))
    yr, pr = np.radians(yaw), np.radians(pitch)
    fwd = np.array([-np.sin(yr) * np.cos(pr), -np.sin(pr), np.cos(yr) * np.cos(pr)])
    err = float(np.degrees(np.arccos(np.clip(fwd @ vy, -1, 1))))
    return des_yaw, des_pitch, err, d


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class AimTeacher:
    """指令类最近方块前脸中心 → 比例控制相机 + 对准前进;epsilon 掺随机(覆盖度)。"""

    def __init__(self, rng, epsilon=0.08):
        self.rng, self.eps = rng, epsilon

    def __call__(self, noop, pose, gt_blocks, goal_cls):
        a = dict(noop)
        blocks = gt_blocks[goal_cls]
        cands = [aim_solution(pose, (b[0] + .5, b[1] + .5, b[2])) for b in blocks]
        des_yaw, des_pitch, err, d = min(cands, key=lambda c: c[3])
        if self.rng.random() < self.eps:
            a["camera_yaw"] = float(self.rng.normal(0, 10))
            a["camera_pitch"] = float(self.rng.normal(0, 6))
            a["forward"] = bool(self.rng.random() < 0.3)
            return a, err, d
        a["camera_yaw"] = float(np.clip(0.6 * wrap180(des_yaw - pose[3]),
                                        -MAX_CAM, MAX_CAM))
        a["camera_pitch"] = float(np.clip(0.6 * (des_pitch - pose[4]),
                                          -MAX_CAM, MAX_CAM))
        if err < 10.0 and d > REACH_STOP:
            a["forward"] = True
        return a, err, d


class TokenTeacher:
    """观测一致教师(v6):只消费学生同款 token,不用位姿 oracle → BC 可学性由构造保证。

    教训(v2–v5 连败根因):位姿投影教师在目标出视野时仍能直转目标——学生 token 里
    此时无任何方向信息,重获取不可学(mamba_seed 特权教师教训的隐蔽复发)。
    行为:goal token 可见(p_goal>τ)→ 瞄准其 (cx,cy)(水平 FOV≈100°/竖直 70° 映射),
    居中且 area<近距阈 → forward;不可见 → 匀速 yaw 搜索(方向随段随机)+ pitch 缓回。"""
    HFOV, VFOV = 100.0, 70.0
    TAU, AREA_NEAR, CENT = 0.22, 0.038, 0.05   # 0.030 停太远够不着 2.8 格线;0.045 贴脸切换后找不到新目标(v10 教师 p2=86°)
    GAIN, HOLD = 0.5, 4          # 低增益防过冲丢目标;短记忆抗检测闪断

    def __init__(self, rng, epsilon=0.05):
        self.rng, self.eps = rng, epsilon
        self.search_dir = 1.0
        self.last_off, self.hold_left = None, 0

    def new_segment(self):
        # 搜索方向恒右转:随机方向=不可观测潜变量,同观测下标签 ±15° 对冲,
        # BC 条件均值=0 → 学生永远学不会发起搜索(v12 切换率钉死 0.17 的根因)
        self.search_dir = 1.0
        self.last_off, self.hold_left = None, 0

    def __call__(self, noop, toks, goal_idx, pitch_now):
        a = dict(noop)
        if self.rng.random() < self.eps:
            a["camera_yaw"] = float(self.rng.normal(0, 10))
            a["camera_pitch"] = float(self.rng.normal(0, 6))
            return a
        pg = toks[:, 6 + goal_idx] * (toks[:, 4] > 0).astype(np.float32)
        score = pg * toks[:, 5]                         # 面积×概率:大连通域更可信
        j = int(np.argmax(score))                       # (取最高p会咬小噪点)
        off = None
        if pg[j] > self.TAU:
            off = (toks[j, 0] - 0.5, toks[j, 1] - 0.5, toks[j, 5])
            self.last_off, self.hold_left = off, self.HOLD
        elif self.hold_left > 0:                        # 闪断:按记忆位置继续压
            self.hold_left -= 1
            off = self.last_off
        if off is not None:
            offx, offy, area = off
            if abs(offx) > 0.02:                        # 死区防抖
                a["camera_yaw"] = float(np.clip(offx * self.HFOV * self.GAIN,
                                                -MAX_CAM, MAX_CAM))
            if abs(offy) > 0.02:
                a["camera_pitch"] = float(np.clip(offy * self.VFOV * self.GAIN,
                                                  -MAX_CAM, MAX_CAM))
            if abs(offx) < self.CENT and abs(offy) < self.CENT \
                    and area < self.AREA_NEAR:
                a["forward"] = True
        else:                                           # 不可见:搜索
            a["camera_yaw"] = float(15.0 * self.search_dir)
            a["camera_pitch"] = float(np.clip(-0.3 * pitch_now, -8, 8))
            if toks[:, 5].max() > 0.20:                 # 贴脸(超大连通域):后退拉开视野
                a["back"] = True                        # (v10 教训:贴墙切换重获取不可能)
        return a


class TokenHead:
    """G1 验收的 conv 分割头 → 连通域 → 每帧 [K, 6+C+1] token(几何 + 概率)。

    v6 教训:pf 提案池化命名假阳性泛滥(灰墙冒充铁,教师锁假目标 err 89°)——
    G1 已量化:提案并集 0.15 vs conv 稠密 0.53。token 必须从**验收过的分割通道**
    的连通域构建;pf 提案留给开放集,不再承担核心类命名。"""

    def __init__(self, vectors="runs/g1_vectors.pt", K=8, device="cuda",
                 conv_head="runs/g1_conv_head.pt", min_area=150):
        import cv2 as _cv2
        from net.fovea_twotower.yolo_unified import UnifiedYoloe26, pad384
        from train.fovea_twotower.eval_g1 import ConvSegHead
        self.cv2 = _cv2
        self.u = UnifiedYoloe26(device=device, pf_w=None)
        self.pad = pad384
        self.head = ConvSegHead().to(device).eval()
        self.head.load_state_dict(torch.load(conv_head, map_location=device,
                                             weights_only=False))
        self.K, self.D, self.min_area = K, 6 + len(CLASSES) + 1, min_area

    @torch.no_grad()
    def __call__(self, rgb_hwc):
        img = self.pad(np.ascontiguousarray(rgb_hwc))
        prob = self.head(self.u.embed(img)[0].float())[0].softmax(0)  # [C+1,384,640]
        lab = prob.argmax(0).cpu().numpy().astype(np.uint8)
        prob_np = prob.cpu().numpy()
        cands = []
        for ci in range(len(CLASSES)):
            n, cc, stats, cent = self.cv2.connectedComponentsWithStats(
                (lab == ci).astype(np.uint8), 8)
            for j in range(1, n):
                x, y, w, h, area = stats[j]
                if area < self.min_area:
                    continue
                m = cc == j
                p = prob_np[:, m].mean(1)                       # [C+1] 域内均值
                cands.append((float(area) * float(p[ci]),
                              [cent[j][0] / 640, cent[j][1] / 384,
                               w / 640, h / 384, float(p[ci]),
                               area / (640 * 384)], p))
        cands.sort(key=lambda c: -c[0])
        toks = np.zeros((self.K, self.D), np.float32)
        for j, (_, geo, p) in enumerate(cands[:self.K]):
            toks[j, :6] = geo
            toks[j, 6:] = p
        return toks


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
        rgb = np.asarray(obs["rgb"])
        if rgb.shape[0] in (1, 3):
            rgb = rgb.transpose(1, 2, 0)
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
                from train.fovea_twotower.train_track_cmd import goal_relative
                rel = goal_relative(toks[None], np.array([goal]))[0]
                exec_a = student(rel, noop)                      # 教师只出标签
            else:
                exec_a = a
            obs, *_ = env.step(exec_a)
            rgb = np.asarray(obs["rgb"])
            if rgb.shape[0] in (1, 3):
                rgb = rgb.transpose(1, 2, 0)
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
