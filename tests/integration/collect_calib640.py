#!/usr/bin/env python3
"""YOLO 向量校准数据采集器:640×360 原始帧 + 位姿 + raycast + GT 方块坐标。

服务 G1 闸门(knowledge/design_fovea_yolo_fasttower.md §2 预登记判据:铁矿 mIoU≥0.5
且比零样本文本 prompt 基线 +0.3,必须用 640×360 原始渲染帧)。与 collect_s8 的差异:
  · 帧不裁剪:存原始 [T,3,360,640] u8(每 stride 步存一帧,控制体积);
  · 逐帧存位姿 (x,y,z,yaw,pitch) 与 raycast(target_block 坐标/键/距离)——投影 GT 用;
  · 策略只观察不攻击(方块永存,GT 全程有效):围绕矿墙扫视/平移/进退;
  · GT 方块坐标:课程把中心铁矿放在准星正前方,起手用 raycast 锚定其绝对坐标,
    其余方块按课程相对偏移推出(铁矿5 + 煤矿2 + 泥土2,后两类=校准硬负类)。

用法(冒烟→全量):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. .venv/bin/python \
      tests/integration/collect_calib640.py --episodes 1 --steps 24 --out runs/data/calib640_smoke
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. .venv/bin/python \
      tests/integration/collect_calib640.py --episodes 12 --out runs/data/calib640
"""
import argparse
import json
import os
import time

import numpy as np

# 课程相对偏移(x_off, y_off),z 均在墙面;与 build_calib_course 的 setblock 一一对应
ORE_OFFSETS = {
    "iron_ore": [(0, 0), (1, 0), (-1, 0), (0, 1), (-1, 1)],
    "coal_ore": [(2, 0), (2, 1)],
    "dirt":     [(-2, 0), (-2, 1)],
}


def sample_offsets(rng):
    """随机布局:铁 5(含锚定块 (0,0))/煤 2/泥土 2,墙面 x∈[-3,3]×y∈[0,2] 无重叠。

    固定十字布局会让学习向量有机会背"位置先验"而非纹理语义;随机布局的留出局
    是 G1 结论稳健性的对照。锚定块必须保留在 (0,0)(准星起手正对它)。"""
    cells = [(x, y) for x in range(-3, 4) for y in range(0, 3) if (x, y) != (0, 0)]
    rng.shuffle(cells)
    return {"iron_ore": [(0, 0)] + cells[:4],
            "coal_ore": cells[4:6],
            "dirt": cells[6:8]}


def build_calib_course(wall_z=7, offsets=None):
    """C2 同构房间:后墙石头,眼平中心铁矿 + 干扰块;不发工具(无破坏可能)。"""
    offsets = offsets or ORE_OFFSETS
    cmds = [
        "gamemode survival @p",
        "difficulty peaceful",
        "tp @p ~ ~ ~ 0 0",
        f"fill ~-4 ~-1 ~-2 ~4 ~4 ~{wall_z} minecraft:air",
        f"fill ~-4 ~-2 ~-2 ~4 ~-2 ~{wall_z} minecraft:stone",
        f"fill ~-4 ~-1 ~{wall_z} ~4 ~4 ~{wall_z} minecraft:stone",
        "clear @p",
    ]
    for blk, offs in offsets.items():
        for xo, yo in offs:
            cmds.append(f"setblock ~{xo} ~{yo} ~{wall_z} minecraft:{blk}")
    return cmds


WALL_Z_VARIANTS = (5, 7, 9, 11)


class ObservePolicy:
    """观察策略:不攻击。目标视角随机游走(yaw∈±35°,pitch∈[-25,12]),平滑逼近;
    随机穿插平移/进退(前进有限额,防脸贴墙)。用 full.yaw/pitch 反馈闭环。"""

    def __init__(self, rng, max_fwd=6):
        self.rng = rng
        self.tgt_yaw = 0.0
        self.tgt_pitch = 0.0
        self.hold = 0
        self.fwd_credit = max_fwd            # 净前进限额(前进-1,后退+1)

    def __call__(self, t, noop, obs=None):
        a = dict(noop)
        full = obs["full"] if obs is not None else None
        yaw = float(getattr(full, "yaw", 0.0)) if full is not None else 0.0
        pitch = float(getattr(full, "pitch", 0.0)) if full is not None else 0.0
        if self.hold <= 0:                   # 换一个目标视角
            self.tgt_yaw = float(self.rng.uniform(-35, 35))
            self.tgt_pitch = float(self.rng.uniform(-25, 12))
            self.hold = int(self.rng.integers(6, 16))
        self.hold -= 1
        a["camera_yaw"] = float(np.clip(0.4 * (self.tgt_yaw - yaw), -18, 18))
        a["camera_pitch"] = float(np.clip(0.4 * (self.tgt_pitch - pitch), -12, 12))
        r = self.rng.random()
        if r < 0.12 and self.fwd_credit > 0:
            a["forward"] = True
            self.fwd_credit -= 1
        elif r < 0.24 and self.fwd_credit < 6:
            a["back"] = True
            self.fwd_credit += 1
        elif r < 0.34:
            a["left"] = True
        elif r < 0.44:
            a["right"] = True
        return a


def _pose(full):
    return [float(getattr(full, k, 0.0)) for k in ("x", "y", "z", "yaw", "pitch")]


def _ray(full):
    """raycast → (block_xyz int[3], translation_key, dist);无命中 → 零/空。"""
    try:
        tb = full.raycast_result.target_block
        key = tb.translation_key or ""
        if not key:
            return [0, 0, 0], "", -1.0
        d = float(np.sqrt((tb.x + 0.5 - full.x) ** 2 + (tb.y + 0.5 - full.y) ** 2
                          + (tb.z + 0.5 - full.z) ** 2))
        return [int(tb.x), int(tb.y), int(tb.z)], key, d
    except Exception:  # noqa
        return [0, 0, 0], "", -1.0


def _frame(rgb):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))          # [3,H,W] u8


def anchor_gt_blocks(env, noop, offsets, max_tries=40, anchor_key="iron_ore"):
    """起手准星正对中心锚定块:step noop 直到 raycast 命中 anchor_key,锚定绝对坐标;
    再按 offsets 推全部 GT 方块。返回 {cls: [[x,y,z],...]} 或 None。"""
    for _ in range(max_tries):
        obs, *_ = env.step(noop)
        xyz, key, _d = _ray(obs["full"])
        if anchor_key in key:
            cx, cy, cz = xyz
            gt = {blk: [[cx + xo, cy + yo, cz] for xo, yo in offs]
                  for blk, offs in offsets.items()}
            return gt, obs
    return None, obs


def run(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    os.makedirs(args.out, exist_ok=True)
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
    print(f"[calib] env up (port {args.port}) → {args.out}", flush=True)

    rng = np.random.default_rng(args.seed)
    wz_list = args.wall_zs or list(WALL_Z_VARIANTS)
    n_done = 0
    for ep in range(args.episodes):
        wall_z = wz_list[ep % len(wz_list)]
        name = f"calib_v{wall_z}_s{args.seed}_e{ep}"
        outp = os.path.join(args.out, name + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        if args.hard_neg:                  # 纯石墙房:无矿块,全帧负样本(conv头假阳性专治)
            offsets = {k: [] for k in ORE_OFFSETS}
            obs, _ = env.reset(options={"fast_reset": True,
                                        "extra_commands": build_calib_course(wall_z, offsets)})
            for _ in range(args.settle):
                obs, *_ = env.step(noop)
            gt = {k: [] for k in ORE_OFFSETS}
        else:
            offsets = sample_offsets(rng) if args.rand_layout else ORE_OFFSETS
            obs, _ = env.reset(options={"fast_reset": True,
                                        "extra_commands": build_calib_course(wall_z, offsets)})
            for _ in range(args.settle):
                obs, *_ = env.step(noop)
            gt, obs = anchor_gt_blocks(env, noop, offsets)
        if gt is None:
            print(f"[calib] ✗ {name} 锚定失败(raycast 未命中铁矿),跳过", flush=True)
            continue
        pol = ObservePolicy(rng)
        frames, poses, ray_xyz, ray_key, ray_d = [], [], [], [], []
        for t in range(args.steps):
            a = pol(t, noop, obs)
            obs, *_ = env.step(a)
            if t % args.stride == 0:
                full = obs["full"]
                frames.append(_frame(obs["rgb"]))
                poses.append(_pose(full))
                xyz, key, d = _ray(full)
                ray_xyz.append(xyz)
                ray_key.append(key)
                ray_d.append(d)
        np.savez_compressed(
            outp, frames=np.stack(frames).astype(np.uint8),
            pose=np.array(poses, np.float32),
            ray_xyz=np.array(ray_xyz, np.int64),
            ray_key=np.array(ray_key), ray_dist=np.array(ray_d, np.float32),
            gt_blocks=json.dumps(gt),
            meta=json.dumps({"wall_z": wall_z, "steps": args.steps,
                             "stride": args.stride, "episode": ep,
                             "rand_layout": bool(args.rand_layout)}))
        n_done += 1
        print(f"[calib] ✓ {name} T={len(frames)} wall_z={wall_z} "
              f"{time.time()-t0:.0f}s [{n_done} done]", flush=True)
    env.close()
    print(f"[calib] DONE {n_done} → {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/data/calib640")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--stride", type=int, default=3, help="每 N 步存一帧")
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rand_layout", action="store_true",
                   help="每局随机矿块布局(反位置先验对照;默认固定十字)")
    p.add_argument("--hard_neg", action="store_true",
                   help="纯石墙房(无矿),全帧负样本——专治铁类假阳性")
    p.add_argument("--wall_zs", type=int, nargs="+", default=[],
                   help="覆盖墙距序列(默认 WALL_Z_VARIANTS)")
    p.add_argument("--port", type=int, default=8533)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
