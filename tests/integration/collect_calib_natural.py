#!/usr/bin/env python3
"""引擎 B 自然世界扩产:修 M-IRON Step0 的感知迁移崩(conv头对自然纹理满屏幻觉铁)。

复用 collect_calib640 的全部机制(课程墙/锚定/观察策略/npz schema),唯一差异=
世界与背景:WorldType.DEFAULT,每局 spreadplayers 随机落点(多生物群系),
  正样本局:落点处清一个观察口袋+石墙+矿块(与 calib640 同构,GT 锚定同款)
            ——矿在石墙上但四周/地面/天际=自然世界(域内正样本);
  负样本局:不放任何课程方块,纯自然帧全背景(专治自然纹理幻觉铁)。
已知妥协(如实登记):①自然界偶发天然裸露煤/铁进入负帧=标签噪声(地表罕见,
接受);②dirt 类在自然界大量存在而 GT 只标课程块=该类标签噪声(dirt 本为
吸收混淆的负类,接受);③统一 time set noon(光照多样性留待需要时扩)。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 CRAFTGROUND_JVM_MAX_MEMORY=2G PYTHONPATH=. \
    .venv/bin/python tests/integration/collect_calib_natural.py \
      --episodes 30 --out runs/data/calib_nat --port 8770
  加 --pure_neg 采纯自然负样本局。
"""
import argparse
import json
import os
import time

import numpy as np

from tests.integration.collect_calib640 import (ObservePolicy, _frame, _pose,
                                                anchor_gt_blocks,
                                                sample_offsets)

ORE_CLASSES = ("iron_ore", "coal_ore", "dirt")


def relocate_cmds(rng, superflat=False):
    """阶段1(fast_reset):随机落点。fast_reset=杀+重生后一个tick内连发命令,
    spreadplayers 后区块未加载,任何建造必须延后到阶段2(add_commands)。"""
    px, pz = int(rng.integers(-4000, 4000)), int(rng.integers(-4000, 4000))
    if superflat:
        return ["gamemode survival @p", "difficulty peaceful", "time set noon",
                f"tp @p {px} -60 {pz}", "clear @p"]
    return [
        "gamemode survival @p",
        "difficulty peaceful",
        "time set noon",
        f"spreadplayers {px} {pz} 0 128 false @p",
        "clear @p",
    ]


def build_cmds(wall_z, offsets):
    """阶段2(add_commands,区块已加载):朝向归零+观察口袋+石墙+矿块。"""
    cmds = [
        "tp @p ~ ~ ~ 0 0",
        f"fill ~-4 ~-1 ~-2 ~4 ~4 ~{wall_z} minecraft:air",
        f"fill ~-4 ~-1 ~{wall_z} ~4 ~4 ~{wall_z} minecraft:stone",
    ]
    for blk, offs in offsets.items():
        for xo, yo in offs:
            cmds.append(f"setblock ~{xo} ~{yo} ~{wall_z} minecraft:{blk}")
    return cmds


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
        world_type=(WorldType.SUPERFLAT if args.superflat else WorldType.DEFAULT),
        seed=args.world_seed,
        request_raycast=True,
        initial_extra_commands=["gamemode survival @p", "difficulty peaceful"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    print(f"[nat] env up (port {args.port}) → {args.out}", flush=True)

    rng = np.random.default_rng(args.seed)
    wz_list = (5, 7, 9)
    n_done = 0
    for ep in range(args.episodes):
        wall_z = wz_list[ep % len(wz_list)]
        tag = "neg" if args.pure_neg else f"pos_v{wall_z}"
        name = f"nat_{tag}_s{args.seed}_e{ep}"
        outp = os.path.join(args.out, name + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        offsets = None if args.pure_neg else sample_offsets(rng)
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": relocate_cmds(rng, args.superflat)})
        for _ in range(args.settle):
            obs, *_ = env.step(noop)
        time.sleep(2.5)                         # 区块生成是墙钟秒级,unlimited TPS
        for _ in range(5):                      # 下 tick 不足以等它(smoke 1/3 教训)
            obs, *_ = env.step(noop)
        if args.log_wall:                       # 木簇墙:铁位换 oak_log,GT 键=log
            offsets = {("log" if k == "iron_ore" else k): v
                       for k, v in offsets.items()}
        if args.pure_neg:
            env.add_commands(["tp @p ~ ~ ~ 0 0"])
            for _ in range(5):
                obs, *_ = env.step(noop)
            gt = {k: [] for k in ORE_CLASSES}
        else:
            tgt_p = float(np.degrees(np.arctan2(1.12, wall_z)))
            for _ in range(14):                 # 相机动作闭环归零朝向
                f0 = obs["full"]                # (tp 置向不生效——woodgt 发现)
                dy = (0 - float(getattr(f0, "yaw", 0.0)) + 180) % 360 - 180
                dp = tgt_p - float(getattr(f0, "pitch", 0.0))   # 俯角对准锚定块
                                                # (眼高在 feet 层锚块上方1.1格,
                                                #  平视命中 yo=2 石行——lw_dbg2 破案)
                if abs(dy) < 2 and abs(dp) < 2:
                    break
                a = dict(noop)
                a["camera_yaw"] = float(np.clip(dy, -15, 15))
                a["camera_pitch"] = float(np.clip(dp, -10, 10))
                obs, *_ = env.step(a)
            gt = None
            akey = "log" if args.log_wall else "iron_ore"
            for _try in range(2):               # 失败重建一次(迟到的区块)
                env.add_commands(build_cmds(wall_z, offsets))
                for _ in range(10):
                    obs, *_ = env.step(noop)
                gt, obs = anchor_gt_blocks(env, noop, offsets, anchor_key=akey)
                if gt is not None:
                    break
                time.sleep(2.0)
            if gt is None:
                print(f"[nat] ✗ {name} 锚定失败(地形不利),跳过", flush=True)
                continue
        pol = ObservePolicy(rng)
        frames, poses, ray_xyz, ray_key, ray_d = [], [], [], [], []
        from tests.integration.collect_calib640 import _ray
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
                             "natural": True, "pure_neg": bool(args.pure_neg)}))
        n_done += 1
        print(f"[nat] ✓ {name} T={len(frames)} {time.time()-t0:.0f}s "
              f"[{n_done} done]", flush=True)
    env.close()
    print(f"[nat] DONE {n_done} → {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/data/calib_nat")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--settle", type=int, default=25, help="spreadplayers 后等区块加载")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world_seed", default="natcalib1")
    p.add_argument("--pure_neg", action="store_true", help="纯自然负样本局")
    p.add_argument("--superflat", action="store_true",
                   help="超平坦世界(零地形风险;log 域差距由段2真树闭环兜底)")
    p.add_argument("--log_wall", action="store_true",
                   help="木簇墙模式(铁位→oak_log,GT键=log;止损自然树GT后的主正样本)")
    p.add_argument("--port", type=int, default=8770)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
