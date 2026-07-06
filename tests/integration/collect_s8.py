#!/usr/bin/env python3
"""S8 轨迹采集器:CraftGround CPU 渲染跑 2×2 设计(策略强弱 × 起点难易)→ eval_s8 契约 npz。

两阶段(采集纯 CPU、latify 用 GPU,互不阻塞):
  --mode collect   Xvfb+llvmpipe 软件渲染(0 显存,可与 GPU 训练并行;见
                   [CraftGround 纯CPU渲染] 记忆),每条轨迹存 raw npz:
                     frames  u8  [T,3,126,126]  中心裁剪 126(与 Gaming500Dataset
                                                img_size=126 crop=center 同口径)
                     gray    u8  [T,45,80]      周边消息用低清灰度(_periph_msgs 口径)
                     dx,dy   f32 [T-1]          指令相机增量(deg→px:/0.15,VPT 约定)
                     keys    u8  [T-1,20]       KEY_NAMES 位序(V2 布尔键直映)
                     gui     u8  [T-1]           inventory 开=1
                     dt      f32 [T-1]           恒 2(匹配 15Hz 训练口径,30Hz 源/2)
                     score/start_id/policy_strong/start_hard/meta
  --mode latify    raw → eval_s8.encode_traj_dir 契约:{lat[T,81,384] fp16(dino_encode),
                   act[T-1,24](act_featurize),msg[T,11](_periph_msgs),score,...}

2×2 设计(S8b 存亡命题的对抗结构,step5 doc §2):
  策略:strong=MineForward 脚本(C2:对准前方矿脉、推进、持续攻击;--epsilon 掺随机)
        weak=均匀随机;
  起点:easy=完整课程(runs/curriculum_repl/courses/C2_cave_mine_iron.json,手持石镐)
        hard=同课程但剥离 head-start(拿走石镐 → 徒手采铁掉落为零,强策略也难得分)。
  score=C2 口径:结束时背包 raw_iron+iron_ore 计数(dump_inventory translation_key)。

用法(先冒烟 1 条,再全量;多进程并行=多端口多实例,产物同目录汇合):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. python tests/integration/collect_s8.py \
      --mode collect --n-per-cell 1 --steps 64 --out runs/data/s8_raw          # 冒烟
  PYTHONPATH=. python tests/integration/collect_s8.py \
      --mode latify --raw runs/data/s8_raw --out runs/data/s8_traj             # GPU
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from train.gaming500.dataset import KEY_NAMES

# V2 动作布尔键 ←→ KEY_NAMES 位序:移动键 OS 名≠V2 语义名,须显式映射;其余仅去前缀
_K2V = {"w": "forward", "a": "left", "s": "back", "d": "right", "space": "jump"}
V2_KEYS = [_K2V.get(k[len("key_"):], k[len("key_"):]) for k in KEY_NAMES]
DEG2PX = 1.0 / 0.15          # VPT 约定:1px ≈ 0.15deg;记录还原成 px 口径
COURSE = "runs/curriculum_repl/courses/C2_cave_mine_iron.json"
HEADSTART_MARK = "stone_pickaxe"   # hard 起点=剥离含此标记的命令(拿走工具)


# ── 策略 ──────────────────────────────────────────────────────────────
class RandomPolicy:
    """weak:每步独立采样;概率偏 forward/attack 但无方向性。"""
    strong = 0

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, t, noop):
        a = dict(noop)
        a["forward"] = bool(self.rng.random() < 0.3)
        a["jump"] = bool(self.rng.random() < 0.1)
        a["attack"] = bool(self.rng.random() < 0.3)
        a["camera_yaw"] = float(self.rng.normal(0, 15))
        a["camera_pitch"] = float(self.rng.normal(0, 8))
        return a


class MineForwardPolicy:
    """strong(C2 口径):课程把矿脉摆在正前方 z+7、视线初始化 (0,0)——
    前进数步到墙前,然后持续攻击;小幅扫视覆盖 2×2 脉面。--epsilon 概率换随机动作
    (给分数造梯度,score 不再二值)。"""
    strong = 1

    def __init__(self, rng, epsilon=0.15):
        self.rng = rng
        self.eps = epsilon
        self._rand = RandomPolicy(rng)

    def __call__(self, t, noop):
        if self.rng.random() < self.eps:
            return self._rand(t, noop)
        a = dict(noop)
        if t < 10:                                   # 走近墙面
            a["forward"] = True
        else:                                        # 贴墙持续挖,缓慢扫过脉面
            a["attack"] = True
            a["camera_yaw"] = float(6 * np.sin(t / 25.0))
            a["camera_pitch"] = float(3 * np.sin(t / 40.0))
        return a


# ── 观测/记账 ─────────────────────────────────────────────────────────
def frame_pair(rgb, size=126):
    """obs rgb(CHW 或 HWC)→ (frame u8 [3,126,126] 中心裁剪, gray u8 [45,80])。"""
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)                 # → HWC
    h, w = arr.shape[:2]
    s = min(h, w)
    crop = arr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    frame = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), (80, 45),
                      interpolation=cv2.INTER_AREA)
    return frame.transpose(2, 0, 1).copy(), gray


def act_row(a):
    """V2 动作 dict → (dx_px, dy_px, keys20 u8, gui u8)。"""
    keys = np.array([bool(a.get(k, False)) for k in V2_KEYS], np.uint8)
    return (float(a.get("camera_yaw", 0.0)) * DEG2PX,
            float(a.get("camera_pitch", 0.0)) * DEG2PX,
            keys, np.uint8(bool(a.get("inventory", False))))


def score_c2(full_obs):
    """C2 计分:背包 raw_iron + iron_ore 总数。"""
    n = 0
    try:
        for it in full_obs.inventory:
            if getattr(it, "count", 0) > 0 and "iron" in (it.translation_key or ""):
                n += it.count
    except Exception:  # noqa
        pass
    return float(n)


# ── collect ──────────────────────────────────────────────────────────
def run_collect(args):
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    course = json.load(open(args.course))["commands"]
    course_hard = [c for c in course if HEADSTART_MARK not in c]
    os.makedirs(args.out, exist_ok=True)

    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,   # 软件 GL,无 CUDA 互操作
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    print(f"[collect] env up (port {args.port}) → {args.out}", flush=True)

    rng = np.random.default_rng(args.seed)
    cells = [(ps, sh) for ps in (1, 0) for sh in (0, 1)]   # (policy_strong, start_hard)
    n_done = 0
    for ep in range(args.n_per_cell):
        for ps, sh in cells:
            name = f"c2_p{ps}h{sh}_s{args.seed}_e{ep}"
            outp = os.path.join(args.out, name + ".npz")
            if os.path.exists(outp):
                continue
            t0 = time.time()
            cmds = course_hard if sh else course
            obs, _ = env.reset(options={"fast_reset": True,
                                        "extra_commands": list(cmds)})
            for _ in range(args.settle):               # 命令异步,留沉降帧
                obs, *_ = env.step(noop)
            pol = (MineForwardPolicy(rng, args.epsilon) if ps
                   else RandomPolicy(rng))
            frames, grays, rows = [], [], []
            f0, g0 = frame_pair(obs["rgb"])
            frames.append(f0)
            grays.append(g0)
            for t in range(args.steps):
                a = pol(t, noop)
                obs, *_ = env.step(a)
                fr, gr = frame_pair(obs["rgb"])
                frames.append(fr)
                grays.append(gr)
                rows.append(act_row(a))
            sc = score_c2(obs["full"])
            dx = np.array([r[0] for r in rows], np.float32)
            dy = np.array([r[1] for r in rows], np.float32)
            keys = np.stack([r[2] for r in rows])
            gui = np.array([r[3] for r in rows], np.uint8)
            np.savez_compressed(
                outp, frames=np.stack(frames), gray=np.stack(grays),
                dx=dx, dy=dy, keys=keys, gui=gui,
                dt=np.full(len(rows), 2.0, np.float32),
                score=sc, start_id=int(args.seed * 1000 + ep),
                policy_strong=ps, start_hard=sh,
                meta=json.dumps({"course": os.path.basename(args.course),
                                 "epsilon": args.epsilon, "steps": args.steps}))
            n_done += 1
            print(f"[collect] ✓ {name} score={sc} T={len(frames)} "
                  f"{time.time()-t0:.0f}s [{n_done} done]", flush=True)
    env.close()
    print(f"[collect] DONE {n_done} trajs → {args.out}", flush=True)


# ── latify ───────────────────────────────────────────────────────────
def run_latify(args):
    import torch
    from train.fovea_twotower.train_r1 import dino_encode
    from net.fovea_twotower.tower import act_featurize
    from train.gaming500.dataset import _periph_msgs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.raw, "*.npz")))
    assert files, f"{args.raw} 下无 raw npz"
    done = skip = 0
    for fp in files:
        outp = os.path.join(args.out, os.path.basename(fp))
        if os.path.exists(outp):
            skip += 1
            continue
        z = np.load(fp)
        with torch.no_grad():
            lat = dino_encode(dino, torch.from_numpy(z["frames"]).to(dev))
            act = act_featurize(*(torch.from_numpy(np.asarray(z[k], np.float32))
                                  for k in ("dx", "dy", "keys", "gui", "dt")))
        msg = _periph_msgs(list(z["gray"]))
        np.savez_compressed(
            outp, lat=lat.float().cpu().numpy().astype(np.float16),
            act=act.float().numpy().astype(np.float16), msg=msg,
            score=z["score"], start_id=z["start_id"],
            policy_strong=z["policy_strong"], start_hard=z["start_hard"])
        done += 1
        print(f"[latify] ✓ {os.path.basename(fp)} T={len(z['frames'])} "
              f"[{done} done/{skip} skip]", flush=True)
    print(f"[latify] DONE {done}/{skip} skip → {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["collect", "latify"], default="collect")
    p.add_argument("--out", default="runs/data/s8_raw")
    p.add_argument("--course", default=COURSE)
    p.add_argument("--n-per-cell", type=int, default=8, help="2×2 每格轨迹数")
    p.add_argument("--steps", type=int, default=240, help="每轨迹步数(软渲染 ~1.6s/帧)")
    p.add_argument("--settle", type=int, default=10, help="课程命令沉降帧")
    p.add_argument("--epsilon", type=float, default=0.15, help="强策略的随机掺入")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8531)
    p.add_argument("--raw", default="runs/data/s8_raw", help="latify 输入")
    args = p.parse_args()
    (run_collect if args.mode == "collect" else run_latify)(args)


if __name__ == "__main__":
    main()
