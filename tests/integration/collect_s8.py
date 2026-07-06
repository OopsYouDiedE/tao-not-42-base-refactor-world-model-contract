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


def build_c2_course(wall_z=7, pickaxe=True):
    """参数化 C2 课程(wall_z=7 逐字复现 courses/C2_cave_mine_iron.json)。

    wall_z 变墙距=变**起点**(DINO 潜空间外观 + 接近距离都随之变):S8a 需多个可区分起点
    (start_id=variant*2+hard),墙距是最省的起点变量,强策略靠 raycast 自适应仍能采到。
    pickaxe=False → hard 起点:不发石镐,徒手挖铁矿掉落为零(score 恒 0,S8b 难起点)。"""
    cmds = [
        "gamemode survival @p",
        "difficulty peaceful",
        "tp @p ~ ~ ~ 0 0",
        f"fill ~-3 ~-1 ~-2 ~3 ~4 ~{wall_z} minecraft:air",
        f"fill ~-3 ~-2 ~-2 ~3 ~-2 ~{wall_z} minecraft:stone",       # 地板
        f"fill ~-3 ~-1 ~{wall_z} ~3 ~4 ~{wall_z} minecraft:stone",  # 后墙
        f"setblock ~0 ~ ~{wall_z} minecraft:iron_ore",              # 眼平中心矿(强策略必挖点)
        f"setblock ~1 ~ ~{wall_z} minecraft:iron_ore",
        f"setblock ~-1 ~ ~{wall_z} minecraft:iron_ore",
        f"setblock ~0 ~1 ~{wall_z} minecraft:iron_ore",
        f"setblock ~-1 ~1 ~{wall_z} minecraft:iron_ore",
        "clear @p",
    ]
    if pickaxe:
        cmds.append("item replace entity @p weapon.mainhand with minecraft:stone_pickaxe 1")
    return cmds


WALL_Z_VARIANTS = (5, 6, 7, 8, 9, 10)   # 起点变体的墙距序列


# ── 策略 ──────────────────────────────────────────────────────────────
class RandomPolicy:
    """weak:每步独立采样;概率偏 forward/attack 但无方向性。"""
    strong = 0

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, t, noop, obs=None):
        a = dict(noop)
        a["forward"] = bool(self.rng.random() < 0.3)
        a["jump"] = bool(self.rng.random() < 0.1)
        a["attack"] = bool(self.rng.random() < 0.3)
        a["camera_yaw"] = float(self.rng.normal(0, 15))
        a["camera_pitch"] = float(self.rng.normal(0, 8))
        return a


def _raycast(full):
    """(命中方块 translation_key 或 "", 玩家眼到方块中心的欧氏距离 或 大数)。"""
    import math
    try:
        tb = full.raycast_result.target_block
        key = tb.translation_key or ""
        if not key:
            return "", 1e9
        d = math.sqrt((tb.x + 0.5 - full.x) ** 2 + (tb.y + 0.5 - full.y) ** 2
                      + (tb.z + 0.5 - full.z) ** 2)
        return key, d
    except Exception:  # noqa
        return "", 1e9


class MineForwardPolicy:
    """strong(C2 口径):闭环采矿。课程把铁矿脉摆在正前方 z+7,眼平 (x0,y0) 一块正对
    准星,视线初始化 (0,0)。用 raycast_result 反馈,分两相:
      · 接近:准星命中的方块还够不到(dist>REACH)→ forward 走近,pitch 归零;
      · 采矿:够得到(dist≤REACH)→ **停 forward**(旧版全程 forward 会把玩家顶穿
        单层矿墙、把 raw_iron 掉落甩在身后拾取不到=score=0 病根),attack 持续挖。
        **准星命中 iron_ore 时锁死相机**(破坏需同格连续 ~23 tick,任何移动都会清零),
        破完该矿→raycast 不再是矿→按 SCAN 抬 pitch 找上一列 y1 矿,如此逐块吃掉脉面;
        每几步补一记 forward 轻触,把落地矿吸进 1 格拾取半径。
    --epsilon 概率换随机动作(给 score 造梯度)。"""
    strong = 1
    REACH = 3.2                          # 可挖距离(生存 ~4.5,留裕度确保贴脸)
    SCAN = (0.0, -12.0, -24.0, -36.0)    # 未命中矿时的 pitch 搜索档(负=上抬,扫 y0→y1)
    SCAN_DWELL = 22                       # 每档停留(> 破坏时长才不空扫)
    FWD_BUDGET = 16                       # **到墙后**允许的 forward 步数上限:够钻穿单层墙+吸拾,
                                          # 用尽即锁死(否则破洞后 raycast 见远处→无限前进跑出矿脉丢掉落)

    def __init__(self, rng, epsilon=0.15):
        self.rng = rng
        self.eps = epsilon
        self._rand = RandomPolicy(rng)
        self._pitch = 0.0                # 当前累计 pitch(相对初始 0)
        self._scan_i = 0
        self._scan_t = 0
        self._arrived = False            # 首次够到墙后置位:此后不再无限接近
        self._fwd_left = self.FWD_BUDGET

    def __call__(self, t, noop, obs=None):
        if self.rng.random() < self.eps:
            return self._rand(t, noop)
        a = dict(noop)
        full = obs["full"] if obs is not None else None
        if full is None:                             # 无反馈兜底:走+挖
            a["forward"] = True
            a["attack"] = True
            return a
        key, dist = _raycast(full)
        near = dist <= self.REACH
        if near:
            self._arrived = True
        # 偏航自纠:课程令玩家正对墙(yaw≈0),epsilon 随机偏航会累积把朝向带偏→forward 走歪、
        # 采矿脱靶(校准 e0 score=0 病根)。用 full.yaw 反馈把朝向拉回 0(命中矿锁死时不纠,免脱靶)。
        yaw = float(getattr(full, "yaw", 0.0))
        dyaw = float(np.clip(-0.5 * yaw, -20.0, 20.0))
        if not self._arrived:                        # ── 接近相(仅到墙前):走近,准星归眼平+朝向归零 ──
            a["forward"] = True
            a["camera_pitch"] = float(-self._pitch)
            a["camera_yaw"] = dyaw
            self._pitch = 0.0
            return a
        # ── 采矿相(已到墙):停走为主,持续挖,逐块吃脉面,落地矿靠限额 forward 吸拾 ──
        a["attack"] = True
        if near and "iron_ore" in key:               # 命中矿:锁死相机专心破坏
            a["camera_pitch"] = 0.0
            self._scan_t = 0
        elif near:                                   # 够到但非矿:抬头搜索脉面 + 朝向归零
            a["camera_yaw"] = dyaw
            self._scan_t += 1
            if self._scan_t >= self.SCAN_DWELL:
                self._scan_t = 0
                self._scan_i = (self._scan_i + 1) % len(self.SCAN)
                target = self.SCAN[self._scan_i]
                a["camera_pitch"] = float(target - self._pitch)
                self._pitch = target
        else:                                        # 前方已空(钻穿/无块):限额内前进吸拾+朝向归零
            a["camera_yaw"] = dyaw
            if self._fwd_left > 0:
                a["forward"] = True
                self._fwd_left -= 1
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


def _mined_iron(full_obs):
    """诊断:mined_statistics 里破坏过的 iron_ore 计数(map<key,int>)。"""
    try:
        ms = full_obs.mined_statistics
        return int(sum(v for k, v in dict(ms).items() if "iron" in k))
    except Exception:  # noqa
        return -1


# ── collect ──────────────────────────────────────────────────────────
def run_collect(args):
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    variants = WALL_Z_VARIANTS[:args.variants]
    os.makedirs(args.out, exist_ok=True)

    from craftground.initial_environment_config import WorldType
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,   # 软件 GL,无 CUDA 互操作
        world_type=WorldType.SUPERFLAT, seed="s8fovea",  # 平坦+定 seed:出生点确定,课程房间每回合一致(DEFAULT 会嵌进山体/洞穴)
        request_raycast=True,                          # populate raycast_result:闭环采矿靠它对准/判距(默认 False=矿永挖不到)
        mined_stat_keys=["iron_ore"],                  # 诊断:mined_statistics 记录破坏的铁矿数(mod 自动补 minecraft: 前缀,勿重复)
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
        for v_idx, wall_z in enumerate(variants):
          for ps, sh in cells:
            name = f"c2_v{v_idx}_p{ps}h{sh}_s{args.seed}_e{ep}"
            outp = os.path.join(args.out, name + ".npz")
            if os.path.exists(outp):
                continue
            t0 = time.time()
            start_id = v_idx * 2 + sh                # (变体,难度)=一个"同起点"簇(S8a 桶)
            cmds = build_c2_course(wall_z, pickaxe=not sh)
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
                a = pol(t, noop, obs)
                obs, *_ = env.step(a)
                fr, gr = frame_pair(obs["rgb"])
                frames.append(fr)
                grays.append(gr)
                rows.append(act_row(a))
            sc = score_c2(obs["full"])
            mined_iron = _mined_iron(obs["full"])   # 诊断:破坏了几块铁矿(区分"没挖到"vs"没拾取")
            dx = np.array([r[0] for r in rows], np.float32)
            dy = np.array([r[1] for r in rows], np.float32)
            keys = np.stack([r[2] for r in rows])
            gui = np.array([r[3] for r in rows], np.uint8)
            np.savez_compressed(
                outp, frames=np.stack(frames), gray=np.stack(grays),
                dx=dx, dy=dy, keys=keys, gui=gui,
                dt=np.full(len(rows), 2.0, np.float32),
                score=sc, start_id=int(start_id),
                policy_strong=ps, start_hard=sh,
                meta=json.dumps({"course": "c2_parametric", "wall_z": wall_z,
                                 "variant": v_idx, "epsilon": args.epsilon,
                                 "steps": args.steps}))
            n_done += 1
            fz = obs["full"]
            print(f"[collect] ✓ {name} score={sc} mined_iron={mined_iron} "
                  f"pos=({fz.x:.1f},{fz.y:.1f},{fz.z:.1f}) T={len(frames)} "
                  f"{time.time()-t0:.0f}s [{n_done} done]", flush=True)
    env.close()
    print(f"[collect] DONE {n_done} trajs → {args.out}", flush=True)


# ── latify ───────────────────────────────────────────────────────────
def run_latify(args):
    import torch
    from train.fovea_twotower.data_utils import dino_encode
    from net.backbone import build_backbone
    from net.config import BackboneConfig
    from net.fovea_twotower.tower import act_featurize
    from train.gaming500.dataset import _periph_msgs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(dev).eval()
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
    p.add_argument("--course", default=COURSE, help="(已弃用:课程改内建参数化 build_c2_course)")
    p.add_argument("--variants", type=int, default=6,
                   help="起点变体数(墙距 WALL_Z_VARIANTS 前 N;每变体×难度=一个 S8a 起点桶)")
    p.add_argument("--n-per-cell", type=int, default=8, help="每(变体×2×2 格)轨迹数")
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
