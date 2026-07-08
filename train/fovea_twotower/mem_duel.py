#!/usr/bin/env python3
"""M1 接口对决:读 Mamba 潜向量 vs 朴素运行统计 vs goal-only(慢→快条件通道的正面确认)。

命题(用户 2026-07-08):"快塔读慢塔 Mamba 潜向量而非文本序列,是否有独特优势?"
任务设计落在潜向量的理论优势区——历史依赖+当前不可见+亚符号:
  两相记忆重获取:相A(脚本带看 ~60 步)镜头扫过左/右墙,铁矿簇随机在一侧(另侧煤);
  相B(考试)镜头回中(铁不可见),指令"铁"——最优=凭记忆朝正确侧转。

三臂唯一差异=条件向量 cond_t(策略干同为反应式 MLP,杜绝注意力窗口偷看历史):
  goal    cond=0                          下界:同观测不同标签,BC 均值归零 → ~50%
  mamba   cond=小 Mamba2(从零,联合训练)在线吞逐步特征流的 h_t   正面确认对象
  runmean cond=同款逐步特征的累积均值(无参数,无 schema 的最强朴素基线)  独特性对照

预登记判据(先于结果):
  通道有效: mamba 相B初始转向正确率 ≥0.8 且 重获取步数 ≤0.7×goal-only;
  独特性:   mamba 显著优于 runmean(episode bootstrap CI 不含 0);
            若 mamba≈runmean → 如实记"有效但可被运行统计替代,独特性未证"。
  n=30/臂;评测课程序列由 --seed 固定并落盘 courses.json(审计修复:固定评测集)。
参考臂(不进主判):oracle 文本(側 one-hot 直给)——带 schema 的文本预期打平,
  其成本是"字段须被预先设计",在报告里陈述。

用法:
  DISPLAY=:99 ... PYTHONPATH=. .venv/bin/python train/fovea_twotower/mem_duel.py \
      --mode collect --episodes 100 --out runs/data/memduel
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/mem_duel.py --mode train --arm mamba
  DISPLAY=:99 ... --mode eval --arm mamba --episodes 30
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fovea_twotower.token_stream import CLASSES, TokenHead, as_hwc, wrap180
from tests.integration.collect_calib640 import _pose
from train.minecraft.vpt_action import CAMERA_BINS, bin_to_camera, camera_to_bin

CAM_NORM = 120.0
FEAT_DIM = 11                  # 逐步池化特征:加权cx/cy(铁),各类maxP(4),meanConf,n_act,goal onehot(3)
COND_DIM = 64
TOUR = ((-80, 18), (80, 18))   # 相A:先看 -x 侧墙(yaw≈-80... MC约定 yaw<0 朝+x)再另一侧
                               # 注意 MC 符号:yaw=-80 → forward_x=+sin80>0 → 看 +x 侧墙
PHASE_B_SCRAM = 0.0            # 相B回中(yaw→0,墙中央无矿,两侧均不可见)


def build_mem_course(side, wall_z=7):
    """铁簇嵌在 side(+1=+x/-1=-x)**侧墙** z≈1.5(方位角≈±76°,稳出 ±50° 半FOV),
    煤簇在对侧侧墙;前墙纯石。v1 教训:放前墙 ±3(方位 23°)相B直接可见,没隔离记忆。"""
    xi, xc = 6 * side, -6 * side
    return [
        "gamemode survival @p",
        "difficulty peaceful",
        "tp @p ~ ~ ~ 0 0",
        f"fill ~-6 ~-1 ~-3 ~6 ~4 ~{wall_z} minecraft:air",
        f"fill ~-6 ~-2 ~-3 ~6 ~-2 ~{wall_z} minecraft:stone",
        f"fill ~-6 ~-1 ~{wall_z} ~6 ~4 ~{wall_z} minecraft:stone",
        f"fill ~-6 ~-1 ~-3 ~-6 ~4 ~{wall_z} minecraft:stone",   # 左侧墙
        f"fill ~6 ~-1 ~-3 ~6 ~4 ~{wall_z} minecraft:stone",     # 右侧墙
        f"setblock ~{xi} ~ ~1 minecraft:iron_ore",
        f"setblock ~{xi} ~1 ~1 minecraft:iron_ore",
        f"setblock ~{xi} ~ ~2 minecraft:iron_ore",
        f"setblock ~{xc} ~ ~1 minecraft:coal_ore",
        f"setblock ~{xc} ~1 ~1 minecraft:coal_ore",
        "clear @p",
    ]


def step_feat(toks, goal_idx):
    """[K,10] token → [11] 逐步池化特征(供条件编码器;策略仍吃原始 token)。"""
    p = toks[:, 6:]                                # [K,4]
    w = p[:, 0] * (toks[:, 4] > 0)                 # 铁概率加权
    sw = w.sum() + 1e-6
    f = np.zeros(FEAT_DIM, np.float32)
    f[0] = float((w * toks[:, 0]).sum() / sw)      # 铁加权 cx(亚符号"在哪侧")
    f[1] = float((w * toks[:, 1]).sum() / sw)
    f[2:6] = p.max(0)
    f[6] = float(toks[:, 4].mean())
    f[7] = float((toks[:, 4] > 0).sum()) / len(toks)
    f[8 + goal_idx] = 1.0
    return f


def tour_action(t, noop, yaw_now):
    """相A脚本:按 TOUR 时间表转头驻留;返回 None 表示相A结束。"""
    acc = 0
    for tgt, dwell in TOUR:
        if t < acc + dwell:
            a = dict(noop)
            a["camera_yaw"] = float(np.clip(0.5 * wrap180(tgt - yaw_now), -18, 18))
            return a
        acc += dwell
    if t < acc + 10:                               # 回中
        a = dict(noop)
        a["camera_yaw"] = float(np.clip(0.5 * wrap180(0 - yaw_now), -18, 18))
        return a
    return None


# ── 模型:反应式策略 + 可换条件通道 ──────────────────────────────────
class MambaCond(nn.Module):
    """从零小 Mamba2 编码器:逐步特征流 → h_t(联合训练=可微记忆学"记什么")。"""

    def __init__(self, d=COND_DIM, layers=2):
        super().__init__()
        from mamba_ssm import Mamba2
        self.inp = nn.Linear(FEAT_DIM, d)
        self.blocks = nn.ModuleList([Mamba2(d_model=d, d_state=32, headdim=16)
                                     for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

    def forward(self, feats):                      # [B,T,11] → [B,T,d]
        x = self.inp(feats)
        for blk, nm in zip(self.blocks, self.norms):
            x = x + blk(nm(x))
        return x


class RunMeanCond(nn.Module):
    """朴素对照:累积均值(无参数记忆)→ 线性投影对齐维度。"""

    def __init__(self, d=COND_DIM):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, d)

    def forward(self, feats):                      # [B,T,11]
        csum = feats.cumsum(1)
        cnt = torch.arange(1, feats.shape[1] + 1, device=feats.device
                           ).view(1, -1, 1).float()
        return self.proj(csum / cnt)


class MemPolicy(nn.Module):
    """反应式:goal 交叉注意选槽 + [槽表征|上步动作|cond] → MLP → 相机bins+键。
    无时序干=条件通道是历史信息唯一入口(实验隔离的关键)。"""

    def __init__(self, arm, d=192, n_keys=20):
        super().__init__()
        self.arm = arm
        self.slot = nn.Linear(8, d)
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.cond = (MambaCond() if arm == "mamba"
                     else RunMeanCond() if arm == "runmean" else None)
        cin = d + 22 + (COND_DIM if self.cond else 0) + (2 if arm == "oracle" else 0)
        self.mlp = nn.Sequential(nn.Linear(cin, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU())
        self.cam = nn.Linear(d, 2 * CAMERA_BINS)
        self.key = nn.Linear(d, n_keys)

    def forward(self, rel_toks, prev_a, feats, oracle_side=None):
        # rel_toks [B,T,K,8], prev_a [B,T,22], feats [B,T,11]
        s = self.slot(rel_toks)                                   # [B,T,K,d]
        att = (s @ self.q).softmax(-1)                            # [B,T,K]
        sel = (att[..., None] * s).sum(2)                         # [B,T,d]
        parts = [sel, prev_a]
        if self.cond is not None:
            parts.append(self.cond(feats))
        if self.arm == "oracle":
            parts.append(oracle_side)                             # [B,T,2] one-hot
        h = self.mlp(torch.cat(parts, -1))
        return self.cam(h).view(*h.shape[:2], 2, CAMERA_BINS), self.key(h)


# ── collect:记忆教师示范 ─────────────────────────────────────────────
def run_collect(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from net.fovea_twotower.token_stream import TokenTeacher

    os.makedirs(args.out, exist_ok=True)
    th = TokenHead(conv_head=args.conv_head)
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    rng = np.random.default_rng(args.seed)
    gcls = CLASSES.index("iron_ore")
    n = 0
    for ep in range(args.episodes):
        name = f"md_s{args.seed}_e{ep}"
        outp = os.path.join(args.out, name + ".npz")
        if os.path.exists(outp):
            continue
        side = int(rng.choice([-1, 1]))
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_mem_course(side)})
        for _ in range(8):
            obs, *_ = env.step(noop)
        rgb = as_hwc(obs["rgb"])
        rec = {k: [] for k in ("toks", "feats", "dx", "dy", "keys", "phase")}
        # 相A:带看(脚本执行,教师标签=脚本动作)
        t = 0
        while True:
            a = tour_action(t, noop, _pose(obs["full"])[3])
            if a is None:
                break
            toks = th(rgb)
            rec["toks"].append(toks)
            rec["feats"].append(step_feat(toks, gcls))
            rec["dx"].append(a.get("camera_yaw", 0.0) / 0.15)
            rec["dy"].append(a.get("camera_pitch", 0.0) / 0.15)
            rec["keys"].append(np.zeros(20, np.uint8))
            rec["phase"].append(0)
            obs, *_ = env.step(a)
            rgb = as_hwc(obs["rgb"])
            t += 1
        # 相B:记忆教师——先凭"记忆"(=side,采集特权,信息在相A流里)转向,token 可见后交给 token 追踪
        tt = TokenTeacher(rng, epsilon=0.0)
        tt.new_segment()
        # 记忆教师:朝记住的一侧搜索。MC 符号:正 camera_yaw → yaw增 → 朝 -x;
        # side=+1(铁在+x)须负方向(v1 把符号写反,连同 first_turn 判据一起)
        tt.search_dir = float(-side)
        for tb in range(args.phase_b_steps):
            toks = th(rgb)
            a = tt(noop, toks, gcls, _pose(obs["full"])[4])
            rec["toks"].append(toks)
            rec["feats"].append(step_feat(toks, gcls))
            rec["dx"].append(a.get("camera_yaw", 0.0) / 0.15)
            rec["dy"].append(a.get("camera_pitch", 0.0) / 0.15)
            keys = np.zeros(20, np.uint8)
            keys[0] = bool(a.get("forward", False))
            rec["keys"].append(keys)
            rec["phase"].append(1)
            obs, *_ = env.step(a)
            rgb = as_hwc(obs["rgb"])
        np.savez_compressed(
            outp, toks=np.stack(rec["toks"]).astype(np.float16),
            feats=np.stack(rec["feats"]).astype(np.float16),
            dx=np.array(rec["dx"], np.float32), dy=np.array(rec["dy"], np.float32),
            keys=np.stack(rec["keys"]), phase=np.array(rec["phase"], np.int64),
            side=side)
        n += 1
        if n % 10 == 0:
            print(f"[md] {n} eps", flush=True)
    env.close()
    print(f"[md] DONE {n} → {args.out}", flush=True)


# ── train ────────────────────────────────────────────────────────────
def load_md(fp, gcls=0):
    z = np.load(fp, allow_pickle=True)
    toks = z["toks"].astype(np.float32)            # [T,K,10]
    T, K, _ = toks.shape
    geo = toks[..., :6]
    pg = toks[..., 6 + gcls:7 + gcls]
    other = toks[..., 6:].copy()
    other[..., gcls] = -1
    rel = np.concatenate([geo, pg, other.max(-1, keepdims=True)], -1)  # [T,K,8]
    act = np.zeros((T, 22), np.float32)
    act[:, 0] = np.clip(z["dx"] / CAM_NORM, -1, 1)
    act[:, 1] = np.clip(z["dy"] / CAM_NORM, -1, 1)
    act[:, 2:] = z["keys"]
    return (rel, z["feats"].astype(np.float32), act,
            z["phase"], int(z["side"]))


def run_train(args):
    dev = "cuda"
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    hold = max(1, min(6, len(files) // 4))
    tr_f, te_f = files[:-hold], files[-hold:]
    data = [load_md(f) for f in tr_f]
    pol = MemPolicy(args.arm).to(dev)
    print(f"[md:{args.arm}] {sum(p.numel() for p in pol.parameters())/1e6:.2f}M "
          f"| train {len(tr_f)} eps", flush=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    w_cam = None
    allb = torch.cat([camera_to_bin(torch.from_numpy(d[2][:, :2])) for d in data])
    cnt = torch.bincount(allb.flatten(), minlength=CAMERA_BINS).float()
    w_cam = ((cnt.sum() / (cnt + 1)).sqrt().clamp(max=3.0)).to(dev)
    for step in range(args.steps):
        bi = rng.choice(len(data), 8)
        Tm = min(data[i][0].shape[0] for i in bi)
        rel = torch.stack([torch.from_numpy(data[i][0][:Tm]) for i in bi]).to(dev)
        fea = torch.stack([torch.from_numpy(data[i][1][:Tm]) for i in bi]).to(dev)
        act = torch.stack([torch.from_numpy(data[i][2][:Tm]) for i in bi]).to(dev)
        orc = torch.stack([torch.tensor([1., 0.] if data[i][4] < 0 else [0., 1.])
                           .expand(Tm, 2) for i in bi]).to(dev)
        prev = torch.zeros_like(act)
        prev[:, 1:] = act[:, :-1]
        if rng.random() < 0.5:
            prev = prev * 0
        cam, key = pol(rel, prev, fea, orc)
        tgt = camera_to_bin(act[..., :2])
        loss = (F.cross_entropy(cam.flatten(0, 2).float(), tgt.flatten().to(dev),
                                weight=w_cam)
                + F.binary_cross_entropy_with_logits(key.float(), act[..., 2:]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"[md:{args.arm}] {step} loss={loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"pol": pol.state_dict(), "arm": args.arm}, args.ckpt)
    print(f"[md:{args.arm}] saved {args.ckpt}", flush=True)


# ── eval:闭环三判据 ──────────────────────────────────────────────────
def run_eval(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    pol = MemPolicy(ck["arm"]).to(dev).eval()
    pol.load_state_dict(ck["pol"])
    th = TokenHead(conv_head=args.conv_head)
    gcls = CLASSES.index("iron_ore")
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    rng = np.random.default_rng(args.seed)
    sides = [int(rng.choice([-1, 1])) for _ in range(args.episodes)]  # 固定课程序列
    res = []
    for ep, side in enumerate(sides):
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_mem_course(side)})
        for _ in range(8):
            obs, *_ = env.step(noop)
        rgb = as_hwc(obs["rgb"])
        rel_hist, fea_hist, prev_hist = [], [], [np.zeros(22, np.float32)]
        # 相A:脚本带看(策略只喂观测,不出动作)
        t = 0
        while True:
            a = tour_action(t, noop, _pose(obs["full"])[3])
            if a is None:
                break
            toks = th(rgb)
            rel_hist.append(_rel(toks, gcls))
            fea_hist.append(step_feat(toks, gcls))
            row = np.zeros(22, np.float32)
            row[0] = np.clip(a.get("camera_yaw", 0) / 0.15 / CAM_NORM, -1, 1)
            row[1] = np.clip(a.get("camera_pitch", 0) / 0.15 / CAM_NORM, -1, 1)
            prev_hist.append(row)
            obs, *_ = env.step(a)
            rgb = as_hwc(obs["rgb"])
            t += 1
        # 相B:策略接管
        yaw0 = _pose(obs["full"])[3]
        cum_yaw = 0.0
        lock_t = None
        for tb in range(args.phase_b_steps):
            toks = th(rgb)
            rel_hist.append(_rel(toks, gcls))
            fea_hist.append(step_feat(toks, gcls))
            with torch.no_grad():
                rel = torch.from_numpy(np.stack(rel_hist))[None].to(dev)
                fea = torch.from_numpy(np.stack(fea_hist))[None].to(dev)
                prev = torch.from_numpy(np.stack(prev_hist[:len(rel_hist)]))[None].to(dev)
                orc = torch.tensor([[1., 0.] if side < 0 else [0., 1.]]
                                   ).expand(1, rel.shape[1], 2).to(dev)
                cam, key = pol(rel, prev, fea, orc)
            cb = cam[0, -1].float().argmax(-1).cpu()
            val = bin_to_camera(cb).numpy() * CAM_NORM
            a = dict(noop)
            a["camera_yaw"] = float(np.clip(val[0] * 0.15, -18, 18))
            a["camera_pitch"] = float(np.clip(val[1] * 0.15, -18, 18))
            if torch.sigmoid(key[0, -1, 0]) > 0.5:
                a["forward"] = True
            row = np.zeros(22, np.float32)
            row[:2] = np.clip(val / CAM_NORM, -1, 1)
            row[2] = bool(a.get("forward", False))
            prev_hist.append(row)
            cum_yaw += a["camera_yaw"]
            if tb == 9:
                # MC 符号:朝 side=+1(+x) 的正确转向 = 负 cum_yaw(v1 判据反了)
                first_turn = np.sign(cum_yaw) == -np.sign(side)
            if lock_t is None and toks[:, 6 + gcls].max() > 0.5 \
                    and abs(toks[np.argmax(toks[:, 6 + gcls]), 0] - 0.5) < 0.1:
                lock_t = tb
            obs, *_ = env.step(a)
            rgb = as_hwc(obs["rgb"])
        res.append(dict(side=side, first_turn=bool(first_turn),
                        lock_t=lock_t if lock_t is not None else args.phase_b_steps))
        print(f"[ep{ep}] side={side:+d} turn={'✓' if first_turn else '✗'} "
              f"lock={res[-1]['lock_t']}", flush=True)
    env.close()
    ft = np.array([r["first_turn"] for r in res], float)
    lk = np.array([r["lock_t"] for r in res], float)
    boots = [np.mean(rng.choice(ft, len(ft))) for _ in range(2000)]
    out = dict(arm=ck["arm"], n=len(res),
               first_turn_rate=float(ft.mean()),
               first_turn_ci=[float(np.percentile(boots, 2.5)),
                              float(np.percentile(boots, 97.5))],
               lock_median=float(np.median(lk)), episodes=res)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[md:{ck['arm']}] first_turn={out['first_turn_rate']:.2f} "
          f"CI{out['first_turn_ci']} lock_med={out['lock_median']:.0f} → {args.out}",
          flush=True)


def _rel(toks, gcls):
    geo = toks[:, :6]
    pg = toks[:, 6 + gcls:7 + gcls]
    other = toks[:, 6:].copy()
    other[:, gcls] = -1
    return np.concatenate([geo, pg, other.max(-1, keepdims=True)],
                          -1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["collect", "train", "eval"], required=True)
    p.add_argument("--arm", choices=["goal", "mamba", "runmean", "oracle"],
                   default="mamba")
    p.add_argument("--data", default="runs/data/memduel")
    p.add_argument("--out", default="runs/data/memduel")
    p.add_argument("--ckpt", default="")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--phase_b_steps", type=int, default=60)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--conv_head", default="runs/g1_conv_head_v4.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8660)
    args = p.parse_args()
    if not args.ckpt:
        args.ckpt = f"runs/memduel_{args.arm}/best.pt"
    if args.mode == "collect":
        run_collect(args)
    elif args.mode == "train":
        run_train(args)
    else:
        if not args.out.endswith(".json"):
            args.out = f"runs/memduel_eval_{args.arm}.json"
        run_eval(args)


if __name__ == "__main__":
    main()
