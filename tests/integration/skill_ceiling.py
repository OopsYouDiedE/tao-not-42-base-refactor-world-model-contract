#!/usr/bin/env python3
"""技能上限终审:Minecraft 技能课程 × 闭环成功率(键鼠闭环,非代理指标)。

设计原则(承过去教训):**终审只认闭环技能成功率**,不看潜空间/探针/loss。每个技能有
确定性成功检测(obs['full'] 库存/统计/存活),先量教师(脚本示范)与基线(noop/random)的
成功率 → 环境天花板;再拿学到的快塔跑同一 harness → 快塔的学习上限。

技能课程(从易到难,均 setblock 摆确定目标 + 给对应工具,成功=闭环产物进背包):
  survive       空平坦生存,成功=N 步后未死
  chop_wood     正前方 oak_log 墙,徒手,成功=背包出现 log
  mine_stone    stone 墙 + 木镐,成功=cobblestone
  mine_iron     iron_ore 墙 + 石镐,成功=raw_iron(= 旧 C2)
  mine_diamond  diamond_ore 墙 + 铁镐,成功=diamond(感知更难:矿纹稀有 OOD)

策略:teacher(collect_s8 的 raycast 闭环采矿器,块无关=瞄准+挖)/ noop / random /
      learned(载 TrackNav/BC ckpt,后续接)。

用法(RAW 渲染):
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. ./.venv/bin/python \
      tests/integration/skill_ceiling.py --skill mine_iron --policy teacher \
      --episodes 20 --steps 220 --out runs/ceiling/mine_iron_teacher.json
"""
import argparse
import json
import os

import numpy as np
import torch

from tests.integration.collect_s8 import (DEG2PX, V2_KEYS, MineForwardPolicy,
                                          RandomPolicy, frame_pair)
from tests.integration.test_utils import crop128
from train.fovea_twotower.gate_fasthead import decode_action
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE

# 技能 = (目标方块 | None=生存, 工具 | None, 成功库存关键词子串)
SKILLS = {
    "survive":      dict(block=None,          tool=None,             success=None),
    "chop_wood":    dict(block="oak_log",     tool="wooden_axe",     success=["log", "plank"]),
    "mine_stone":   dict(block="stone",       tool="wooden_pickaxe", success=["cobblestone"]),
    "mine_iron":    dict(block="iron_ore",    tool="stone_pickaxe",  success=["iron"]),
    "mine_diamond": dict(block="diamond_ore", tool="iron_pickaxe",   success=["diamond"]),
}
WALL_Z = (5, 6, 7, 8, 9, 10)


def build_course(block, tool, wall_z):
    """通用课程:清空 + 正前方 z+wall_z 摆一面 block 墙(眼平中心一块正对准星)+ 给 tool。"""
    cmds = ["gamemode survival @p", "difficulty peaceful", "tp @p ~ ~ ~ 0 0",
            f"fill ~-3 ~-1 ~-2 ~3 ~4 ~{wall_z} minecraft:air",
            f"fill ~-3 ~-2 ~-2 ~3 ~-2 ~{wall_z} minecraft:stone", "clear @p"]
    if block:
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (-1, 1)]:
            cmds.append(f"setblock ~{dx} ~{dy} ~{wall_z} minecraft:{block}")
        cmds.insert(4, f"fill ~-3 ~-1 ~{wall_z} ~3 ~4 ~{wall_z} minecraft:stone")  # 后墙托底
    if tool:
        cmds.append(f"item replace entity @p weapon.mainhand with minecraft:{tool} 1")
    return cmds


def _np_rgb(rgb):
    """ZEROCOPY 下 obs['rgb'] 是 CUDA uint8 张量;RAW 下是 numpy。统一成 HWC numpy。"""
    return rgb.cpu().numpy() if hasattr(rgb, "cpu") else rgb


def inv_count(full, keys):
    """背包里 translation_key 含任一子串的物品计数之和。"""
    n = 0
    try:
        for it in full.inventory:
            if getattr(it, "count", 0) > 0 and any(k in (it.translation_key or "") for k in keys):
                n += it.count
    except Exception:  # noqa
        pass
    return n


class LearnedFastHead:
    """载 BCPolicy(dinov3)快塔,逐帧 crop128→encode_frames→因果时序头→decode_action。

    每 episode(t==0)重置历史窗;跨 episode 复用同一模型(免重复载权重)。视觉闭环控制。
    """

    def __init__(self, ckpt, device, max_len, greedy, rng, backbone="dinov3"):
        from net.bc import BCConfig, build_bc_policy
        from net.config import BackboneConfig
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        cs = ck.get("cfg", {})
        cfg = BCConfig(backbone=BackboneConfig(kind=backbone), d=cs.get("d", 384),
                       heads=cs.get("heads", 6), layers=cs.get("layers", 4), dropout=0.0,
                       max_len=max(128, max_len), action_dim=ACTION_DIM,
                       n_mouse=N_MOUSE, camera_bins=CAMERA_BINS)
        self.policy = build_bc_policy(cfg).to(device).eval()
        missing, unexpected = self.policy.load_state_dict(ck["policy"], strict=False)
        assert not [m for m in missing if not m.startswith("backbone.")], missing[:6]
        assert not unexpected, unexpected[:6]
        self.device, self.max_len, self.greedy, self.rng = device, max_len, greedy, rng
        print(f"✅ learned 快塔载入 {ckpt}(step={ck.get('step')})", flush=True)
        self._reset()

    def _reset(self):
        self.fh, self.ah = [], []
        self.prev = np.zeros(ACTION_DIM, np.float32)

    @torch.no_grad()
    def __call__(self, t, noop, obs=None):
        if t == 0:
            self._reset()
        f = self.policy.encode_frames(crop128(_np_rgb(obs["rgb"])).to(self.device))[:, 0]
        self.fh.append(f)
        self.ah.append(torch.from_numpy(self.prev).to(self.device).view(1, -1))
        fseq = torch.stack(self.fh[-self.max_len:], 1)
        aseq = torch.stack(self.ah[-self.max_len:], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            cam, key = self.policy(fseq.float(), aseq.float())
        a, key_on = decode_action(cam[0, -1], key[0, -1], noop, self.greedy, self.rng)
        self.prev = np.zeros(ACTION_DIM, np.float32)
        self.prev[N_MOUSE:] = key_on
        return a


def make_policy(name, rng, device="cpu", ckpt=None, max_len=64, greedy=False, backbone="dinov3"):
    if name == "teacher":
        return MineForwardPolicy(rng, epsilon=0.0)          # 纯闭环(无探索噪声)
    if name == "random":
        return RandomPolicy(rng)
    if name == "noop":
        return lambda t, noop, obs=None: dict(noop)
    if name == "learned":
        assert ckpt, "learned 需 --ckpt"
        return LearnedFastHead(ckpt, device, max_len, greedy, rng, backbone)
    raise ValueError(f"未知 policy {name}")


def run(skill, policy_name, episodes, steps, settle, port, seed, out,
        ckpt=None, max_len=64, greedy=False, backbone="dinov3", save_demo=None):
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    sk = SKILLS[skill]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="ceiling", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=port, verbose=False)
    noop = no_op_v2()
    env.reset()
    rng = np.random.default_rng(seed)
    # learned 快塔跨 episode 复用同一模型(t==0 内部重置);teacher/random 每局新建
    persistent = (make_policy(policy_name, rng, device, ckpt, max_len, greedy, backbone)
                  if policy_name == "learned" else None)
    rows = []
    for ep in range(episodes):
        wall_z = WALL_Z[ep % len(WALL_Z)]
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_course(sk["block"], sk["tool"], wall_z)})
        for _ in range(settle):
            obs, *_ = env.step(noop)
        policy = persistent or make_policy(policy_name, rng)
        base = inv_count(obs["full"], sk["success"]) if sk["success"] else 0
        died = False
        frames, dxs, dys, keys_l = [], [], [], []            # demo 录制(save_demo 时)
        for t in range(steps):
            if save_demo:
                frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])   # [3,126,126] u8
            a = policy(t, noop, obs)
            if save_demo:
                dxs.append(float(a.get("camera_yaw", 0.0)) * DEG2PX)
                dys.append(float(a.get("camera_pitch", 0.0)) * DEG2PX)
                keys_l.append([float(bool(a.get(k, False))) for k in V2_KEYS])
            obs, *_ = env.step(a)
            if bool(getattr(obs["full"], "is_dead", False)):
                died = True
                break
        if sk["success"] is None:
            ok = not died                                    # 生存技能
            got = 0
        else:
            got = inv_count(obs["full"], sk["success"]) - base
            ok = got > 0
        if save_demo and frames and (not sk["success"] or ok):   # 只存成功示范(clean BC)
            frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])  # 末帧无动作(s8 契约:frames T,dx T-1)
            os.makedirs(save_demo, exist_ok=True)
            np.savez_compressed(
                os.path.join(save_demo, f"{skill}_ep{ep:03d}.npz"),
                frames=np.stack(frames).astype(np.uint8),
                dx=np.array(dxs, np.float32), dy=np.array(dys, np.float32),
                keys=np.array(keys_l, np.uint8),
                gui=np.zeros(len(dxs), np.uint8),
                score=np.float32(max(got, 1)), policy_strong=np.int64(1),
                start_hard=np.int64(ep % 2))
        rows.append({"ep": ep, "wall_z": wall_z, "success": bool(ok), "got": int(got), "died": died})
        print(f"[{skill}/{policy_name}] ep{ep} wall_z={wall_z} success={ok} got={got}", flush=True)
    env.close()
    rate = float(np.mean([r["success"] for r in rows]))
    res = {"skill": skill, "policy": policy_name, "episodes": episodes, "steps": steps,
           "success_rate": round(rate, 4),
           "mean_got": round(float(np.mean([r["got"] for r in rows])), 3),
           "per_episode": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[{skill}/{policy_name}] SUCCESS_RATE={rate:.3f} → {out}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill", choices=list(SKILLS), required=True)
    p.add_argument("--policy", default="teacher", help="teacher/noop/random/learned")
    p.add_argument("--ckpt", default=None, help="learned 快塔 ckpt(BCPolicy)")
    p.add_argument("--backbone", default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--greedy", action="store_true", default=False)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--save_demo", default=None, help="存教师示范(s8 格式)到该目录,供 BC")
    args = p.parse_args()
    out = args.out or f"runs/ceiling/{args.skill}_{args.policy}.json"
    run(args.skill, args.policy, args.episodes, args.steps, args.settle,
        args.port, args.seed, out, args.ckpt, args.max_len, args.greedy, args.backbone,
        args.save_demo)


if __name__ == "__main__":
    main()
