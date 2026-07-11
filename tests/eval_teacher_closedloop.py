# -*- coding: utf-8 -*-
"""VPT 教师在 CraftGround 闭环的伐木能力评测——蒸馏路线的一次性定罪实验。

动机(2026-07-11 用户指令"确认快慢塔结构如何才能真的工作"):所有 GRPO run 的
oak_log 锚点全 0,病灶假设=快塔 attack 无持续性,解法=向 VPT rl-from-foundation-2x
教师蒸馏(bc_distill*)。**未验前提**:教师策略经我们的动作契约翻译层在本环境闭环
到底能不能砍下木头。能 ⇒ 契约无罪,蒸馏路线成立,剩下是数据/容量问题;
不能 ⇒ 病灶在环境/契约层,蒸馏堆数据无用。

两种动作解码模式(--mode):
  joint    联合采样(教师本尊口径):buttons 8641 类采样→因子化 20 键;camera 元动作
           开时对 121 类采样→mu-law undiscretize 出角度。保留键间相关结构。
  marginal 边缘采样(学生蒸馏目标口径):teacher_to_v2 的 p_keys 逐键 Bernoulli +
           remap_cam 后的 bin 分布逐轴 Categorical → bins_to_deg。
           = 学生若完美拟合蒸馏目标能达到的行为上限(丢键间相关)。
两个模式都过同一翻译层常量,联合成功而边缘失败 ⇒ 边缘化丢了技能,蒸馏目标要改。

锚点统计(不可刷,不进训练信号):inv_events 首次入包 tick、attack 占空比、
相机幅度、死亡截断。证据联络表图落 runs/teacher_closedloop/。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.minecraft.vpt_teacher import (_MAPPING, TEACHER_CAM_KWARGS,  # noqa: E402
                                         TEACHER_KEY_TO_V2, VPTTeacher,
                                         teacher_to_v2, remap_cam)
from net.vpt_lib.actions import CameraQuantizer                    # noqa: E402
from train.craftground.action_contract import V2_KEYS, bins_to_deg  # noqa: E402

OUT_DIR = Path("runs/teacher_closedloop")


def make_env(seed: str, port: int):
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.initial_environment_config import (Difficulty, GameMode,
                                                        WorldType)
    from craftground.screen_encoding_modes import ScreenEncodingMode
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        gamemode=GameMode.SURVIVAL, difficulty=Difficulty.PEACEFUL,
        world_type=WorldType.DEFAULT, seed=seed,
        screen_encoding_mode=ScreenEncodingMode.RAW, requires_heightmap=True)
    cfg.set_allow_mob_spawn(False); cfg.freeze_time(True); cfg.freeze_weather(True)
    return CraftGroundEnvironment(cfg,
                                  action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                                  port=port, find_free_port=True, verbose=False)


def sample_action_joint(pd: dict, quant: CameraQuantizer, rng) -> tuple[list, float, float]:
    """教师联合分布采样 → (V2 按键名列表, yaw_deg, pitch_deg)。"""
    pb = pd["buttons"][0].exp().numpy()
    b_idx = int(rng.choice(len(pb), p=pb / pb.sum()))
    factored = _MAPPING.BUTTON_IDX_TO_FACTORED[b_idx]              # 教师键序 0/1
    pressed = [k for i, k in enumerate(V2_KEYS) if factored[TEACHER_KEY_TO_V2[i]]]
    cam_on = not bool(_MAPPING.BUTTON_IDX_TO_CAMERA_META_OFF[b_idx])
    yaw = pitch = 0.0
    if cam_on:
        pc = pd["camera"][0].exp().numpy()
        c_idx = int(rng.choice(len(pc), p=pc / pc.sum()))
        pitch_bin, yaw_bin = divmod(c_idx, 11)                     # [pitch, yaw] 轴序
        pitch = float(quant.undiscretize(np.array([pitch_bin]))[0])
        yaw = float(quant.undiscretize(np.array([yaw_bin]))[0])
    return pressed, yaw, pitch


def sample_action_marginal(pd: dict, rng) -> tuple[list, float, float]:
    """蒸馏目标口径采样(边缘分布,丢键间相关)。"""
    p_keys, cam_t, _ = teacher_to_v2(pd)
    ours = remap_cam(cam_t)[0]                                     # [2, CAM_BINS]
    pressed = [k for i, k in enumerate(V2_KEYS)
               if rng.random() < float(p_keys[0, i])]
    pr = ours.numpy()
    bins = [int(rng.choice(pr.shape[-1], p=pr[a] / pr[a].sum())) for a in range(2)]
    deg = bins_to_deg(np.array(bins))
    return pressed, float(deg[0]), float(deg[1])


def contact_sheet(frames: list, path: Path) -> None:
    from PIL import Image
    sel = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).astype(int)]
    h, w = sel[0].shape[:2]
    sheet = np.zeros((2 * h, 4 * w, 3), np.uint8)
    for i, f in enumerate(sel):
        sheet[(i // 4) * h:(i // 4 + 1) * h, (i % 4) * w:(i % 4 + 1) * w] = f
    Image.fromarray(sheet).save(path)


def run_episode(env, teacher, no_op, mode: str, ticks: int, rng, quant) -> dict:
    obs, _ = env.reset()
    for _ in range(60):
        obs = env.step(no_op())[0]
    state = None
    inv_steps, frames_ev = {}, []
    key_count = {k: 0 for k in V2_KEYS}
    cam_abs, death_step = [], None
    t0 = time.time()
    for t in range(ticks):
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        if t % max(1, ticks // 24) == 0:
            frames_ev.append(np.asarray(cv2.resize(rgb, (160, 90))))
        small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        pd, state = teacher.forward_frames(torch.from_numpy(small[None]), state,
                                           first0=(t == 0))
        if mode == "joint":
            pressed, yaw, pitch = sample_action_joint(pd, quant, rng)
        else:
            pressed, yaw, pitch = sample_action_marginal(pd, rng)
        a = no_op()
        a["camera_yaw"], a["camera_pitch"] = yaw, pitch
        for k in pressed:
            a[k] = True
            key_count[k] += 1
        cam_abs.append(abs(yaw) + abs(pitch))
        obs = env.step(a)[0]
        full = obs["full"]
        for it in full.inventory:
            name = it.translation_key.split(".")[-1]
            if it.count > 0 and name not in inv_steps:
                inv_steps[name] = t
        if getattr(full, "is_dead", False):
            death_step = t
            break
    n = t + 1
    return dict(mode=mode, ticks_run=n, sps=round(n / (time.time() - t0), 1),
                inv_steps=inv_steps, death_step=death_step,
                key_duty={k: round(v / n, 4) for k, v in key_count.items() if v},
                cam_abs_mean=round(float(np.mean(cam_abs)), 3),
                frames=frames_ev)


def run_student_episode(env, student, no_op, ticks: int, device: str,
                        goal_text: str = "", slow_url: str = "",
                        slow_model: str = "") -> dict:
    """学生闭环(蒸馏验收锚点)。复用 grpo_pixel.rollout 的部署路径。

    goal 三臂:零 goal(默认)/固定词表短语(--goal-text)/真慢塔(--slow-url)。
    """
    from train.craftground.grpo_pixel import rollout, parse_slow_reply, SlowTower

    if slow_url:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        slow = SlowTower(slow_url, lambda xs: st.encode(xs, normalize_embeddings=True),
                         device, model=slow_model)
    elif goal_text:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        v = torch.as_tensor(st.encode([goal_text], normalize_embeddings=True)[0],
                            dtype=torch.float32)
        fixed = torch.cat([v, torch.tensor([0.5, 0.5])]).to(device)

        class _FixedSlow:                             # 词表内固定 goal(aim=画面中心)
            latencies: list = []
            fails = 0

            def __call__(self, rgb, state=""):
                rep = parse_slow_reply("")
                rep["subgoal"] = goal_text
                return (fixed, rep)
        slow = _FixedSlow()
    else:
        class _ZeroSlow:                              # 零指导桩:goal=0 向量,不联网
            latencies: list = []
            fails = 0

            def __call__(self, rgb, state=""):
                return (torch.zeros(384 + 2, device=device), parse_slow_reply(""))
        slow = _ZeroSlow()

    t0 = time.time()
    r = rollout(env, student, slow, no_op, np.random.default_rng(0),
                ticks, device, temp=1.0)
    n = int(r["imgs"].shape[0]) if hasattr(r["imgs"], "shape") else ticks
    keys = np.asarray(r["keys"])                      # [T, 20]
    duty = {k: round(float(keys[:, i].mean()), 4)
            for i, k in enumerate(V2_KEYS) if keys[:, i].any()}
    return dict(mode="student", ticks_run=int(keys.shape[0]),
                sps=round(keys.shape[0] / (time.time() - t0), 1),
                inv_steps=dict(r["inv_steps"]), death_step=r["death_step"],
                key_duty=duty,
                cam_abs_mean=round(float(np.abs(np.asarray(r["cam_deg"])).sum(-1).mean()), 3),
                frames=r["frames"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=1200)
    ap.add_argument("--mode", default="joint", choices=["joint", "marginal", "student"])
    ap.add_argument("--init-from", default="",
                    help="student 模式:PixelTower checkpoint(零 goal 闭环,蒸馏验收锚点)")
    ap.add_argument("--goal-text", default="",
                    help="student 模式:固定 goal 短语(MiniLM 编码+aim 中心;测词表内 goal 干预)")
    ap.add_argument("--slow-url", default="",
                    help="student 模式:真慢塔 base_url(测部署分布 goal 干预);与 --goal-text 互斥")
    ap.add_argument("--slow-model", default="qwen3_vl_8b_fp8")
    ap.add_argument("--tag", default="", help="student 模式结果文件后缀(默认取 ckpt 目录名)")
    ap.add_argument("--model", default="runs/data/models/vpt_teacher/2x.model")
    ap.add_argument("--weights",
                    default="runs/data/models/vpt_teacher/rl-from-foundation-2x.weights")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--seed-tries", type=int, default=8)
    ap.add_argument("--port", type=int, default=14300)
    args = ap.parse_args()

    from craftground.environment.action_space import no_op_v2
    rng = np.random.default_rng(args.seed)
    quant = CameraQuantizer(**TEACHER_CAM_KWARGS)
    teacher = student = None
    if args.mode == "student":
        # 学生零 goal 闭环 = 蒸馏验收锚点(got_log>0 才算起效);复用 grpo_pixel.rollout,
        # 慢塔用零指导桩(goal=0 向量),自标定照跑——与 GRPO 部署路径同代码。
        from net.pixel_tower import PixelTowerConfig, build_pixel_tower
        from train.craftground.grpo_pixel import IMG_HW
        from train.craftground.action_contract import CAM_BINS
        assert args.init_from, "student 模式必须 --init-from"
        cfg = PixelTowerConfig(img_hw=IMG_HW, goal_dim=384 + 2, n_keys=len(V2_KEYS),
                               camera_bins=CAM_BINS)
        student = build_pixel_tower(cfg).to(args.device)
        ck = torch.load(args.init_from, map_location=args.device, weights_only=True)
        student.load_state_dict(ck["tower"])
        print(f"student init {args.init_from} (bc_step={ck.get('step')})", flush=True)
    else:
        teacher = VPTTeacher(args.model, args.weights, device=args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.episodes):
        env, wseed, has_tree = None, "", False
        for att in range(args.seed_tries):                # 有树 seed 筛选(与 grpo 同款)
            wseed = str(int(rng.integers(0, 1 << 30)))
            env = make_env(wseed, args.port + ep)
            obs0, _ = env.reset()
            for _ in range(60):
                obs0 = env.step(no_op_v2())[0]
            has_tree = any("leaves" in h.block_name or "log" in h.block_name
                           for h in obs0["full"].height_info)
            if has_tree or att == args.seed_tries - 1:
                break
            env.close()
        if args.mode == "student":
            r = run_student_episode(env, student, no_op_v2, args.ticks, args.device,
                                    goal_text=args.goal_text, slow_url=args.slow_url,
                                    slow_model=args.slow_model)
        else:
            r = run_episode(env, teacher, no_op_v2, args.mode, args.ticks, rng, quant)
        env.close()
        frames = r.pop("frames")
        tag = args.tag or (Path(args.init_from).parent.name if args.init_from else args.mode)
        name = tag if args.mode == "student" else args.mode
        contact_sheet(frames, OUT_DIR / f"{name}_ep{ep}.png")
        r.update(episode=ep, world_seed=wseed, has_tree=has_tree)
        results.append(r)
        logs = {k: v for k, v in r["inv_steps"].items() if "log" in k}
        print(f"[ep{ep}] seed={wseed} tree={has_tree} ticks={r['ticks_run']} "
              f"sps={r['sps']} attack_duty={r['key_duty'].get('attack', 0)} "
              f"logs={logs or '无'} inv={list(r['inv_steps'])}", flush=True)

    got_log = sum(any("log" in k for k in r["inv_steps"]) for r in results)
    tag = args.tag or (Path(args.init_from).parent.name if args.init_from else args.mode)
    name = tag if args.mode == "student" else args.mode
    agg = dict(mode=args.mode, episodes=len(results), got_log=got_log,
               init_from=args.init_from or None,
               ticks=args.ticks, ts=time.strftime("%F %T"), per_episode=results)
    out = Path(f"docs/results/teacher_closedloop_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, ensure_ascii=False, indent=1))
    print(f"== {args.mode}: {got_log}/{len(results)} episodes 拿到 log → {out}", flush=True)


if __name__ == "__main__":
    main()
