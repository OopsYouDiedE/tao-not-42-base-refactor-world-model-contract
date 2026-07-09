#!/usr/bin/env python3
"""GRPO-Pixel:像素快塔 + Omni 慢塔 + Haiku 判官(2026-07-09 起的规范实现)。

三条用户裁决,写死在本文件的设计里:
  1. **以 GRPO 为规范。** 相对优势由判官排序给,不由手工程序统计给。
  2. **苦涩的教训。** 不为单个游戏打感知补丁:输入是原始像素,不是 YOLOE 解析槽位、
     不是手标 log/iron/coal/dirt 的分割头、不是 8 角凸包造的树干 GT。
     退役:net/fovea_twotower/token_stream.py + g1_conv_head + g1_vectors,
           net/bc/policy.py 的冻结 DINOv3 骨干。
  3. **不写死"该怎么做"。** 慢塔只给语义指示(文本子目标 + 目标像素);
     "像素 → 相机增量"这个连续标定由快塔**学**出来,不许用 FOV 几何公式代劳。

数据流:
    CraftGround 原始像素 640x360 --下采样--> [3,90,160]
        ↓                                    ↑ goal_vec[386] = MiniLM(subgoal)[384] ⊕ aim/1000[2]
    PixelTower(从零) --> cam bins(11) + keys(20)  ← 慢塔 Omni 每 SLOW_EVERY tick 刷新一次
        ↓ 温度采样
    4 条 rollout / 组(同 world seed)
        ↓ 联络表图 + 行为文本
    Haiku 判官从好到差排名 --> 名次取负 --> group_advantage(z 归一) --> REINFORCE
        loss = adv * ( CE(cam_logits, 采样bin) + BCE(key_logits, 采样key) )

里程碑(inv_events)**只作不可刷的汇报锚点**,不进训练信号。

与旧 grpo_r1.update 的两处关键修正(旧实现让慢塔静默失效):
    旧: g = torch.zeros(1,1)      新: goal_vec 真的接进 FiLM 条件
    旧: prev[1:,0] = 0.0          新: prev_action 逐步真实回填

用法:
    bash tests/serve_omni_nvfp4.sh                       # 慢塔(GPU_UTIL=0.85)
    Xvfb :99 -screen 0 1280x720x24 &
    DISPLAY=:99 /workspace/venv-mc/bin/python train/craftground/grpo_pixel.py \
        --groups 4 --rollout-ticks 400 --smoke
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.fovea_twotower.grpo_harness import group_advantage    # noqa: E402

OUT = Path("runs/grpo_pixel")          # 必须在工作区内:判官(claude CLI)要 Read 联络表图
SLOW_EVERY = 20                        # 慢塔刷新周期(tick);20 tick = 1s = 1Hz
IMG_HW = (90, 160)
MODEL = "nemotron_3_nano_omni"

# CraftGround V2 里的 20 个二值键(与 TrackNavConfig.n_keys=20 一致)
V2_KEYS = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use",
           "drop", "inventory", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4",
           "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9"]
CAM_BINS = 11
CAM_MAX_DEG = 18.0                     # 每 tick 相机增量上限(与 StudentPolicy 同口径)

SLOW_SYSTEM = """\
You are the slow-system planner for a Minecraft agent. You see one game frame.
Long-horizon goal: get wood, then a wooden pickaxe, then stone, then iron.

Answer with ONE line of JSON and nothing else:
{"subgoal": "<short imperative, <=6 words>", "aim": [X, Y]}

"aim" is the point the agent should put its crosshair on, in normalised image coordinates
0..1000, where (0,0) is top-left and (1000,1000) is bottom-right. Centre is (500,500).
It must land ON the block the agent should break or walk to next. Do not aim at the sky.
Do not copy any coordinates from these instructions; read them off the image.
"""

# 判官 rubric:只描述**任务目标**(哪条更接近拿到木头→镐→铁),不描述"该怎么操作"。
RUBRIC = """任务背景:智能体在 Minecraft 生存模式里的长程任务是独立获得铁。当前锚点:先拿到木头(原木)。
下面几条是同一世界、同一策略的并行尝试。每条证据 = 一张 8 帧时间均匀抽样的联络表图(先后从左到右、上到下) + 行为统计文本。
把它们按"向 拿到木头→木镐→石头→铁 推进的真实进度与意图质量"从好到差排名。参考阶梯:
瘫痪(不移动不按键) < 有动作但无方向(原地打转、乱跳、无目标游走) < 有目标性(持续朝树木接近、对树攻击、路线明确) < 拿到原木 < 木板 < 木镐/圆石/石镐 < 拿到铁
防刷分警告:文本里的统计量可以靠原地乱转刷高,不可单独作为进度证据;必须结合图中场景与移动/攻击行为判断。图文矛盾时以图为准。
最好=名次1。真分不出高下的允许并列(同名次),不要为拉开差距而编造。
先用 Read 工具逐张读取下面列出的联络表图,再逐条作答。
输出格式严格为每行一条『第N条: 名次X』(N 从 0 起,X 为数字),不输出其他内容。"""


# ────────────────────────────────────────────────────── 慢塔

class SlowTower:
    """Omni(NVFP4,本地 vLLM)。读一帧 → 文本子目标 + 目标像素。"""

    def __init__(self, base_url: str, encode_text, device: str):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.encode_text = encode_text
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}
        self.latencies: list[float] = []
        self.fails = 0

    def _b64(self, rgb: np.ndarray) -> str:
        b = io.BytesIO()
        Image.fromarray(rgb).save(b, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    def __call__(self, rgb: np.ndarray) -> tuple[torch.Tensor, str, list[float]]:
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SLOW_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "image_url", "image_url": {"url": self._b64(rgb)}},
                              {"type": "text", "text": "Next subgoal and aim point."}]}],
                max_tokens=48, temperature=0.2, top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
            )
            txt = (r.choices[0].message.content or "").strip()
            d = json.loads(re.search(r"\{.*\}", txt, re.S).group())
            subgoal = str(d["subgoal"])[:40]
            aim = [float(np.clip(v, 0, 1000)) for v in d["aim"][:2]]
        except Exception:  # noqa: BLE001  慢塔失灵 ⇒ 降级到零指导,不阻塞快环
            self.fails += 1
            subgoal, aim = "", [500.0, 500.0]
        self.latencies.append(time.perf_counter() - t0)

        if subgoal not in self.cache:
            v = self.encode_text([subgoal or "explore"])[0]
            self.cache[subgoal] = torch.as_tensor(v, dtype=torch.float32)
        goal = torch.cat([self.cache[subgoal],
                          torch.tensor([aim[0] / 1000.0, aim[1] / 1000.0])])
        return goal.to(self.device), subgoal, aim


# ────────────────────────────────────────────────────── 判官

def contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    sel = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).astype(int)]
    h, w = sel[0].shape[:2]
    sheet = np.zeros((2 * h, 4 * w, 3), np.uint8)
    for i, f in enumerate(sel):
        r, c = divmod(i, 4)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    Image.fromarray(sheet).save(path)


def evidence_text(r: dict) -> str:
    keys = r["keys"]
    ms = "、".join(f"{k}(第{v}步)" for k, v in sorted(r["inv_steps"].items(),
                                                    key=lambda x: x[1])) or "无"
    pose = r["pose"]
    disp = float(np.abs(np.diff(pose[:, [0, 2]], axis=0)).sum()) if len(pose) > 1 else 0.0
    fwd = float(keys[:, V2_KEYS.index("forward")].mean())
    atk = float(keys[:, V2_KEYS.index("attack")].mean())
    goals = "→".join(f"第{s}步『{g}』" for s, g in r["goal_log"][:6]) or "无"
    return (f"里程碑:{ms};总步数 {len(keys)};水平位移 {disp:.0f} 格;"
            f"前进键占比 {fwd:.2f};攻击键占比 {atk:.2f};"
            f"相机总转动 {float(np.abs(r['cam_deg']).sum()):.0f} 度;"
            f"慢塔子目标轨迹:{goals}")


def _parse_ranks(out: str, k: int) -> dict | None:
    got = {int(m.group(1)): float(m.group(2)) for m in
           re.finditer(r"第\s*(\d+)\s*条\s*[:：]\s*名次\s*([\d.]+)", out)}
    return got if len(got) == k and set(got) == set(range(k)) else None


def judge(g: int, rolls: list[dict]) -> tuple[np.ndarray, dict]:
    """Haiku 排序 → 名次取负当分数 → 组内 z 归一化当优势。"""
    lines = []
    for j, r in enumerate(rolls):
        img = (OUT / f"g{g}_r{j}.png").resolve()
        contact_sheet(r["frames"], img)
        lines.append(f"### 第{j}条\n联络表图:{img}\n{evidence_text(r)}")
    prompt = RUBRIC + "\n\n" + "\n".join(lines)
    (OUT / f"g{g}_judge_prompt.txt").write_text(prompt)

    ranks = None
    for _ in range(2):
        p = subprocess.run(["claude", "-p", "--model", "haiku", prompt],
                           capture_output=True, text=True, timeout=600)
        (OUT / f"g{g}_judge_reply.txt").write_text(p.stdout)
        ranks = _parse_ranks(p.stdout, len(rolls))
        if ranks:
            break

    if ranks:
        scores = [-ranks[j] for j in range(len(rolls))]
        meta = {"judge": "haiku", "ranks": ranks, "fallback": False}
    else:  # 判官两轮失败 ⇒ 回退不可刷的里程碑机器分(并记数,别静默)
        scores = [len(r["inv_events"]) for r in rolls]
        meta = {"judge": "fallback_milestone", "ranks": None, "fallback": True}
    return group_advantage(scores), meta


# ────────────────────────────────────────────────────── rollout & update

def bins_to_deg(b: np.ndarray) -> np.ndarray:
    """mu-law 分箱 → 度。bin 中心 [-1,1] 经 mu-law 解压后乘 CAM_MAX_DEG。"""
    x = (b.astype(np.float32) / (CAM_BINS - 1)) * 2 - 1          # [-1,1]
    mu = 8.0
    v = np.sign(x) * (np.power(1 + mu, np.abs(x)) - 1) / mu
    return v * CAM_MAX_DEG


def rollout(env, tower, slow, no_op, rng, ticks: int, device: str, temp: float) -> dict:
    from craftground.environment.action_space import no_op_v2  # noqa: F401
    obs, _ = env.reset()
    for _ in range(60):                                   # 等 "Loading terrain..."
        obs = env.step(no_op())[0]

    cfg = tower.cfg
    goal = torch.zeros(cfg.goal_dim, device=device)
    subgoal, aim = "", [500.0, 500.0]
    imgs, prevs, cam_b, key_b, frames, pose, cam_deg = [], [], [], [], [], [], []
    goal_log, inv_events, inv_steps = [], set(), {}
    prev = np.zeros(cfg.n_mouse + cfg.n_keys, np.float32)

    for t in range(ticks):
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        if t % SLOW_EVERY == 0:                            # 慢塔按自身节拍刷新
            goal, subgoal, aim = slow(rgb)
            goal_log.append((t, subgoal))
        if t % max(1, ticks // 24) == 0:
            frames.append(np.asarray(Image.fromarray(rgb).resize((160, 90))))

        small = np.asarray(Image.fromarray(rgb).resize((IMG_HW[1], IMG_HW[0])),
                           dtype=np.float32) / 255.0
        img = torch.from_numpy(small).permute(2, 0, 1)[None, None].to(device)
        pv = torch.from_numpy(prev)[None, None].to(device)
        with torch.no_grad():
            cam_l, key_l = tower(img, goal[None], pv)
        cam_l = cam_l[0, -1, 0] / temp                     # [n_mouse, bins]
        key_l = key_l[0, -1, 0] / temp                     # [n_keys]
        cb = torch.distributions.Categorical(logits=cam_l).sample().cpu().numpy()
        kp = torch.bernoulli(torch.sigmoid(key_l)).cpu().numpy().astype(np.int8)

        deg = bins_to_deg(cb)
        a = no_op()
        a["camera_yaw"], a["camera_pitch"] = float(deg[0]), float(deg[1])
        for i, k in enumerate(V2_KEYS):
            if kp[i]:
                a[k] = True
        obs = env.step(a)[0]
        full = obs["full"]

        for it in full.inventory:
            name = it.translation_key.split(".")[-1]
            if it.count > 0 and name not in inv_events:
                inv_events.add(name)
                inv_steps[name] = t

        imgs.append(small); prevs.append(prev.copy())
        cam_b.append(cb); key_b.append(kp); cam_deg.append(deg)
        pose.append([full.x, full.y, full.z])
        prev = np.concatenate([deg / CAM_MAX_DEG, kp.astype(np.float32)])

    return dict(imgs=np.stack(imgs), prevs=np.stack(prevs), cam=np.stack(cam_b),
                keys=np.stack(key_b), cam_deg=np.stack(cam_deg),
                pose=np.asarray(pose, np.float32), frames=frames,
                goal_log=goal_log, inv_events=inv_events, inv_steps=inv_steps,
                goal_last=goal.detach().cpu())


def update(tower, opt, rolls, adv, seq: int, device: str) -> float:
    """REINFORCE:loss = adv * ( CE(cam, 采样bin) + BCE(key, 采样key) )。

    CE 打在**采样到的** bin 上 ⇒ 等价 -log π(a);故 adv*CE 的梯度即
    ∇ adv·(-log π(a)),最小化它 = 最大化 adv·log π(a)。
    """
    tot = n = 0.0
    for r, a_w in zip(rolls, adv):
        if abs(float(a_w)) < 1e-6:
            continue
        T = len(r["cam"])
        goal = r["goal_last"].to(device)[None]
        for i0 in range(0, max(T - seq, 1), seq):
            sl = slice(i0, i0 + seq)
            img = torch.from_numpy(r["imgs"][sl]).permute(0, 3, 1, 2)[None].to(device)
            pv = torch.from_numpy(r["prevs"][sl])[None].to(device)
            cam_l, key_l = tower(img, goal, pv)            # goal/prev 真的进梯度
            cb = torch.from_numpy(r["cam"][sl]).long().to(device)
            kp = torch.from_numpy(r["keys"][sl].astype(np.float32)).to(device)
            ce = F.cross_entropy(cam_l[0, :, 0].reshape(-1, CAM_BINS), cb.reshape(-1))
            bce = F.binary_cross_entropy_with_logits(key_l[0, :, 0], kp)
            loss = float(a_w) * (ce + bce)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tower.parameters(), 1.0)
            opt.step()
            tot += float(loss); n += 1
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--rollout-ticks", type=int, default=400)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--smoke", action="store_true", help="短 rollout,只验链路")
    args = ap.parse_args()

    if "DISPLAY" not in os.environ:
        sys.exit("need DISPLAY (Xvfb :99)")
    if args.smoke:
        args.rollout_ticks, args.groups, args.seq = 120, 1, 32

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.initial_environment_config import Difficulty, GameMode, WorldType
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    encode_text = lambda xs: st.encode(xs, normalize_embeddings=True)  # noqa: E731

    cfg = PixelTowerConfig(img_hw=IMG_HW, goal_dim=384 + 2, n_keys=len(V2_KEYS),
                           camera_bins=CAM_BINS)
    tower = build_pixel_tower(cfg).to(device)
    opt = torch.optim.AdamW(tower.parameters(), lr=args.lr)
    print(f"PixelTower params = {sum(p.numel() for p in tower.parameters()) / 1e6:.2f} M",
          flush=True)

    slow = SlowTower(args.base_url, encode_text, device)
    rng = np.random.default_rng(0)

    for g in range(args.groups):
        wseed = str(int(rng.integers(0, 1 << 30)))
        env_cfg = InitialEnvironmentConfig(
            image_width=640, image_height=360,
            gamemode=GameMode.SURVIVAL, difficulty=Difficulty.PEACEFUL,
            world_type=WorldType.DEFAULT, seed=wseed,
            screen_encoding_mode=ScreenEncodingMode.RAW)
        env_cfg.set_allow_mob_spawn(False); env_cfg.freeze_time(True)
        env_cfg.freeze_weather(True)
        env = CraftGroundEnvironment(env_cfg,
                                     action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                                     port=args.port + g, find_free_port=True, verbose=False)
        t0 = time.time()
        rolls = [rollout(env, tower, slow, no_op_v2, rng, args.rollout_ticks, device, args.temp)
                 for _ in range(args.per_group)]
        env.close()

        adv, jmeta = judge(g, rolls)
        loss = update(tower, opt, rolls, adv, args.seq, device)
        torch.save(dict(tower=tower.state_dict(), cfg=vars(cfg), group=g),
                   OUT / "tower.pt")

        m = dict(group=g, world_seed=wseed, n=len(rolls),
                 adv=[round(float(a), 3) for a in adv], adv_var=round(float(np.var(adv)), 4),
                 **jmeta,
                 milestones={k: sum(1 for r in rolls if k in r["inv_events"])
                             for k in ["oak_log", "birch_log", "dark_oak_log", "spruce_log",
                                       "oak_planks", "wooden_pickaxe", "cobblestone",
                                       "stone_pickaxe", "raw_iron"]},
                 slow_fail=slow.fails,
                 slow_lat_p50=round(float(np.percentile(slow.latencies, 50)), 3),
                 loss=round(loss, 4), wall_s=round(time.time() - t0, 0))
        with (OUT / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
        print(f"[g{g}] {json.dumps(m, ensure_ascii=False, default=str)}", flush=True)


if __name__ == "__main__":
    main()
