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
    CraftGround 原始像素 640x360 --下采样--> [3,90,160] --帧堆叠 S=4--> [12,90,160]
        ↓                                    ↑ goal_vec[386] = MiniLM(subgoal)[384] ⊕ aim/1000[2]
    PixelTower(从零) --> cam bins(11) + keys(20)  ← 慢塔 Omni 每 SLOW_EVERY tick 刷新一次
        ↓ 温度采样(T=1,eval 模式)
    4 条 rollout / 组(同 world seed)
        ↓ 联络表图 + 行为文本
    Haiku 判官从好到差排名 --> 名次取负 --> group_advantage(z 归一) --> REINFORCE
        loss = adv * ( CE(cam_logits/temp, 采样bin) + BCE(key_logits/temp, 采样key) )
        一个 group 梯度累积后单次 opt.step(严格 on-policy)

里程碑(inv_events)**只作不可刷的汇报锚点**,不进训练信号。

2026-07-10 五个 log π 修复(next_session §3,详见 update() docstring):
    ① 双侧 T=1+帧堆叠  ② eval 采样/train 更新(dropout=0)  ③ 温度双侧一致
    ④ goal 逐 tick 落盘并回放(含 aim)  ⑤ 组内梯度累积单次 opt.step
数学验收单测:tests/unit/test_grpo_pixel_fixes.py(CUDA 冒烟)。

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

from net.calibration import (SelfCalib, control_mode, fit_latency,  # noqa: E402
                             flow_shift, probe_plan_multi)
from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.craftground.action_contract import (CAM_BINS, CAM_MAX_DEG,  # noqa: E402
                                               V2_KEYS, bins_to_deg, stack_frames)
from train.fovea_twotower.grpo_harness import group_advantage    # noqa: E402

OUT = Path("runs/grpo_pixel")          # 必须在工作区内:判官(claude CLI)要 Read 联络表图
SLOW_EVERY = 20                        # 慢塔刷新周期(tick);20 tick = 1s = 1Hz
IMG_HW = (90, 160)
MODEL = "nemotron_3_nano_omni"

# 设计 2 会话契约(2026-07-10,设计文档 §9/§11):无状态重提示 + 状态全外置。
# 每次调用 = 固定 system(prefix cache 可命中)+ TASK/STATE/PHYSICS 行 + 当前帧。
# system 是**游戏无关**的:任务目标走 TASK 行、物理参数走 PHYSICS 行(自标定测出),
# 领域知识不焊死在提示词里——换游戏只换输入行,不换契约。
SLOW_SYSTEM = """\
You are the slow-system planner of a real-time game agent. Each call you get:
a TASK line (the long-horizon objective), a STATE line (time, inventory/resources,
displacement, your recent subgoals and whether they got done), an optional PHYSICS line
(camera gain / field of view / move speed, measured by the agent itself), and the
current game frame.

Answer with ONE line of JSON and nothing else:
{"prev_done": true|false, "decision": "continue|switch|replan",
 "subgoal": "<short imperative, <=6 words>", "aim": [X, Y],
 "done_when": "<machine-checkable condition, <=8 words>"}

- "prev_done": whether YOUR previous subgoal (shown in STATE) is now achieved.
- "decision": continue = keep previous subgoal; switch = new target; replan = stuck, change approach.
- "aim" is the pixel the low-level controller should put its crosshair/cursor on,
  normalised 0..1000, (0,0) top-left, (1000,1000) bottom-right, centre (500,500).
  It must land ON the object to interact with or walk to next. Do not aim at the sky.
  Do not copy any coordinates from these instructions; read them off the image.
"""

DEFAULT_TASK = ("Minecraft survival: obtain wood, then a wooden pickaxe, "
                "then stone, then iron.")

DECISIONS = ("continue", "switch", "replan")   # 微决策词表(留出决策 acc 0.937 的同族)


def parse_slow_reply(txt: str) -> dict:
    """慢塔单行 JSON 的容错解析(无 LLM 可单测)。失灵字段逐个降级,不整体作废。"""
    d = {}
    try:
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group())
    except Exception:  # noqa: BLE001
        pass
    subgoal = str(d.get("subgoal", ""))[:40]
    try:
        aim = [float(np.clip(float(v), 0, 1000)) for v in list(d.get("aim", []))[:2]]
        assert len(aim) == 2
    except Exception:  # noqa: BLE001
        aim = [500.0, 500.0]
    dec = str(d.get("decision", "switch")).strip().lower()
    return dict(subgoal=subgoal, aim=aim,
                prev_done=bool(d.get("prev_done", False)),
                decision=dec if dec in DECISIONS else "switch",
                done_when=str(d.get("done_when", ""))[:60],
                parsed=bool(d))


def state_line(t: int, inv_steps: dict, pose_hist: list, goal_hist: list) -> str:
    """外置状态行:模型的记忆活在这里,不占潜向量一个比特(设计 2)。

    goal_hist 元素 = goal_log 条目 (t, subgoal, aim, done_when, decision, prev_done)。
    """
    inv = "、".join(f"{k}@{v}" for k, v in sorted(inv_steps.items(),
                                                 key=lambda x: x[1])[-6:]) or "empty"
    disp = 0.0
    if len(pose_hist) > 1:
        p = np.asarray(pose_hist, np.float32)
        disp = float(np.abs(np.diff(p[:, [0, 2]], axis=0)).sum())
    lines = []
    for e in goal_hist[-3:]:
        mark = "done" if len(e) > 5 and e[5] else "open"
        lines.append(f"t{e[0]}'{e[1]}'({mark})")
    hist = " -> ".join(lines) or "none"
    return (f"STATE t={t} inventory:{inv} displacement:{disp:.0f}blocks "
            f"recent_subgoals:{hist}")

# 判官 rubric v2(2026-07-10):只给**任务**,不给手写进度阶梯——阶梯是人写的价值函数,
# 会经排序渗进训练信号(苦涩的教训);推进程度由判官从证据自行判断。游戏无关,任务走模板。
RUBRIC_TMPL = """任务:{task}
下面几条是同一世界、同一策略的并行尝试。每条证据 = 一张 8 帧时间均匀抽样的联络表图(先后从左到右、上到下) + 行为统计文本。
把它们按"对完成上述任务的真实推进与意图质量"从好到差排名。不提供进度阶梯:请从图中场景变化与行为证据自行判断谁推进得更远、意图更明确。
防刷分警告:文本统计量可以靠原地乱转刷高,不可单独作为进度证据;必须结合图中场景与行为判断。图文矛盾时以图为准。
最好=名次1。真分不出高下的允许并列(同名次),不要为拉开差距而编造。
先用 Read 工具逐张读取下面列出的联络表图,再逐条作答。
输出格式严格为每行一条『第N条: 名次X』(N 从 0 起,X 为数字),不输出其他内容。"""


# ────────────────────────────────────────────────────── 慢塔

class SlowTower:
    """Omni(NVFP4,本地 vLLM)。读一帧 → 文本子目标 + 目标像素。"""

    def __init__(self, base_url: str, encode_text, device: str, task: str = DEFAULT_TASK):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.encode_text = encode_text
        self.device = device
        self.task = task                                  # 领域知识只活在这一行
        self.cache: dict[str, torch.Tensor] = {}
        self.latencies: list[float] = []
        self.fails = 0

    def _b64(self, rgb: np.ndarray) -> str:
        b = io.BytesIO()
        Image.fromarray(rgb).save(b, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    def __call__(self, rgb: np.ndarray, state: str = "") -> tuple[torch.Tensor, dict]:
        """设计 2:每次全新短会话,状态经 state 行外置注入。返回 (goal_vec, 解析字段 dict)。"""
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SLOW_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "text", "text": f"TASK: {self.task}\n"
                               + (state or "STATE t=0 (fresh start)")},
                              {"type": "image_url", "image_url": {"url": self._b64(rgb)}},
                              {"type": "text", "text": "Next subgoal."}]}],
                max_tokens=96, temperature=0.2, top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
            )
            rep = parse_slow_reply((r.choices[0].message.content or "").strip())
            if not rep["parsed"]:
                self.fails += 1
        except Exception:  # noqa: BLE001  慢塔失灵 ⇒ 降级到零指导,不阻塞快环
            self.fails += 1
            rep = parse_slow_reply("")
        self.latencies.append(time.perf_counter() - t0)

        subgoal, aim = rep["subgoal"], rep["aim"]
        if subgoal not in self.cache:
            v = self.encode_text([subgoal or "explore"])[0]
            self.cache[subgoal] = torch.as_tensor(v, dtype=torch.float32)
        goal = torch.cat([self.cache[subgoal],
                          torch.tensor([aim[0] / 1000.0, aim[1] / 1000.0])])
        return goal.to(self.device), rep


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
    goals = "→".join(f"第{e[0]}步『{e[1]}』" + ("✓" if e[5] else "")
                     for e in r["goal_log"][:6]) or "无"
    return (f"里程碑:{ms};总步数 {len(keys)};水平位移 {disp:.0f} 格;"
            f"前进键占比 {fwd:.2f};攻击键占比 {atk:.2f};"
            f"相机总转动 {float(np.abs(r['cam_deg']).sum()):.0f} 度;"
            f"慢塔子目标轨迹:{goals}")


def _parse_ranks(out: str, k: int) -> dict | None:
    got = {int(m.group(1)): float(m.group(2)) for m in
           re.finditer(r"第\s*(\d+)\s*条\s*[:：]\s*名次\s*([\d.]+)", out)}
    return got if len(got) == k and set(got) == set(range(k)) else None


def judge(g: int, rolls: list[dict], task: str = DEFAULT_TASK) -> tuple[np.ndarray, dict]:
    """Haiku 排序 → 名次取负当分数 → 组内 z 归一化当优势。

    全量落盘的 (证据, 排序) 对同时是未来本地 RM 的训练数据(设计文档 §11.4:
    名次可离线展开成成对偏好,轨迹比较从"每组 1 次 API 排序"演进到"无限次本地打分")。
    """
    lines = []
    for j, r in enumerate(rolls):
        img = (OUT / f"g{g}_r{j}.png").resolve()
        contact_sheet(r["frames"], img)
        lines.append(f"### 第{j}条\n联络表图:{img}\n{evidence_text(r)}")
    prompt = RUBRIC_TMPL.format(task=task) + "\n\n" + "\n".join(lines)
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

def calibrate(env, no_op, obs) -> tuple[SelfCalib, object]:
    """开局自标定(加固版):先压低视线(看地面纹理,天空低纹理测不出流),
    多幅度对称探针 → 延迟扫描 → 按延迟错位配对测增益/曲线 → 模式脉冲 →
    步速(pose,特权只进训练侧)。约 40 tick。产出 SelfCalib 随 episode 携带。
    """
    calib = SelfCalib(img_w=IMG_HW[1], img_h=IMG_HW[0])

    def small(o):
        return np.asarray(Image.fromarray(np.asarray(o["rgb"], np.uint8))
                          .resize((IMG_HW[1], IMG_HW[0])), np.float32)

    def cam_step(yaw, pitch):
        a = no_op()
        a["camera_yaw"], a["camera_pitch"] = float(yaw), float(pitch)
        return env.step(a)[0]

    obs = cam_step(0, 20)                                 # 压低视线:地面纹理
    plan = probe_plan_multi((2.0, 4.0, 8.0))
    frames, cmds = [small(obs)], []
    for yaw, pitch in plan:
        obs = cam_step(yaw, pitch)
        frames.append(small(obs))
        cmds.append((yaw, pitch))
    for _ in range(3):                                    # 尾部 no-op:延迟>0 时的响应帧
        obs = env.step(no_op())[0]
        frames.append(small(obs))
        cmds.append((0.0, 0.0))
    flows = np.array([flow_shift(frames[i], frames[i + 1])[:2]
                      for i in range(len(cmds))], np.float32)
    lag, lag_corr = fit_latency(flows, np.array(cmds, np.float32))
    if lag_corr >= 0.3:                                   # 测得出延迟才采信
        calib.latency_ticks = lag
    k = calib.latency_ticks or 0
    for i in range(len(plan)):                            # 按延迟错位配对喂增益/曲线证据
        calib.update_camera(frames[i + k], frames[i + k + 1], *plan[i])
    calib.fit_curve()

    pulse = 6.0                                           # 模式脉冲:1 tick 命令 + 3 tick 静默
    f_a = small(obs)
    obs = cam_step(pulse, 0)
    f_b = small(obs)
    mag_during = abs(flow_shift(f_a, f_b)[0])             # 脉冲期的流
    afters = []
    for _ in range(3):
        obs = env.step(no_op())[0]
        f_c = small(obs)
        afters.append(abs(flow_shift(f_b, f_c)[0]))       # 静默期的流:骤停=位置增量
        f_b = f_c
    calib.mode = control_mode(mag_during, float(np.mean(afters)))
    obs = cam_step(-pulse, 0)                             # 回正净漂移

    p0 = obs["full"]
    for _ in range(6):                                    # 前进 6 tick 测步速(训练侧)
        a = no_op()
        a["forward"] = True
        obs = env.step(a)[0]
    p1 = obs["full"]
    calib.update_locomotion(float(np.hypot(p1.x - p0.x, p1.z - p0.z)), 6)
    return calib, obs


def rollout(env, tower, slow, no_op, rng, ticks: int, device: str, temp: float,
            do_calib: bool = True) -> dict:
    from craftground.environment.action_space import no_op_v2  # noqa: F401
    obs, _ = env.reset()
    for _ in range(60):                                   # 等 "Loading terrain..."
        obs = env.step(no_op())[0]
    calib = SelfCalib(img_w=IMG_HW[1], img_h=IMG_HW[0])
    if do_calib:                                          # 自标定:物理参数测出来,不写死
        calib, obs = calibrate(env, no_op, obs)

    cfg = tower.cfg
    tower.eval()                                          # 修复②:采样在确定性网络上
    goal = torch.zeros(cfg.goal_dim, device=device)
    imgs, prevs, goals, cam_b, key_b, frames, pose, cam_deg = [], [], [], [], [], [], [], []
    goal_log, inv_events, inv_steps = [], set(), {}
    prev = np.zeros(cfg.n_mouse + cfg.n_keys, np.float32)
    fstack: list[np.ndarray] = []                         # 最近 frame_stack 帧,旧→新

    for t in range(ticks):
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        if t % SLOW_EVERY == 0:                            # 慢塔按自身节拍刷新
            st = state_line(t, inv_steps, pose, goal_log) + "\n" + calib.physics_line()
            goal, rep = slow(rgb, st)
            if rep["prev_done"] and goal_log:              # prev_done 指上一条:回填标记
                goal_log[-1][5] = True
            goal_log.append([t, rep["subgoal"], rep["aim"], rep["done_when"],
                             rep["decision"], False])      # 修复④配套:aim/done_when 落盘
        if t % max(1, ticks // 24) == 0:
            frames.append(np.asarray(Image.fromarray(rgb).resize((160, 90))))

        small = np.asarray(Image.fromarray(rgb).resize((IMG_HW[1], IMG_HW[0])),
                           dtype=np.float32) / 255.0
        fstack.append(small)
        if len(fstack) < cfg.frame_stack:                  # 开局首帧填充
            fstack = [small] * (cfg.frame_stack - len(fstack)) + fstack
        fstack = fstack[-cfg.frame_stack:]
        stacked = np.concatenate(fstack, axis=2)           # [H,W,3S] 旧→新
        img = torch.from_numpy(stacked).permute(2, 0, 1)[None, None].to(device)
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
        goals.append(goal.detach().cpu().numpy())          # 修复④:goal 逐 tick 落盘
        cam_b.append(cb); key_b.append(kp); cam_deg.append(deg)
        pose.append([full.x, full.y, full.z])
        prev = np.concatenate([deg / CAM_MAX_DEG, kp.astype(np.float32)])

    return dict(imgs=np.stack(imgs), prevs=np.stack(prevs), goals=np.stack(goals),
                cam=np.stack(cam_b), keys=np.stack(key_b), cam_deg=np.stack(cam_deg),
                pose=np.asarray(pose, np.float32), frames=frames,
                goal_log=goal_log, inv_events=inv_events, inv_steps=inv_steps,
                calib=calib)


def update(tower, opt, rolls, adv, chunk: int, temp: float, device: str) -> float:
    """REINFORCE(on-policy):loss = adv * ( CE(cam, 采样bin) + BCE(key, 采样key) )。

    CE 打在**采样到的** bin 上 ⇒ 等价 -log π(a);最小化 adv·CE = 最大化 adv·log π(a)。

    2026-07-10 修复(编号对应 next_session §3):
      ① 采样/更新同分布:双侧都是 T=1 + frame_stack(帧堆叠由 stack_frames 与采样端
         deque 逐字节同序),失配从结构上消灭;tick 沿 batch 维成批,尾部 tick 不再丢弃。
      ② 采样 tower.eval() / 更新 tower.train()(且 cfg.dropout=0,两模式恒等)。
      ③ 温度一致:损失打在与采样相同的 logits/temp 上,π 是同一个分布。
      ④ goal 逐 tick 回放(rollout 已逐 tick 落盘 386 维向量,不再抹成 goal_last)。
      ⑤ 一个 group 全部梯度累积后**单次 opt.step()**:采样分布 = 被更新分布,
         这是严格 on-policy 的 REINFORCE;不做多步复用,故无需 ratio/clip/KL。
         (若未来改多步复用,必须补 importance ratio,见 arch_current §5.4。)
    chunk 只是显存分块,数学上无意义(梯度按全组 tick 数归一后累加)。
    """
    tower.train()
    opt.zero_grad()
    active = [(r, float(a_w)) for r, a_w in zip(rolls, adv) if abs(float(a_w)) >= 1e-6]
    denom = float(sum(len(r["cam"]) for r, _ in active))
    if denom == 0:
        return 0.0
    s = tower.cfg.frame_stack
    tot = 0.0
    for r, a_w in active:
        t_n = len(r["cam"])
        stacked = stack_frames(r["imgs"], s)               # [T,3S,H,W] 与采样端同序
        for i0 in range(0, t_n, chunk):                    # 覆盖全部 tick,含尾段
            sl = slice(i0, min(i0 + chunk, t_n))
            img = torch.from_numpy(stacked[sl]).unsqueeze(1).to(device)   # [B,1,3S,H,W]
            pv = torch.from_numpy(r["prevs"][sl]).unsqueeze(1).to(device)  # [B,1,·]
            goal = torch.from_numpy(r["goals"][sl]).to(device)             # [B,386] 逐tick
            cam_l, key_l = tower(img, goal, pv)            # goal/prev 真的进梯度
            cam_l = cam_l[:, 0, 0] / temp                  # [B,n_mouse,bins] 与采样同温度
            key_l = key_l[:, 0, 0] / temp                  # [B,n_keys]
            cb = torch.from_numpy(r["cam"][sl]).long().to(device)
            kp = torch.from_numpy(r["keys"][sl].astype(np.float32)).to(device)
            ce = F.cross_entropy(cam_l.reshape(-1, CAM_BINS), cb.reshape(-1),
                                 reduction="sum") / cam_l.shape[1]
            bce = F.binary_cross_entropy_with_logits(key_l, kp, reduction="sum") \
                / key_l.shape[1]
            loss = a_w * (ce + bce) / denom                # 全组 tick 均值口径
            loss.backward()                                # 只累积,不 step
            tot += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(tower.parameters(), 1.0)
    opt.step()                                             # 修复⑤:整组唯一一次 step
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--rollout-ticks", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=128,
                    help="更新时的显存分块(tick 数);纯工程参数,不改数学")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--task", default=DEFAULT_TASK,
                    help="长程任务目标(领域知识只活在此行,换游戏换这里)")
    ap.add_argument("--no-calib", action="store_true", help="跳过开局自标定探针")
    ap.add_argument("--init-from", default="",
                    help="BC 暖启动 checkpoint(bc_vpt_warmstart 产出;GRPO 只做精修,"
                         "受控对照见 conclusion_fasttower_skill_ceiling)")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--smoke", action="store_true", help="短 rollout,只验链路")
    args = ap.parse_args()

    if "DISPLAY" not in os.environ:
        sys.exit("need DISPLAY (Xvfb :99)")
    if args.smoke:
        args.rollout_ticks, args.groups, args.chunk = 120, 1, 32

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
    if args.init_from:                     # BC 暖启动(结构 cfg 必须一致,load 严格校验)
        ck = torch.load(args.init_from, map_location=device, weights_only=True)
        tower.load_state_dict(ck["tower"])
        print(f"init from {args.init_from} (bc_step={ck.get('step')})", flush=True)
    opt = torch.optim.AdamW(tower.parameters(), lr=args.lr)
    print(f"PixelTower params = {sum(p.numel() for p in tower.parameters()) / 1e6:.2f} M",
          flush=True)

    slow = SlowTower(args.base_url, encode_text, device, task=args.task)
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
        rolls = [rollout(env, tower, slow, no_op_v2, rng, args.rollout_ticks, device,
                         args.temp, do_calib=not args.no_calib)
                 for _ in range(args.per_group)]
        env.close()

        adv, jmeta = judge(g, rolls, task=args.task)
        loss = update(tower, opt, rolls, adv, args.chunk, args.temp, device)
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
                 calib=rolls[0]["calib"].physics_line(),
                 loss=round(loss, 4), wall_s=round(time.time() - t0, 0))
        with (OUT / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
        print(f"[g{g}] {json.dumps(m, ensure_ascii=False, default=str)}", flush=True)


if __name__ == "__main__":
    main()
