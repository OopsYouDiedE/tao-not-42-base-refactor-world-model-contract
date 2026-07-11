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
    4 条 rollout / 组(同 world seed;seed 经出生点有树先验筛选;死亡即截断)
        ↓ 联络表图 + 行为文本
    判官成对比较(pairwise_v2:C(4,2) 对×正反两问,默认 tie + 举证 + 一致性门)
        --> Copeland 计分 --> group_advantage(z 归一;全并列 ⇒ 全零跳过) --> REINFORCE
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

快塔双实现(--tower,默认 v1,checkpoint 分文件互不污染):
    v1 = PixelTower(像素 conv stem + FiLM;结构与 bc_vpt checkpoint 契约冻结);
    v2 = TokenPolicyTower(DINO patch+地图+语言 token,goal-as-query;装配在
         train/craftground/tower_v2.py,符号标定/降级口径见该文件头)。

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
from train.craftground.tower_v2 import (DinoFrontend, V2Config, V2Policy,  # noqa: E402
                                        V2Runtime, v2_replay)
from train.fovea_twotower.grpo_harness import group_advantage    # noqa: E402

OUT = Path("runs/grpo_pixel")          # 必须在工作区内:判官(claude CLI)要 Read 联络表图
SLOW_EVERY = 20                        # 慢塔刷新周期(tick);20 tick = 1s = 1Hz
IMG_HW = (90, 160)
MODEL = "nemotron_3_nano_omni"         # 默认慢塔;--slow-model 可换(慢塔是可替换件,设计文档 §10)

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
 "subgoal": "<verb> <item>", "aim": [X, Y],
 "done_when": "<machine-checkable condition, <=8 words>"}

- "subgoal" MUST be exactly two tokens: a verb from {mine, collect, craft, open,
  close, use, drop, kill} + one specific block/item name in snake_case,
  e.g. "mine oak_log", "collect dirt", "craft crafting_table", "open inventory".
  This is the controller's trained goal language; free-form phrases are ignored by it.

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

# 判官协议 pairwise_v2(2026-07-10 用户定罪裁决,替换全排序 rank_v1)。
# rank_v1 定罪证据:同内容 4 退化渲染,判官 3/3 轮给严格全序(幻觉率 100%),按画质
# 而非推进排序;8×4×2000 run 中 g1_r2(溺亡后卡死亡画面)被排名次 1。
# 三处修法:①成对比较替代全排序(C(k,2) 对,正反各问一次);②举证责任倒置
# (默认 tie,判胜负必须引用具体帧号/统计量,引用空洞按 tie 计);③一致性门
# (正反不一致 ⇒ tie;胜负图有环 ⇒ 全组并列)。Copeland 计分合成名次;
# 全并列 ⇒ adv 全零 ⇒ update 的 |adv|<1e-6 跳过。落盘记录带 protocol 字段,
# 将来 RM 训练按协议过滤(rank_v1 数据有画质偏好污染)。
PAIR_RUBRIC_TMPL = """任务:{task}
下面 A、B 两条是同一世界、同一策略的并行尝试。每条证据 = 一张 8 帧时间均匀抽样的联络表图(先后从左到右、上到下) + 行为统计文本。
问题:对完成上述任务的真实推进与意图质量,A 和 B 谁更好?
**默认结论是 tie(不分胜负)。**只有当你能引用具体证据——帧号(如「A 第 5 帧出现树干特写」)或统计量差异(如「位移 100 vs 24 格」)——支持一方确实推进更远时,才允许判胜负。
防刷分警告:文本统计量可以靠原地乱转刷高,不可单独作为进度证据;必须结合图中场景与行为判断。图文矛盾时以图为准。
画质、清晰度、色彩观感与任务推进无关,不得作为证据。
先用 Read 工具读取两张联络表图,再作答。
输出严格为单行 JSON,不输出其他内容:
{{"winner": "A"|"B"|"tie", "evidence": "<引用的具体帧号或统计量;tie 时可留空>"}}"""


# ────────────────────────────────────────────────────── 慢塔

class SlowTower:
    """Omni(NVFP4,本地 vLLM)。读一帧 → 文本子目标 + 目标像素。

    vocab_snap:hindsight 词表 json 路径(空串关闭)。开启时把慢塔自由措辞的 subgoal
    经 MiniLM 余弦最近邻投影回词表短语——goal 三臂对照(2026-07-11)实证:同一
    checkpoint,词表内 goal 的 attack 占空比 ~0.55 vs 自由措辞 ~0.21 vs 零 goal ~0.13,
    学生首块木头即来自词表内 goal;分布外文本只吃到 FiLM 通道一小半带宽。
    """

    def __init__(self, base_url: str, encode_text, device: str, task: str = DEFAULT_TASK,
                 model: str = MODEL, vocab_snap: str = ""):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.encode_text = encode_text
        self.device = device
        self.task = task                                  # 领域知识只活在这一行
        self.cache: dict[str, torch.Tensor] = {}
        self.latencies: list[float] = []
        self.fails = 0
        self.vocab_names: list[str] = []
        self.vocab_mat: torch.Tensor | None = None        # [N,384] L2 归一
        if vocab_snap:
            vb = json.loads(Path(vocab_snap).read_text())
            self.vocab_names = list(vb.keys())
            m = torch.tensor(list(vb.values()), dtype=torch.float32)
            self.vocab_mat = torch.nn.functional.normalize(m, dim=-1)

    SNAP_MIN_COS = 0.6   # 低于此相似度不投影(裸投影实测把 "chop down the tree" 投成
                         # "drop wooden axe"@0.49——错误注入比不投更糟;配套修法是
                         # SLOW_SYSTEM 要求 "<verb> <item>" 词表语言,投影只做规范化)

    def _snap(self, subgoal: str) -> str:
        """慢塔措辞 → 词表最近邻短语(嵌入同源 MiniLM,余弦;低置信保留原话)。"""
        if self.vocab_mat is None or not subgoal:
            return subgoal
        v = torch.as_tensor(self.encode_text([subgoal])[0], dtype=torch.float32)
        sims = self.vocab_mat @ v
        i = int(sims.argmax())
        return self.vocab_names[i] if float(sims[i]) >= self.SNAP_MIN_COS else subgoal

    def _b64(self, rgb: np.ndarray) -> str:
        b = io.BytesIO()
        Image.fromarray(rgb).save(b, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    def __call__(self, rgb: np.ndarray, state: str = "") -> tuple[torch.Tensor, dict]:
        """设计 2:每次全新短会话,状态经 state 行外置注入。返回 (goal_vec, 解析字段 dict)。"""
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(
                model=self.model,
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
        if self.vocab_mat is not None and subgoal:
            rep["subgoal_raw"] = subgoal               # 原话落盘,投影可審计
            subgoal = rep["subgoal"] = self._snap(subgoal)
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
    death = (f"第 {r['death_step']} 步死亡(episode 截断);"
             if r.get("death_step") is not None else "")
    return (f"{death}里程碑:{ms};总步数 {len(keys)};水平位移 {disp:.0f} 格;"
            f"前进键占比 {fwd:.2f};攻击键占比 {atk:.2f};"
            f"相机总转动 {float(np.abs(r['cam_deg']).sum()):.0f} 度;"
            f"慢塔子目标轨迹:{goals}")


def _parse_pair(txt: str) -> dict | None:
    """成对判官单行 JSON 解析。举证责任倒置的机械检查:判胜负却引用空洞
    (evidence 缺失或不含任何数字——帧号/统计量必然带数字)⇒ 降级为 tie。"""
    try:
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group())
    except Exception:  # noqa: BLE001
        return None
    w = str(d.get("winner", "")).strip().lower()
    ev = str(d.get("evidence", ""))
    if w not in ("a", "b", "tie"):
        return None
    if w in ("a", "b") and not re.search(r"\d", ev):
        w = "tie"
    return {"winner": w, "evidence": ev}


def judge_pair_call(img_a: str, ev_a: str, img_b: str, ev_b: str, task: str,
                    model: str, effort: str | None = None) -> tuple[str, str, dict | None]:
    """单次成对判官调用(重试 1 次;异常/解析失败返回 parsed=None,上层按 tie 计)。"""
    prompt = (PAIR_RUBRIC_TMPL.format(task=task)
              + f"\n\n### A\n联络表图:{img_a}\n{ev_a}\n### B\n联络表图:{img_b}\n{ev_b}")
    cmd = ["claude", "-p", "--model", model]
    if effort:                             # Sonnet 对照臂用 --effort low;Haiku 主臂不带
        cmd += ["--effort", effort]
    out, parsed = "", None
    for _ in range(2):
        try:
            p = subprocess.run(cmd + [prompt], capture_output=True, text=True,
                               timeout=600)
            out = p.stdout
        except Exception:  # noqa: BLE001  单次调用失灵不拖垮整组
            continue
        parsed = _parse_pair(out)
        if parsed:
            break
    return prompt, out, parsed


def _win_graph_cycle(wins: dict[int, set], k: int) -> bool:
    """胜负图找环(A>B>C>A 一类;k≤8,DFS 三色)。"""
    color = [0] * k

    def dfs(u: int) -> bool:
        color[u] = 1
        for v in wins.get(u, ()):
            if color[v] == 1 or (color[v] == 0 and dfs(v)):
                return True
        color[u] = 2
        return False

    return any(color[i] == 0 and dfs(i) for i in range(k))


def judge_pairwise(items: list[tuple[str, str]], task: str, model: str = "haiku",
                   effort: str | None = None, log_path: Path | None = None
                   ) -> tuple[list[float], dict]:
    """成对判官协议 pairwise_v2:C(k,2) 对 × 正反两问 → 一致性门 → Copeland 计分。

    items = [(联络表图路径, 证据文本), ...]。返回 (copeland_scores, meta)。
    调用全量落盘(log_path,jsonl,每行含 prompt/reply/verdict/protocol 字段),
    同时是未来本地 RM 的训练数据;旧 rank_v1 落盘按 protocol 过滤。
    """
    from itertools import combinations
    k = len(items)
    records, outcome, n_fail = [], {}, 0
    for i, j in combinations(range(k), 2):
        res = []
        for (a, b), order in (((i, j), "fwd"), ((j, i), "rev")):
            prompt, out, parsed = judge_pair_call(items[a][0], items[a][1],
                                                  items[b][0], items[b][1],
                                                  task, model, effort)
            if parsed is None:             # 解析失败按 tie 计,并记数(别静默当胜负)
                n_fail += 1
                verdict = "tie"
            elif parsed["winner"] == "tie":
                verdict = "tie"
            else:
                verdict = a if parsed["winner"] == "a" else b
            records.append(dict(protocol="pairwise_v2", pair=[i, j], order=order,
                                a=a, b=b, model=model, effort=effort,
                                verdict=verdict, parsed=parsed,
                                reply=out.strip()[:2000], prompt=prompt))
            res.append(verdict)
        outcome[(i, j)] = res[0] if res[0] == res[1] else "tie"  # 一致性门:正反不一致⇒tie
    if log_path is not None:
        with open(log_path, "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    wins: dict[int, set] = {}
    for (i, j), v in outcome.items():
        if v != "tie":
            wins.setdefault(v, set()).add(j if v == i else i)
    cycle = _win_graph_cycle(wins, k)
    scores = [0.0] * k
    if not cycle:                          # 环 ⇒ 全组并列(scores 全零)
        for (i, j), v in outcome.items():
            if v == "tie":
                continue
            loser = j if v == i else i
            scores[v] += 1.0               # Copeland:胜 +1 / 负 -1 / tie 0
            scores[loser] -= 1.0
    ranks = {str(i): float(1 + sum(1 for s in scores if s > scores[i]))
             for i in range(k)}            # 竞赛名次,并列同名次
    meta = dict(protocol="pairwise_v2",
                judge=model + (f"+effort={effort}" if effort else ""),
                judge_calls=len(records), judge_call_fail=n_fail,
                pair_tie=sum(1 for v in outcome.values() if v == "tie"),
                cycle=cycle, copeland=scores, ranks=ranks)
    return scores, meta


def judge(g: int, rolls: list[dict], task: str = DEFAULT_TASK,
          model: str = "haiku", effort: str | None = None) -> tuple[np.ndarray, dict]:
    """成对判官 → Copeland 分 → 组内 z 归一化当优势。

    全并列(含胜负图有环)⇒ scores 全零 ⇒ group_advantage 全零(std_floor)⇒
    update 的 |adv|<1e-6 条款跳过该组,不产生假梯度。
    """
    items = []
    for j, r in enumerate(rolls):
        img = (OUT / f"g{g}_r{j}.png").resolve()
        contact_sheet(r["frames"], img)
        items.append((str(img), evidence_text(r)))
    (OUT / f"g{g}_items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1))

    scores, meta = judge_pairwise(items, task, model=model, effort=effort,
                                  log_path=OUT / f"g{g}_judge_pairs.jsonl")
    if meta["judge_call_fail"] == meta["judge_calls"]:  # 判官全灭 ⇒ 回退里程碑机器分
        scores = [float(len(r["inv_events"])) for r in rolls]
        meta = {"judge": "fallback_milestone", "protocol": "pairwise_v2",
                "ranks": None, "fallback": True}
    else:
        meta["fallback"] = False
    return group_advantage(scores), meta


# ────────────────────────────────────────────────────── rollout & update

def calibrate(env, no_op, obs) -> tuple[SelfCalib, object, np.ndarray]:
    """开局自标定(加固版):先压低视线(看地面纹理,天空低纹理测不出流),
    多幅度对称探针 → 延迟扫描 → 按延迟错位配对测增益/曲线 → 模式脉冲 →
    步速(pose,特权只进训练侧)。约 40 tick。产出 SelfCalib 随 episode 携带,
    以及标定期已发相机命令净和 net_cmd [2](yaw,pitch,度;v2 里程计的积分起点)。
    """
    calib = SelfCalib(img_w=IMG_HW[1], img_h=IMG_HW[0])
    net_cmd = np.zeros(2, np.float64)

    def small(o):
        return np.asarray(Image.fromarray(np.asarray(o["rgb"], np.uint8))
                          .resize((IMG_HW[1], IMG_HW[0])), np.float32)

    def cam_step(yaw, pitch):
        net_cmd[0] += yaw
        net_cmd[1] += pitch
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
    return calib, obs, net_cmd


def rollout(env, tower, slow, no_op, rng, ticks: int, device: str, temp: float,
            do_calib: bool = True, v2rt: V2Runtime | None = None) -> dict:
    """采一条轨迹。tower = PixelTower(v1)或 V2Policy(v2,配 v2rt 状态机)。"""
    from craftground.environment.action_space import no_op_v2  # noqa: F401
    obs, _ = env.reset()
    for _ in range(60):                                   # 等 "Loading terrain..."
        obs = env.step(no_op())[0]
    calib = SelfCalib(img_w=IMG_HW[1], img_h=IMG_HW[0])
    net_cmd = np.zeros(2)
    if do_calib:                                          # 自标定:物理参数测出来,不写死
        calib, obs, net_cmd = calibrate(env, no_op, obs)

    cfg = tower.tcfg if v2rt is not None else tower.cfg
    tower.eval()                                          # 修复②:采样在确定性网络上
    if v2rt is not None:                                  # v2:地图/钉点/里程计状态机就位
        v2rt.begin(calib, (float(net_cmd[0]), float(net_cmd[1])))
    goal = torch.zeros(getattr(cfg, "goal_dim", 384 + 2), device=device)
    imgs, prevs, goals, cam_b, key_b, frames, pose, cam_deg = [], [], [], [], [], [], [], []
    goal_log, inv_events, inv_steps = [], set(), {}
    prev = np.zeros(cfg.n_mouse + cfg.n_keys, np.float32)
    fstack: list[np.ndarray] = []                         # 最近 frame_stack 帧,旧→新
    death_step = None                                     # 死亡即截断(用户裁决 2026-07-10)

    for t in range(ticks):
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        if t % SLOW_EVERY == 0:                            # 慢塔按自身节拍刷新
            st = state_line(t, inv_steps, pose, goal_log) + "\n" + calib.physics_line()
            goal, rep = slow(rgb, st)
            if rep["prev_done"] and goal_log:              # prev_done 指上一条:回填标记
                goal_log[-1][5] = True
            goal_log.append([t, rep["subgoal"], rep["aim"], rep["done_when"],
                             rep["decision"], False])      # 修复④配套:aim/done_when 落盘
            if v2rt is not None:                           # v2:语言 token 换血 + aim 钉图
                v2rt.on_slow(rep)
        if t % max(1, ticks // 24) == 0:
            frames.append(np.asarray(Image.fromarray(rgb).resize((160, 90))))

        small = np.asarray(Image.fromarray(rgb).resize((IMG_HW[1], IMG_HW[0])),
                           dtype=np.float32) / 255.0
        if v2rt is not None:                               # v2:token 塔路径
            cam_l, key_l = v2rt.tick(small, prev)
            cam_l = cam_l / temp                           # [n_mouse, bins]
            key_l = key_l / temp                           # [n_keys]
        else:                                              # v1:像素塔路径(不动)
            fstack.append(small)
            if len(fstack) < cfg.frame_stack:              # 开局首帧填充
                fstack = [small] * (cfg.frame_stack - len(fstack)) + fstack
            fstack = fstack[-cfg.frame_stack:]
            stacked = np.concatenate(fstack, axis=2)       # [H,W,3S] 旧→新
            img = torch.from_numpy(stacked).permute(2, 0, 1)[None, None].to(device)
            pv = torch.from_numpy(prev)[None, None].to(device)
            with torch.no_grad():
                cam_l, key_l = tower(img, goal[None], pv)
            cam_l = cam_l[0, -1, 0] / temp                 # [n_mouse, bins]
            key_l = key_l[0, -1, 0] / temp                 # [n_keys]
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

        if bool(getattr(full, "is_dead", False)):          # 死亡即截断:上游 MinecraftEnv.kt:473
            death_step = t                                 # 禁死亡界面点击,自主复活不可能;
            break                                          # 不用 respawn 宏,证据文本写明死亡步

    return dict(imgs=np.stack(imgs), prevs=np.stack(prevs), goals=np.stack(goals),
                cam=np.stack(cam_b), keys=np.stack(key_b), cam_deg=np.stack(cam_deg),
                pose=np.asarray(pose, np.float32), frames=frames,
                goal_log=goal_log, inv_events=inv_events, inv_steps=inv_steps,
                calib=calib, death_step=death_step,
                **(v2rt.export() if v2rt is not None else {}))


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
    v2 = isinstance(tower, V2Policy)
    s = None if v2 else tower.cfg.frame_stack
    tot = 0.0
    for r, a_w in active:
        t_n = len(r["cam"])
        if not v2:
            stacked = stack_frames(r["imgs"], s)           # [T,3S,H,W] 与采样端同序
        for i0 in range(0, t_n, chunk):                    # 覆盖全部 tick,含尾段
            sl = slice(i0, min(i0 + chunk, t_n))
            if v2:                                         # v2:按记录 token 回放
                cam_l, key_l = v2_replay(tower, r, sl, device)
            else:
                img = torch.from_numpy(stacked[sl]).unsqueeze(1).to(device)  # [B,1,3S,H,W]
                pv = torch.from_numpy(r["prevs"][sl]).unsqueeze(1).to(device)  # [B,1,·]
                goal = torch.from_numpy(r["goals"][sl]).to(device)           # [B,386] 逐tick
                cam_l, key_l = tower(img, goal, pv)        # goal/prev 真的进梯度
                cam_l, key_l = cam_l[:, 0, 0], key_l[:, 0, 0]
            cam_l = cam_l / temp                           # [B,n_mouse,bins] 与采样同温度
            key_l = key_l / temp                           # [B,n_keys]
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
    ap.add_argument("--slow-model", default=MODEL,
                    help="慢塔 served-model-name(Omni/Qwen-VL 可互换,契约不变)")
    ap.add_argument("--goal-vocab-snap", default="runs/data/vpt_early_goal_vocab.json",
                    help="慢塔 subgoal 最近邻投影回 hindsight 词表(goal 三臂对照定案);空串关闭")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--rollout-ticks", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=128,
                    help="更新时的显存分块(tick 数);纯工程参数,不改数学")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--task", default=DEFAULT_TASK,
                    help="长程任务目标(领域知识只活在此行,换游戏换这里)")
    ap.add_argument("--tower", choices=("v1", "v2"), default="v1",
                    help="v1=像素塔(默认,契约冻结);v2=token 塔(DINO patch+地图+语言)")
    ap.add_argument("--dino", choices=("dinov3", "dinov2"), default="dinov3",
                    help="v2 视觉骨干(dinov3 gated 需 HF_TOKEN;dinov2 开放备选)")
    ap.add_argument("--no-calib", action="store_true", help="跳过开局自标定探针")
    ap.add_argument("--judge-model", default="haiku",
                    help="判官模型(claude CLI --model 值;对照臂可换 sonnet)")
    ap.add_argument("--judge-effort", default="",
                    help="判官推理档(claude CLI --effort:low/medium/high;"
                         "Sonnet-low 为 2026-07-10 A/B 定案口径;空=不传)")
    ap.add_argument("--seed-tries", type=int, default=8,
                    help="world seed 先验筛选:出生点 heightmap 48×48 无树则换 seed 的"
                         "最大尝试数(特权信息只进训练侧,保证任务物理可能)")
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

    v2rt = None
    if args.tower == "v1":                 # v1 路径与 checkpoint 契约不动
        cfg = PixelTowerConfig(img_hw=IMG_HW, goal_dim=384 + 2, n_keys=len(V2_KEYS),
                               camera_bins=CAM_BINS)
        tower = build_pixel_tower(cfg).to(device)
        if args.init_from:                 # BC 暖启动(结构 cfg 必须一致,load 严格校验)
            ck = torch.load(args.init_from, map_location=device, weights_only=True)
            tower.load_state_dict(ck["tower"])
            print(f"init from {args.init_from} (bc_step={ck.get('step')})", flush=True)
    else:                                  # v2:token 塔(DINO patch+地图+语言,goal-as-query)
        vcfg = V2Config(dino=args.dino)
        frontend = DinoFrontend(args.dino, device)
        tower = V2Policy(vcfg, frontend.enc_dim).to(device)
        cfg = tower.tcfg
        if args.init_from:
            ck = torch.load(args.init_from, map_location=device, weights_only=True)
            if "policy" not in ck:
                sys.exit("--init-from 给的是 v1 checkpoint,v2 塔不能加载"
                         "(v2 的 BC 暖启动尚未接线,见 status_built_not_wired)")
            tower.load_state_dict(ck["policy"])
            print(f"init from {args.init_from} (v2)", flush=True)
        v2rt = V2Runtime(tower, frontend, device)
    opt = torch.optim.AdamW(tower.parameters(), lr=args.lr)
    print(f"{type(tower).__name__} params = "
          f"{sum(p.numel() for p in tower.parameters()) / 1e6:.2f} M", flush=True)

    snap = args.goal_vocab_snap if Path(args.goal_vocab_snap or "").exists() else ""
    slow = SlowTower(args.base_url, encode_text, device, task=args.task,
                     model=args.slow_model, vocab_snap=snap)
    rng = np.random.default_rng(0)

    for g in range(args.groups):
        # world seed 先验筛选(实验设计,非拐棍):出生点 heightmap(48×48 列顶层
        # block_name,特权,只进训练侧)无 leaves/log ⇒「拿木头」物理不可能 ⇒ 换 seed。
        # 与受控对照固定 SCENE 同一性质;方法同 tests/probe_dino_aim.py 树候选筛查。
        t0 = time.time()
        env, wseed, has_tree, att = None, "", False, 0
        for att in range(args.seed_tries):
            wseed = str(int(rng.integers(0, 1 << 30)))
            env_cfg = InitialEnvironmentConfig(
                image_width=640, image_height=360,
                gamemode=GameMode.SURVIVAL, difficulty=Difficulty.PEACEFUL,
                world_type=WorldType.DEFAULT, seed=wseed,
                screen_encoding_mode=ScreenEncodingMode.RAW,
                requires_heightmap=True)
            env_cfg.set_allow_mob_spawn(False); env_cfg.freeze_time(True)
            env_cfg.freeze_weather(True)
            env = CraftGroundEnvironment(env_cfg,
                                         action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                                         port=args.port + g, find_free_port=True,
                                         verbose=False)
            obs0, _ = env.reset()
            for _ in range(60):                            # 等地形加载再读 heightmap
                obs0 = env.step(no_op_v2())[0]
            has_tree = any("leaves" in h.block_name or "log" in h.block_name
                           for h in obs0["full"].height_info)
            if has_tree or att == args.seed_tries - 1:
                break
            print(f"[g{g}] seed={wseed} 出生点无树,换 seed", flush=True)
            env.close()
        if not has_tree:
            print(f"[g{g}] {args.seed_tries} 次筛选无树,按最后 seed 继续(如实入档)",
                  flush=True)
        rolls = [rollout(env, tower, slow, no_op_v2, rng, args.rollout_ticks, device,
                         args.temp, do_calib=not args.no_calib, v2rt=v2rt)
                 for _ in range(args.per_group)]
        env.close()

        adv, jmeta = judge(g, rolls, task=args.task, model=args.judge_model,
                           effort=args.judge_effort or None)
        loss = update(tower, opt, rolls, adv, args.chunk, args.temp, device)
        if args.tower == "v1":             # v1/v2 checkpoint 分文件,互不污染
            torch.save(dict(tower=tower.state_dict(), cfg=vars(cfg), group=g),
                       OUT / "tower.pt")
        else:
            torch.save(dict(policy=tower.state_dict(), v2cfg=vars(vcfg),
                            tower_version="v2", group=g), OUT / "tower_v2.pt")

        m = dict(group=g, tower=args.tower, world_seed=wseed, n=len(rolls),
                 seed_attempts=att + 1, seed_has_tree=has_tree,
                 deaths=[r.get("death_step") for r in rolls],
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
