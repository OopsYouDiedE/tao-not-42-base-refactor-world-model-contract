#!/usr/bin/env python3
"""Lumine 式像素直控 Minecraft(Nemotron-3-Nano-Omni NVFP4,单卡 5090)。

对照 Lumine(arXiv:2511.08892,Qwen2-VL-7B 底座,原生玩原神)的四条核心设计,
逐条移植到 Minecraft / CraftGround:

  Lumine                                   本脚本
  ────────────────────────────────────     ────────────────────────────────────
  语言原生动作串,非离散索引                 同左(见下 GRAMMAR)
    "92 0 0 ; Shift W ; Shift W ; F W"
  6 个 33ms 按键块 = 200ms 窗口(30Hz)      5 个 50ms 按键块 = 250ms 窗口(20Hz)
                                            (Minecraft 1 tick = 50ms,1 块 = 1 tick)
  5Hz 感知                                  4Hz 感知(用户指定)
  hybrid thinking:模型自己决定要不要先想    同左,<thought>...</thought> 可选
  滑窗 20 组 (图,动作) 多轮对话             --history 0 = **不滑窗,全灌**
                                            (Mamba 状态常数大小,一帧仅 ~298 token)

为什么放弃 27 维离散索引(v1 的设计错误,已证伪):
  v1 让模型输出 [13,13,9,1,9] 这类索引且**不给历史**。结果它锁死在同一个 chunk 上、
  pitch 撞满 90° 仍继续输出"再低头",150 tick 只放了 3 块。根因不是看不懂画面
  (语义探针里它能数清 Crafter 里的三棵树),而是:
    (a) 索引抹掉了语义,把 VLM 降级成查表器;
    (b) 无历史 ⇒ 它看不到"自己已经低头了",而盯着脚下时画面几乎不变 ⇒ 输出自锁。
  Lumine 的 pitch 累加问题靠"历史里有自己上一步的 ΔPITCH"来闭环,这里照搬。

── 动作语法(GRAMMAR) ──────────────────────────────────────────────────
    <action>ΔYAW ΔPITCH ; c1 ; c2 ; c3 ; c4 ; c5</action>

  ΔYAW   相机水平转动,度,正=向右,整数,[-180,180]
  ΔPITCH 相机俯仰,度,正=向下,整数,[-90,90];**累加**到当前朝向
  Ki     第 i 个 tick(50ms)按下的键集合,空格分隔;`-` 表示全部松开
         合法键:W A S D SPACE SHIFT CTRL LMB RMB
           W/A/S/D=前/左/后/右   SPACE=跳   SHIFT=潜行   CTRL=疾跑
           LMB=攻击/挖掘        RMB=使用/放置方块
  与 Lumine 一致:未列出的键在该 tick 自动松开;相机位移在窗口起始一次性施加。

── 验收(客观,不靠看图) ────────────────────────────────────────────────
  SURVIVAL 模式(创造模式放方块不消耗,无法计数),给 64 圆石:
    placed = 64 - inventory["cobblestone"]      ← 全局计数,与位置无关
  另记 3x3x3 `surrounding_blocks` 与 `raycast_result` 供诊断。

  对照臂 `--scripted-oracle`:写死策略。模型不超过它,"能直控"就不成立。

用法:
    bash tests/serve_omni_nvfp4.sh                    # 需 image 上限 >= --history+2
    Xvfb :99 -screen 0 1280x720x24 &
    DISPLAY=:99 python tests/probe_omni_minecraft_lumine.py --decisions 40 --history 0
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import math

import numpy as np
from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "nemotron_3_nano_omni"
CHUNK = 5              # 5 个 tick = 250ms = 4fps 的一帧间隔
WARMUP_TICKS = 60      # 等 "Loading terrain..." 消失
START_COBBLE = 64

KEYMAP = {  # 键 -> craftground V2 动作 dict 的字段
    "W": "forward", "S": "back", "A": "left", "D": "right",
    "SPACE": "jump", "SHIFT": "sneak", "CTRL": "sprint",
    "LMB": "attack", "RMB": "use",
}

IMG_W, IMG_H, FOV_V = 640, 360, 70.0   # craftground InitialEnvironmentConfig(fov=70),竖直 FOV

def pixel_to_delta(nx: float, ny: float) -> tuple[float, float]:
    """归一化点(0..1000,原点左上)-> (dyaw, dpitch) 度。把该点转到准星处。

    模型的坐标约定由 tests/probe_omni_pointing.py 实测定出:**1000x1000 归一化**,
    指点精度 2.2-5.4 px(640x360 下 <1%)。用它自己的表示喂动作,而不是逼它算角度
    ——同一模型被直接问"该转多少度"时答 -1(符号都错)。
    """
    px, py = nx / 1000.0 * IMG_W, ny / 1000.0 * IMG_H
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    t = math.tan(math.radians(FOV_V) / 2.0)
    dyaw = math.degrees(math.atan(((px - cx) / cx) * t * (IMG_W / IMG_H)))
    dpitch = math.degrees(math.atan(((py - cy) / cy) * t))   # 正 = 向下
    return dyaw, dpitch


ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)
THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.S)

SYSTEM_DEG = """\
You are playing Minecraft. You see the game as a human does: raw pixels, 640x360, with the
crosshair '+' at the centre of the screen and the hotbar at the bottom.

Each turn you emit ONE action line, in exactly this shape:

<action>A B ; c1 ; c2 ; c3 ; c4 ; c5</action>

  A  = camera turn in degrees (integer). positive turns RIGHT, negative LEFT, 0 = no turn.
  B  = camera tilt in degrees (integer). positive looks DOWN, negative looks UP, 0 = no tilt.
       IT IS CUMULATIVE: it adds to where you are ALREADY looking. Straight ahead is 0,
       straight down is 90. If you already tilted down 45, sending 45 again points you at
       your own feet, which is useless. Send 0 to hold the current tilt.
  c1..c5 = the keys held during each of the next 5 game ticks (50 ms each), in order.
       Separate the five groups with ';'. Use '-' for "no keys held this tick".
       Valid keys: W A S D SPACE SHIFT CTRL LMB RMB
         W/A/S/D = walk forward / strafe left / walk back / strafe right
         SPACE = jump   SHIFT = sneak   CTRL = sprint
         LMB = attack / break the block at the crosshair
         RMB = place one cobblestone on the surface at the crosshair

Real examples of valid output (copy this shape exactly; never write the letters A, B, c1):

<action>0 40 ; - ; - ; - ; - ; -</action>
<action>0 0 ; D ; D ; - ; RMB ; -</action>
<action>-20 0 ; W ; W ; W ; - ; RMB</action>
<action>0 10 ; - ; LMB ; LMB ; LMB ; -</action>

You MAY think first, but only when the situation changed and your plan needs revising.
At most 20 words:

<thought>brief reasoning</thought>

Rules that matter:
  - RMB places a block ONLY if the crosshair points at a nearby solid surface.
    Pointing at the sky, or at a spot too far away, does nothing.
  - Placing has a cooldown of a few ticks. Pressing RMB every tick wastes most of them;
    press it at most once or twice per action line.
  - You are in survival mode holding a stack of cobblestone (bottom-right of the screen).
  - Your earlier turns are in this conversation. Use them to remember how far you have
    already tilted down and where you already placed blocks.

Output the optional <thought>, then exactly one <action> line. Nothing else.
"""

# ── pixel 瞄准模式:用模型被训练过的 grounding 表示,而不是角度 ────────────────
SYSTEM_PIX = """\
You are playing Minecraft. You see the game as a human does: raw pixels, with the crosshair '+'
at the centre of the screen and the hotbar at the bottom.

Each turn you emit ONE action line:

<action>X Y ; c1 ; c2 ; c3 ; c4 ; c5</action>

── PART 1: "X Y" is WHERE TO AIM, read off THIS screenshot ──────────────────
Normalised image coordinates 0..1000: (0,0) = top-left, (1000,1000) = bottom-right.
The crosshair sits at (500,500). The camera turns so the point you name moves to the centre.

  Read the CURRENT screenshot every turn and answer: where is the spot I want to act on?
    - I want to place a block on grass that is 2 blocks ahead -> that grass is BELOW centre,
      so name something like `500 700`.
    - The grass I want is ALREADY under the crosshair -> name `500 500` (hold the camera).
    - I am staring at my own feet (screen is all grass, no horizon) -> I tilted too far down.
      Look back up by naming a point ABOVE centre, e.g. `500 250`.

  CRITICAL: this is not a fixed offset to repeat. Naming `500 700` every turn will tilt the
  camera down 15 degrees every turn until you stare at your feet and can build nothing.
  After the camera has moved, the same spot is at a NEW place on screen. Look again.

── PART 2: "c1 ; c2 ; c3 ; c4 ; c5" is a TIMELINE, not a menu ────────────────
It is FIVE consecutive game ticks, 50 ms each, played in order, left to right. c1 happens
first, then c2, then c3, then c4, then c5. Each ci is the SET of keys held during that tick
(space-separated), or `-` for no keys. A key you do not list is released for that tick.

You are composing a 250 ms movement. Use all five slots. Worked example:

  <action>500 500 ; RMB ; - ; D ; D ; RMB</action>
     tick 1: press RMB      -> place a cobblestone where the crosshair points
     tick 2: nothing        -> let the place-cooldown tick over
     tick 3: hold D         -> strafe right
     tick 4: hold D         -> keep strafing (now standing next to the block just placed)
     tick 5: press RMB      -> place the next block beside the first one

  That single line builds two blocks of a wall. THIS is the shape you want.

More valid lines:
  <action>500 700 ; - ; - ; - ; - ; -</action>          (only look down, do nothing)
  <action>500 500 ; W ; W ; W ; - ; RMB</action>        (walk forward 3 ticks, then place)
  <action>420 640 ; RMB ; - ; - ; A ; A</action>        (place, then strafe left)

Keys: W A S D SPACE SHIFT CTRL LMB RMB
  W/A/S/D = walk forward / strafe left / walk back / strafe right
  SPACE = jump   SHIFT = sneak   CTRL = sprint
  LMB = break the block at the crosshair
  RMB = place ONE cobblestone on the surface at the crosshair

── Rules that decide success ────────────────────────────────────────────────
  - RMB does nothing if the crosshair points at the sky, or at a spot too far away.
  - Placing has a cooldown of ~4 ticks. Two RMB in one line is the most that can work,
    and only if they are separated by at least 3 other ticks.
  - To build a ROW, aim NEXT TO the last block you placed, never at it. If the crosshair
    is on cobblestone you already placed, strafe first, or aim at the grass beside it.
  - You are in survival mode holding a stack of cobblestone (bottom-right of the screen).
  - Your earlier turns are in this conversation. Use them to see how far you have already
    tilted and whether the camera actually moved.

You MAY think first, at most 20 words, only when the plan needs revising:
<thought>brief reasoning</thought>

Output the optional <thought>, then exactly one <action> line. Nothing else.
"""

TASK = (
    "GOAL: build a small cobblestone structure on the flat grass — place at least 4 "
    "cobblestone blocks next to each other, forming a platform or a short wall. "
    "Tilt your view down so the crosshair rests on the ground a couple of blocks ahead, "
    "then alternate: place a block, take a small step or strafe, place the next one."
)


# ───────────────────────────────────────────────────────────── parsing

def b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_action(text: str, aim: str = "degrees") -> tuple[float, float, list[list[str]], bool]:
    """解析 Lumine 式动作串。返回 (dyaw, dpitch, 5xkeys, well_formed)。

    aim="degrees": 头两个数是 ΔYAW ΔPITCH(度)
    aim="pixel"  : 头两个数是归一化瞄准点 X Y(0..1000),经 pixel_to_delta 换算
    """
    m = ACTION_RE.search(text)
    body = m.group(1) if m else text
    parts = [p.strip() for p in body.split(";")]
    if len(parts) < 2:
        return 0.0, 0.0, [[] for _ in range(CHUNK)], False

    head = parts[0].split()
    try:
        a, b = float(head[0]), float(head[1])
    except (IndexError, ValueError):
        return 0.0, 0.0, [[] for _ in range(CHUNK)], False
    if aim == "pixel":
        dyaw, dpitch = pixel_to_delta(np.clip(a, 0, 1000), np.clip(b, 0, 1000))
    else:
        dyaw, dpitch = a, b

    chunks: list[list[str]] = []
    for p in parts[1:1 + CHUNK]:
        if p in ("-", ""):
            chunks.append([])
        else:
            keys = [k.upper() for k in p.split() if k.upper() in KEYMAP]
            chunks.append(keys)
    ok = bool(m) and len(chunks) == CHUNK
    while len(chunks) < CHUNK:
        chunks.append([])
    dyaw = float(np.clip(dyaw, -180, 180))
    dpitch = float(np.clip(dpitch, -90, 90))
    return dyaw, dpitch, chunks[:CHUNK], ok


def to_v2(no_op, keys: list[str], dyaw: float, dpitch: float) -> dict:
    a = no_op()
    for k in keys:
        a[KEYMAP[k]] = True
    a["camera_yaw"] = float(dyaw)
    a["camera_pitch"] = float(dpitch)
    return a


# ───────────────────────────────────────────────────────── observations

def cobble_inv(full) -> int:
    return sum(i.count for i in full.inventory if "cobblestone" in i.translation_key)


def cobble_around(full) -> int:
    return sum(1 for b in full.surrounding_blocks if "cobblestone" in b.translation_key)


def ray_target(full) -> str:
    rc = full.raycast_result
    if rc and rc.HasField("target_block"):
        return rc.target_block.translation_key.split(".")[-1]
    return "-"


# 写死对照臂。参数由 12 组(低头角度 x 移动模式)的扫描实测定出:
#   pitch0=45 + strafe_then_place -> 12 决策放 12 块(≈ 每决策 1 块,贴放置冷却上限)
#   pitch0=60 -> 只放 1-6 块。**低头过度是本任务的主要失败模式**。
# 注意 pitch 只在第 0 个决策施加一次(它是累加量),之后恒为 0。
ORACLE_FIRST = {"degrees": "<action>0 45 ; - ; - ; - ; - ; -</action>",
                "pixel": "<action>500 950 ; - ; - ; - ; - ; -</action>"}
ORACLE_LOOP = {"degrees": "<action>0 0 ; D ; D ; - ; RMB ; -</action>",
               "pixel": "<action>500 500 ; D ; D ; - ; RMB ; -</action>"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--decisions", type=int, default=40)
    ap.add_argument("--history", type=int, default=0,
                    help="保留多少组 (图,动作) 历史;0 = 全灌(利用 Mamba 常数状态)")
    ap.add_argument("--outdir", default="docs/results/omni_mc_control")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--scripted-oracle", action="store_true")
    ap.add_argument("--assist-tilt", type=float, default=0.0,
                    help="开局由脚本代替模型把相机下俯 N 度(实测甜区 35-45)。"
                         "用于把'不会转视角(连续标定)'与'不会搭建(离散序列)'拆开。")
    ap.add_argument("--aim", choices=["degrees", "pixel"], default="pixel",
                    help="pixel: 模型输出归一化瞄准点(它被训练过的 grounding 表示);"
                         "degrees: 输出相机角增量(实测符号不稳,见 conclusion 文档)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--greedy", action="store_true", default=False,
                    help="top_k=1。贪心 + 自我历史会导致输出坍塌,默认关闭。")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if "DISPLAY" not in os.environ:
        sys.exit("need DISPLAY (`Xvfb :99 -screen 0 1280x720x24 &`)")

    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.initial_environment_config import Difficulty, GameMode, WorldType
    from craftground.screen_encoding_modes import ScreenEncodingMode

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        gamemode=GameMode.SURVIVAL,        # 创造模式不消耗方块 ⇒ 无法计数
        difficulty=Difficulty.PEACEFUL,
        world_type=WorldType.SUPERFLAT, seed="1234",
        screen_encoding_mode=ScreenEncodingMode.RAW,
        requires_surrounding_blocks=True, request_raycast=True,
    )
    cfg.set_allow_mob_spawn(False)
    cfg.freeze_time(True); cfg.freeze_weather(True)
    cfg.add_initial_inventory([("minecraft:cobblestone", START_COBBLE)])

    env = CraftGroundEnvironment(
        cfg, action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        port=args.port, find_free_port=True, verbose=False,
    )
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    obs, _ = env.reset()
    for _ in range(WARMUP_TICKS):
        obs = env.step(no_op_v2())[0]
    if args.assist_tilt:
        obs = env.step(to_v2(no_op_v2, [], 0.0, args.assist_tilt))[0]
        for _ in range(4):
            obs = env.step(no_op_v2())[0]
    print(f"warmup done: inv={cobble_inv(obs['full'])} "
          f"pitch={obs['full'].pitch:.1f} (assist_tilt={args.assist_tilt})", flush=True)

    assist_note = (f" Your view is ALREADY tilted down {args.assist_tilt:.0f} degrees and the "
                   "crosshair rests on the grass ahead — send tilt 0 to keep it there."
                   if args.assist_tilt else "")

    history: list[dict] = []          # 多轮对话:user(图) / assistant(动作串)
    log, malformed, thoughts = [], 0, 0

    for d in range(args.decisions):
        rgb = np.asarray(obs["rgb"], dtype="uint8")

        if args.scripted_oracle:
            raw = (ORACLE_FIRST if d == 0 else ORACLE_LOOP)[args.aim]
            dt, thought = 0.0, ""
        else:
            user_content = [
                {"type": "image_url", "image_url": {"url": b64_png(rgb)}},
                {"type": "text", "text": (TASK + assist_note) if d == 0 else "Next action."},
            ]
            sysmsg = SYSTEM_PIX if args.aim == "pixel" else SYSTEM_DEG
            msgs = [{"role": "system", "content": sysmsg}] + history + \
                   [{"role": "user", "content": user_content}]
            t0 = time.perf_counter()
            r = client.chat.completions.create(
                model=MODEL, messages=msgs, max_tokens=200,
                temperature=args.temperature, top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": False},
                            **({"top_k": 1} if args.greedy else {})},
            )
            dt = time.perf_counter() - t0
            raw = (r.choices[0].message.content or "").strip()
            tm = THOUGHT_RE.search(raw)
            thought = tm.group(1).strip() if tm else ""
            thoughts += bool(thought)

            # 历史:图 + 模型自己的动作串。history=0 ⇒ 全灌(Mamba 常数状态)
            history.append({"role": "user", "content": user_content})
            history.append({"role": "assistant", "content": raw})
            if args.history > 0:
                history[:] = history[-2 * args.history:]

        dyaw, dpitch, chunks, ok = parse_action(raw, args.aim)
        malformed += (not ok)

        rays = []
        for i, keys in enumerate(chunks):
            # 与 Lumine 一致:相机位移在窗口起始一次性施加
            a = to_v2(no_op_v2, keys, dyaw if i == 0 else 0.0, dpitch if i == 0 else 0.0)
            obs = env.step(a)[0]
            rays.append(ray_target(obs["full"]))

        full = obs["full"]
        placed = START_COBBLE - cobble_inv(full)
        log.append({
            "decision": d, "latency_s": round(dt, 3), "raw": raw[:120],
            "thought": thought, "dyaw": dyaw, "dpitch": dpitch,
            "chunks": chunks, "well_formed": ok,
            "placed_total": placed, "around": cobble_around(full),
            "pitch": round(full.pitch, 1), "ray_targets": rays,
        })
        keystr = "|".join(" ".join(c) if c else "-" for c in chunks)
        print(f"[{d+1:2d}/{args.decisions}] {dt:5.2f}s yaw={dyaw:+5.0f} pitch={dpitch:+4.0f} "
              f"[{keystr}] placed={placed} pitch_abs={full.pitch:5.1f} ray={rays[-1]}"
              f"{'' if ok else ' (malformed)'}{' 💭' if thought else ''}", flush=True)

    final = obs["full"]
    placed = START_COBBLE - cobble_inv(final)
    Image.fromarray(np.asarray(obs["rgb"], dtype="uint8")).save(
        outdir / f"final_{args.tag or ('oracle' if args.scripted_oracle else 'lumine')}.png")

    lat = [x["latency_s"] for x in log]
    summary = {
        "arm": args.tag or ("scripted_oracle" if args.scripted_oracle else "lumine_vlm"),
        "aim": args.aim, "assist_tilt": args.assist_tilt,
        "temperature": args.temperature, "greedy": args.greedy,
        "history_mode": "unbounded" if args.history == 0 else f"sliding_{args.history}",
        "decisions": args.decisions, "chunk_ticks": CHUNK, "nominal_fps": 4,
        "latency_s": {"mean": round(float(np.mean(lat)), 3),
                      "p50": round(float(np.percentile(lat, 50)), 3),
                      "p90": round(float(np.percentile(lat, 90)), 3)},
        "realtime_budget_s": 0.25,
        "realtime_feasible": bool(np.percentile(lat, 90) <= 0.25),
        "malformed": malformed,
        "well_formed_rate": round(1 - malformed / args.decisions, 3),
        "thought_rate": round(thoughts / args.decisions, 3),
        "blocks_placed": placed,
        "cobblestone_around_player": cobble_around(final),
        "final_pitch": round(final.pitch, 1),
        "rmb_ticks": sum(1 for x in log for c in x["chunks"] if "RMB" in c),
        "distinct_actions": len({json.dumps(x["chunks"]) for x in log}),
        "steps": log,
    }
    name = f"summary_{summary['arm']}.json"
    (outdir / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "steps"},
                            indent=2, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    main()
