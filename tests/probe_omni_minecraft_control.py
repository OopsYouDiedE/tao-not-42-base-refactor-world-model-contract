#!/usr/bin/env python3
"""Nemotron-3-Nano-Omni(NVFP4)直接以像素操控 Minecraft 的能力探针。

协议(用户指定):
  - 模型以 **4 fps** 读取 Minecraft 画面;
  - 每读一帧,输出 **5 个动作** 填充到下一帧之前 ⇒ 20 动作/秒。
  - Minecraft 一 tick = 50ms ⇒ 20 tick/s ⇒ **恰好 1 动作 = 1 tick**。

这就是 action chunking(Figure Helix / π0 的慢-快系统同款),也正是本仓
knowledge/design_llm_deep_integration.md §1 说的"慢系统按自身节拍写、快系统非阻塞读"。

时序(实测,非估计):本机 5090 + NVFP4 上,640x360 一帧 + 只输出 5 个整数,
**热态全程 ~0.20s(TTFT 0.154s)**,落在 4fps 的 250ms 预算内 ⇒ 真实时闭环**可行**。
("VLM 推理是秒级"的直觉在这里不成立:输出只有 5 个 token 时解码不是瓶颈;
 秒级数字来自 thinking 模式几千 token 的长生成。)
仍采用 **lockstep(锁步)**:环境在等 VLM 时暂停 —— 让"动作质量"不被丢帧污染。
`vlm_latency_s` 如实记录,`realtime_feasible` 给出 p90 是否仍在 250ms 内。

动作空间:直接复用 train/craftground/env.py 的 DISCRETE_TO_V2(27 维),
**不使用** spaces.py 的 ACTION_NAMES 注释表 —— 该表与实际映射不符
(见 knowledge/design_llm_semantic_layer.md §3.1;实测 use=9 而非 11)。

── 验收(客观,不靠看图) ────────────────────────────────────────────────
用 **SURVIVAL** 模式(创造模式放方块不消耗,无法计数),给 64 个圆石:
  placed = 64 - inventory["cobblestone"]     ← 全局计数,与位置无关
另用 3x3x3 的 `surrounding_blocks` 作为"结构是否长在身边"的局部佐证,
`raycast_result` 记录准星指向,用于诊断 use 为何空放。

── 脚本化 oracle 先行标定的环境事实(2026-07-09 实测) ──────────────────
  * reset 后画面是 "Loading terrain..."(约 60 tick 才出真实地形)⇒ 必须 warmup;
  * 观测延迟一 tick:use 生效在下一帧的 inventory 上;
  * 放置有冷却:连续 use 不会每 tick 放一块;
  * camera pitch 是**累加**的(动作 17 每次 +30°,不自动回正)。
  写死策略 [17,17,9,0,4,4,9,...] 18 tick 放 2 块 ⇒ 工具链可测,模型失败才有意义。

用法:
    bash tests/serve_omni_nvfp4.sh                       # 终端 1
    Xvfb :99 -screen 0 1280x720x24 &                     # 终端 2
    DISPLAY=:99 python tests/probe_omni_minecraft_control.py --decisions 30
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

import numpy as np
from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "nemotron_3_nano_omni"
CHUNK = 5             # 每帧填充的动作数
FPS = 4               # 名义读帧率
WARMUP_TICKS = 60     # 等 "Loading terrain..." 消失
START_COBBLE = 64

ACTION_MENU = """\
0  = no-op (do nothing)
1  = walk forward
2  = walk backward
3  = strafe left
4  = strafe right
5  = jump
7  = attack / break the block you look at
9  = USE -> PLACES one cobblestone on the block face under your crosshair
13 = look down a little (pitch +15 deg, CUMULATIVE)
14 = look up a little   (pitch -15 deg, CUMULATIVE)
15 = turn right a little (yaw +15 deg)
16 = turn left a little  (yaw -15 deg)
17 = look down a lot    (pitch +30 deg, CUMULATIVE)
18 = look up a lot      (pitch -30 deg, CUMULATIVE)
"""

TASK = (
    "Build a small structure out of cobblestone on the flat grass: place at least 4 "
    "cobblestone blocks near each other to form a platform or a short wall. "
    "You hold a stack of cobblestone (see bottom-right of the screen). "
    "To place a block you must FIRST look down at the ground (actions 13/17 tilt the camera "
    "down and the tilt ACCUMULATES), so that the crosshair '+' in the centre of the screen "
    "points at a nearby grass or stone surface. THEN action 9 places a block there. "
    "If the crosshair points at the sky, action 9 does nothing. "
    "Placing has a short cooldown, so do not spam 9 every tick; interleave small moves "
    "(3/4 strafe, 1/2 walk) so the blocks land next to each other instead of on the same spot."
)

SYSTEM_HINT = (
    "You are controlling a Minecraft player from raw pixels. "
    f"Reply with EXACTLY {CHUNK} action indices as a JSON array of integers, e.g. [17,9,0,4,9]. "
    "No prose, no explanation, no markdown fence. Only the JSON array."
)


def b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_actions(text: str, n_actions: int) -> tuple[list[int], bool]:
    """抠出 CHUNK 个合法动作索引。返回 (actions, well_formed)。"""
    m = re.search(r"\[[^\]]*\]", text)
    if m:
        try:
            acts = [int(x) for x in json.loads(m.group())][:CHUNK]
            if len(acts) == CHUNK and all(0 <= a < n_actions for a in acts):
                return acts, True
        except (ValueError, TypeError):
            pass
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    acts = [a for a in nums if 0 <= a < n_actions][:CHUNK]
    acts += [0] * (CHUNK - len(acts))
    return acts, False


def cobble_inv(full) -> int:
    return sum(i.count for i in full.inventory if "cobblestone" in i.translation_key)


def cobble_around(full) -> int:
    return sum(1 for b in full.surrounding_blocks if "cobblestone" in b.translation_key)


def ray_target(full) -> str:
    rc = full.raycast_result
    if rc and rc.HasField("target_block"):
        return rc.target_block.translation_key.split(".")[-1]
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--decisions", type=int, default=30, help="VLM 决策次数(每次 5 动作)")
    ap.add_argument("--outdir", default="docs/results/omni_mc_control")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--scripted-oracle", action="store_true",
                    help="用写死策略跑同一任务,作为动作空间/测量链路的对照臂")
    args = ap.parse_args()

    if "DISPLAY" not in os.environ:
        sys.exit("need DISPLAY (`Xvfb :99 -screen 0 1280x720x24 &` then export DISPLAY=:99)")

    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.initial_environment_config import Difficulty, GameMode, WorldType
    from craftground.screen_encoding_modes import ScreenEncodingMode

    from train.craftground.env import DISCRETE_TO_V2  # 与训练同一份动作表

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        gamemode=GameMode.SURVIVAL,          # 创造模式不消耗方块 ⇒ 无法计数
        difficulty=Difficulty.PEACEFUL,
        world_type=WorldType.SUPERFLAT, seed="1234",
        screen_encoding_mode=ScreenEncodingMode.RAW,
        requires_surrounding_blocks=True, request_raycast=True,
    )
    cfg.set_allow_mob_spawn(False)
    cfg.freeze_time(True)                    # 光照恒定,排除昼夜对视觉的干扰
    cfg.freeze_weather(True)
    cfg.add_initial_inventory([("minecraft:cobblestone", START_COBBLE)])

    env = CraftGroundEnvironment(
        cfg, action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        port=args.port, find_free_port=True, verbose=False,
    )
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    n_actions = len(DISCRETE_TO_V2)

    obs, _ = env.reset()
    for _ in range(WARMUP_TICKS):            # 等 "Loading terrain..." 消失
        obs = env.step(no_op_v2())[0]
    print(f"warmup done: inv={cobble_inv(obs['full'])} around={cobble_around(obs['full'])}",
          flush=True)

    # 对照臂:写死策略。模型必须超过它,否则"能直控"这句话不成立。
    oracle = [17, 17, 9, 0, 4, 4, 9, 0, 4, 4, 9, 0, 3, 3, 3, 3, 9, 0, 1, 9]

    frames, log, malformed = [], [], 0
    for d in range(args.decisions):
        rgb = np.asarray(obs["rgb"], dtype="uint8")
        frames.append(rgb.copy())

        if args.scripted_oracle:
            acts = [oracle[(d * CHUNK + i) % len(oracle)] for i in range(CHUNK)]
            dt, text, ok = 0.0, "<oracle>", True
        else:
            prompt = (
                f"{SYSTEM_HINT}\n\nGOAL: {TASK}\n\n"
                f"AVAILABLE ACTIONS (index = meaning):\n{ACTION_MENU}\n"
                f"Decision {d + 1}/{args.decisions}. Each action runs for one game tick (50 ms).\n"
                + (f"Your previous {CHUNK} actions were {log[-1]['actions']}.\n" if log else "")
                + f"Look at the screenshot and output the next {CHUNK} actions."
            )
            t0 = time.perf_counter()
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": b64_png(rgb)}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=64, temperature=0.2,
                # instruct 模式的 model-card 参数:temperature 0.2 / top_k 1
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
            )
            dt = time.perf_counter() - t0
            text = (r.choices[0].message.content or "").strip()
            acts, ok = parse_actions(text, n_actions)
            malformed += (not ok)

        rays = []
        for a in acts:
            obs = env.step(DISCRETE_TO_V2[a])[0]
            rays.append(ray_target(obs["full"]))

        full = obs["full"]
        placed = START_COBBLE - cobble_inv(full)
        log.append({
            "decision": d, "vlm_latency_s": round(dt, 3), "raw": text[:60],
            "actions": acts, "well_formed": ok, "placed_total": placed,
            "around": cobble_around(full), "pitch": round(full.pitch, 1),
            "ray_targets": rays,
        })
        print(f"[{d + 1:2d}/{args.decisions}] {dt:5.2f}s acts={acts} placed={placed} "
              f"around={log[-1]['around']} pitch={log[-1]['pitch']:6.1f} "
              f"ray={rays[-1]}{'' if ok else ' (malformed)'}", flush=True)

    frames.append(np.asarray(obs["rgb"], dtype="uint8").copy())
    full = obs["full"]
    placed = START_COBBLE - cobble_inv(full)

    sel = frames[:: max(1, len(frames) // 8)][:8]
    Image.fromarray(np.concatenate(sel, axis=1)).save(outdir / "filmstrip.png")
    Image.fromarray(frames[-1]).save(outdir / "final_frame.png")

    lat = [x["vlm_latency_s"] for x in log]
    all_acts = [a for x in log for a in x["actions"]]
    summary = {
        "arm": "scripted_oracle" if args.scripted_oracle else "vlm",
        "decisions": args.decisions, "chunk": CHUNK, "nominal_fps": FPS,
        "vlm_latency_s": {"mean": round(float(np.mean(lat)), 3),
                          "p50": round(float(np.percentile(lat, 50)), 3),
                          "p90": round(float(np.percentile(lat, 90)), 3)},
        "realtime_budget_s": 1.0 / FPS,
        "realtime_feasible": bool(np.percentile(lat, 90) <= 1.0 / FPS),
        "malformed_outputs": malformed,
        "well_formed_rate": round(1 - malformed / args.decisions, 3),
        "blocks_placed": placed,
        "cobblestone_around_player": cobble_around(full),
        "final_pitch": round(full.pitch, 1),
        "action_histogram": {str(a): all_acts.count(a) for a in sorted(set(all_acts))},
        "place_action_share": round(all_acts.count(9) / len(all_acts), 3),
        "steps": log,
    }
    name = "summary_oracle.json" if args.scripted_oracle else "summary_vlm.json"
    (outdir / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "steps"},
                            indent=2, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    main()
