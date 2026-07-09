#!/usr/bin/env python3
"""Nemotron-3-Nano-Omni(NVFP4)直接以像素操控 Minecraft 的能力探针。

协议(用户指定):
  - 模型以 **4 fps** 读取 Minecraft 画面;
  - 每读一帧,输出 **5 个动作** 填充到下一帧之前 ⇒ 20 动作/秒。
  - Minecraft 一 tick = 50ms ⇒ 20 tick/s ⇒ **恰好 1 动作 = 1 tick**,数字对得上。

这就是 action chunking(Figure Helix / π0 的慢-快系统同款),也正是本仓
knowledge/design_llm_deep_integration.md §1 说的"慢系统按自身节拍写、快系统非阻塞读"。

⚠ 时序诚实说明:VLM 单次推理是**秒级**,4fps 要求 250ms/次 —— 真实时闭环做不到。
本脚本因此采用 **lockstep(锁步)**:环境在等 VLM 时暂停。测出的 `vlm_latency_s` 直接
回答"离真 4fps 差多远",而不是假装跑到了。

动作空间:直接复用 train/craftground/env.py 的 DISCRETE_TO_V2(27 维),
**不使用** spaces.py 的 ACTION_NAMES 注释表 —— 该表与实际映射不符
(见 knowledge/design_llm_semantic_layer.md §3.1,实测 use=9 而非 11)。

验收:靠 CraftGround 的 `requires_surrounding_blocks` 真值数方块,不靠看图感觉。

用法:
    bash tests/serve_omni_nvfp4.sh                       # 终端 1
    Xvfb :99 -screen 0 1280x720x24 &                     # 终端 2
    DISPLAY=:99 python tests/probe_omni_minecraft_control.py --decisions 24
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
CHUNK = 5          # 每帧填充的动作数
FPS = 4            # 名义读帧率
TICKS_PER_ACTION = 1

# 供模型选择的动作子集(27 维里与"搭建"相关的那些)。索引严格对应 DISCRETE_TO_V2。
ACTION_MENU = """\
0  = no-op (do nothing)
1  = walk forward
2  = walk backward
3  = strafe left
4  = strafe right
5  = jump
7  = attack / break block
9  = USE  (this PLACES a cobblestone block where you are looking)
13 = look down a little (pitch +15 deg)
14 = look up a little   (pitch -15 deg)
15 = turn right a little (yaw +15 deg)
16 = turn left a little  (yaw -15 deg)
17 = look down a lot    (pitch +30 deg)
18 = look up a lot      (pitch -30 deg)
"""

TASK = (
    "Build a small structure out of cobblestone on the flat ground: "
    "place at least 4 cobblestone blocks next to each other to form a platform or a wall. "
    "You are in CREATIVE mode holding an unlimited stack of cobblestone. "
    "To place a block you must (a) look DOWN at the ground (action 13 or 17), then (b) use action 9. "
    "Placing works only if the crosshair in the center of the screen points at a block face that is close enough."
)

SYSTEM_HINT = (
    "You are controlling a Minecraft player from raw pixels. "
    f"Reply with EXACTLY {CHUNK} action indices as a JSON array of integers, e.g. [13,9,1,9,0]. "
    "No prose, no explanation, no markdown fence. Only the JSON array."
)


def b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_actions(text: str, n_actions: int) -> tuple[list[int], bool]:
    """从模型输出里抠出 CHUNK 个合法动作索引。返回 (actions, well_formed)。"""
    m = re.search(r"\[[^\]]*\]", text)
    if m:
        try:
            raw = json.loads(m.group())
            acts = [int(x) for x in raw][:CHUNK]
            if len(acts) == CHUNK and all(0 <= a < n_actions for a in acts):
                return acts, True
        except (ValueError, TypeError):
            pass
    # 退化路径:抓所有整数
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    acts = [a for a in nums if 0 <= a < n_actions][:CHUNK]
    acts += [0] * (CHUNK - len(acts))  # 不足补 no-op
    return acts, False


def count_placed(env_obs, block: str = "cobblestone") -> int:
    """从 surrounding_blocks 真值里数目标方块数量(不靠看图)。"""
    full = env_obs.get("full")
    if full is None:
        return -1
    blocks = getattr(full, "surrounding_blocks", None)
    if not blocks:
        return -1
    return sum(1 for b in blocks if block in str(b).lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--decisions", type=int, default=24, help="VLM 决策次数(每次 5 动作)")
    ap.add_argument("--outdir", default="docs/results/omni_mc_control")
    ap.add_argument("--port", type=int, default=8030)
    args = ap.parse_args()

    if "DISPLAY" not in os.environ:
        sys.exit("need DISPLAY (run `Xvfb :99 -screen 0 1280x720x24 &` and export DISPLAY=:99)")

    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.initial_environment_config import Difficulty, GameMode, WorldType
    from craftground.screen_encoding_modes import ScreenEncodingMode

    from train.craftground.env import DISCRETE_TO_V2  # 与训练同一份动作表

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        gamemode=GameMode.CREATIVE, difficulty=Difficulty.PEACEFUL,
        world_type=WorldType.SUPERFLAT, seed="1234",
        screen_encoding_mode=ScreenEncodingMode.RAW,
        requires_surrounding_blocks=True,
    )
    cfg.set_allow_mob_spawn(False)
    cfg.freeze_time(True)          # 光照恒定,排除昼夜对视觉的干扰
    cfg.freeze_weather(True)
    cfg.add_initial_inventory([("minecraft:cobblestone", 64)])
    cfg.set_initial_position(0.5, 4.0, 0.5, yaw=0.0, pitch=20.0)  # 略微俯视,方便放置

    env = CraftGroundEnvironment(
        cfg, action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        port=args.port, find_free_port=True, verbose=False,
    )
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    obs, _ = env.reset()
    baseline = count_placed(obs)
    frames, log = [], []
    print(f"reset ok; baseline cobblestone in surroundings = {baseline}", flush=True)

    n_actions = len(DISCRETE_TO_V2)
    malformed = 0
    for d in range(args.decisions):
        rgb = obs["rgb"]
        frames.append(rgb.copy())

        prompt = (
            f"{SYSTEM_HINT}\n\nGOAL: {TASK}\n\n"
            f"AVAILABLE ACTIONS (index = meaning):\n{ACTION_MENU}\n"
            f"Decision {d + 1}/{args.decisions}. "
            f"Each action you output runs for one game tick (50 ms).\n"
            f"Look at the screenshot and output the next {CHUNK} actions."
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

        for a in acts:
            step = env.step(DISCRETE_TO_V2[a])
            obs = step[0]

        placed = count_placed(obs)
        log.append({
            "decision": d, "vlm_latency_s": round(dt, 3), "raw": text[:80],
            "actions": acts, "well_formed": ok, "cobblestone_seen": placed,
        })
        print(f"[{d + 1:2d}/{args.decisions}] {dt:5.2f}s  acts={acts}  "
              f"cobble={placed}  {'' if ok else '(malformed)'}", flush=True)

    frames.append(obs["rgb"].copy())
    final = count_placed(obs)

    # 胶片图:每 4 个决策抽一帧
    sel = frames[:: max(1, len(frames) // 8)][:8]
    strip = np.concatenate([np.asarray(f) for f in sel], axis=1)
    Image.fromarray(strip.astype("uint8")).save(outdir / "filmstrip.png")
    Image.fromarray(frames[-1].astype("uint8")).save(outdir / "final_frame.png")

    lat = [x["vlm_latency_s"] for x in log]
    summary = {
        "decisions": args.decisions,
        "chunk": CHUNK,
        "nominal_fps": FPS,
        "vlm_latency_s": {
            "mean": round(float(np.mean(lat)), 3),
            "p50": round(float(np.percentile(lat, 50)), 3),
            "p90": round(float(np.percentile(lat, 90)), 3),
        },
        "realtime_budget_s": 1.0 / FPS,
        "realtime_feasible": bool(np.percentile(lat, 90) <= 1.0 / FPS),
        "malformed_outputs": malformed,
        "well_formed_rate": round(1 - malformed / args.decisions, 3),
        "place_action_share": round(
            sum(a == 9 for x in log for a in x["actions"]) / (args.decisions * CHUNK), 3),
        "cobblestone_baseline": baseline,
        "cobblestone_final": final,
        "blocks_placed": (final - baseline) if final >= 0 and baseline >= 0 else None,
        "steps": log,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "steps"},
                            indent=2, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    main()
