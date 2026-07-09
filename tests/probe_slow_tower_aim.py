#!/usr/bin/env python3
"""慢塔指点体检:Omni 能不能在真 Minecraft 画面里指出"树在哪个像素"?

这是双系统(慢塔给文本子目标+目标像素,快塔学连续标定)的**地基假设**。
地基不成立就别写 GRPO 循环了。

判据完全客观,不靠看图:
    把慢塔给的归一化像素 -> (dyaw,dpitch) -> 转过去 -> 读 `raycast_result`,
    命中 `*_log` / `*_leaves` 记为指对了树。
对照臂(**同一批画面**,rng_view 独立于 rng_aim):
  `--arm random`   随机瞄一点;
  `--arm constant` 恒用 [430,560](旧版 prompt 里泄漏过的示例坐标)。
慢塔不显著高于这两者 = 地基不成立。

主指标是 **trunk_hit_rate**(命中 `*_log`/`*_wood`)。
早先版本把 `leaves` 也算命中,得到慢塔 0.75 —— 与 constant 臂**完全相等**,
即"0.75"全部来自照抄 prompt 里的示例坐标,零 grounding 信息量。砍木头需要的是树干。

依据(今天实测,见 knowledge/conclusion_omni_pixel_control.md §3.2):
  - Omni 用 **1000x1000 归一化坐标**,指红点误差 2.2-5.4px;
  - 但直接问它"相机该转多少度"会答 -1(符号都错)。
  ⇒ 只让它做 grounding(它强的),连续标定留给快塔(它弱的)。

用法:
    bash tests/serve_omni_nvfp4.sh        # 慢塔
    Xvfb :99 -screen 0 1280x720x24 &
    DISPLAY=:99 /workspace/venv-mc/bin/python tests/probe_slow_tower_aim.py --trials 16
    DISPLAY=:99 /workspace/venv-mc/bin/python tests/probe_slow_tower_aim.py --trials 16 --arm random
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image

MODEL = "nemotron_3_nano_omni"
IMG_W, IMG_H, FOV_V = 640, 360, 70.0
TREE = ("log", "leaves", "wood")       # 命中"树"(宽松)
TRUNK = ("log", "wood")                # 命中"树干"(砍木头真正需要的)

SYSTEM = """\
You are the slow-system planner for a Minecraft agent. You see one 640x360 frame.
The agent's current long-horizon goal is: get wood, then a pickaxe, then iron.

Answer with ONE line of JSON and nothing else:

{"subgoal": "<short imperative, <=6 words>", "aim": [X, Y]}

  "aim" is the point the agent should put its crosshair on, in normalised image
  coordinates 0..1000, where (0,0) is the top-left corner and (1000,1000) is the
  bottom-right corner. The centre of the screen is (500,500).

"aim" must land ON the block the agent should break or walk to next in order to make
progress toward the goal. Do not aim at the sky.

Do not copy any coordinates from these instructions; read them off the image.
"""


def pixel_to_delta(nx: float, ny: float) -> tuple[float, float]:
    px, py = nx / 1000.0 * IMG_W, ny / 1000.0 * IMG_H
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    t = math.tan(math.radians(FOV_V) / 2.0)
    dyaw = math.degrees(math.atan(((px - cx) / cx) * t * (IMG_W / IMG_H)))
    dpitch = math.degrees(math.atan(((py - cy) / cy) * t))
    return dyaw, dpitch


def b64_png(arr: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def ray(full) -> str:
    rc = full.raycast_result
    if rc and rc.HasField("target_block"):
        return rc.target_block.translation_key.split(".")[-1]
    return "-"


def ask_slow_tower(client: OpenAI, rgb: np.ndarray) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": b64_png(rgb)}},
                      {"type": "text", "text": "Next subgoal and aim point."}]}],
        max_tokens=64, temperature=0.2, top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
    )
    dt = time.perf_counter() - t0
    txt = (r.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group()) if m else {}
        aim = [float(v) for v in d.get("aim", [500, 500])][:2]
        return {"subgoal": str(d.get("subgoal", ""))[:40], "aim": aim, "raw": txt[:90]}, dt
    except (ValueError, TypeError, AttributeError):
        return {"subgoal": "", "aim": [500.0, 500.0], "raw": txt[:90]}, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", default="42", help="默认世界 42 = 黑橡木森林")
    ap.add_argument("--arm", choices=["slow_tower", "random", "constant"], default="slow_tower",
                    help="constant = 恒用 prompt 示例里的 [430,560]。"
                         "慢塔若不显著高于它,说明 0.75 是抄示例抄出来的,不是真 grounding。")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--out", default="docs/results/slow_tower_aim.json")
    args = ap.parse_args()

    if "DISPLAY" not in os.environ:
        sys.exit("need DISPLAY")

    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.initial_environment_config import Difficulty, GameMode, WorldType
    from craftground.screen_encoding_modes import ScreenEncodingMode

    cfg = InitialEnvironmentConfig(
        image_width=IMG_W, image_height=IMG_H,
        gamemode=GameMode.SURVIVAL, difficulty=Difficulty.PEACEFUL,
        world_type=WorldType.DEFAULT, seed=args.seed,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        requires_surrounding_blocks=True, request_raycast=True,
    )
    cfg.set_allow_mob_spawn(False)
    cfg.freeze_time(True)
    cfg.freeze_weather(True)

    env = CraftGroundEnvironment(cfg, action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                                 port=args.port, find_free_port=True, verbose=False)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    # 两个独立 rng:rng_view 决定每轮看哪儿(三臂共用同一序列),rng_aim 只给 random 臂用。
    # 之前共用一个 rng => random 臂消费了抽样数,画面序列与慢塔臂错开,三臂不可比。
    rng = np.random.default_rng(0)          # rng_view
    rng_aim = np.random.default_rng(12345)

    def cam(dyaw=0.0, dpitch=0.0):
        a = no_op_v2(); a["camera_yaw"] = float(dyaw); a["camera_pitch"] = float(dpitch)
        return a

    obs, _ = env.reset()
    for _ in range(80):
        obs = env.step(no_op_v2())[0]

    log, hits, trunks, lat = [], 0, 0, []
    for t in range(args.trials):
        # 每轮换个朝向,别老看同一片画面;并把 pitch 归零
        obs = env.step(cam(dyaw=float(rng.uniform(-60, 60)),
                           dpitch=-float(obs["full"].pitch)))[0]
        for _ in range(3):
            obs = env.step(no_op_v2())[0]
        rgb = np.asarray(obs["rgb"], dtype="uint8")
        before = ray(obs["full"])

        if args.arm == "slow_tower":
            d, dt = ask_slow_tower(client, rgb)
            lat.append(dt)
        elif args.arm == "constant":
            d = {"subgoal": "<const=prompt example>", "aim": [430.0, 560.0], "raw": ""}
        else:
            d = {"subgoal": "<random>", "aim": [float(rng_aim.uniform(0, 1000)),
                                                float(rng_aim.uniform(0, 1000))], "raw": ""}
        dyaw, dpitch = pixel_to_delta(*np.clip(d["aim"], 0, 1000))
        obs = env.step(cam(dyaw, dpitch))[0]
        for _ in range(2):
            obs = env.step(no_op_v2())[0]
        after = ray(obs["full"])
        hit = any(k in after for k in TREE)
        trunk = any(k in after for k in TRUNK)
        hits += hit
        trunks += trunk
        log.append({"trial": t, **d, "dyaw": round(dyaw, 1), "dpitch": round(dpitch, 1),
                    "ray_before": before, "ray_after": after,
                    "tree_hit": hit, "trunk_hit": trunk})
        print(f"[{t + 1:2d}/{args.trials}] aim={d['aim']} -> dyaw={dyaw:+6.1f} dpitch={dpitch:+5.1f} "
              f"| ray {before} -> {after} {'✓TREE' if hit else ''}  {d['subgoal']!r}", flush=True)

    rate = hits / args.trials
    trunk_rate = trunks / args.trials
    aims = [tuple(x["aim"]) for x in log]
    summary = {"arm": args.arm, "trials": args.trials, "world_seed": args.seed,
               "tree_hit_rate": round(rate, 3),
               "trunk_hit_rate": round(trunk_rate, 3),
               "distinct_aims": len(set(aims)),
               "copied_prompt_example": sum(1 for a in aims if a == (430.0, 560.0)),
               "mean_latency_s": round(float(np.mean(lat)), 3) if lat else None,
               "steps": log}
    out = Path(args.out).with_name(f"slow_tower_aim_{args.arm}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\narm={args.arm}  **trunk_hit_rate = {trunk_rate:.3f}** ({trunks}/{args.trials})  "
          f"[宽松] tree_hit_rate = {rate:.3f} ({hits}/{args.trials})  "
          f"distinct_aims = {len(set(aims))}  copied_example = {summary['copied_prompt_example']}")
    print(f"wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
