# -*- coding: utf-8 -*-
"""慢塔候选同口径 A/B 探针(5090):契约合规四项 + 延迟,考题一次采集、逐塔重放。

背景与选型见 knowledge/README.md §3、§6.2。本脚本把考题
固化成文件,保证两塔看到逐字节相同的 (frame, STATE) 输入。

用法:
  # ① 采集考题(需 CraftGround + 一台在跑的慢塔服务做 goal_log 递推;跑一条真实 episode)
  python tests/probe_slow_tower_ab.py --collect --base-url http://127.0.0.1:8000/v1 \
      --serve-model nemotron_3_nano_omni --init-from runs/checkpoints/bc_vpt4/best.pt
  # ② 对某塔重放打分(对每个候选各跑一次,起服/停服在外部做)
  python tests/probe_slow_tower_ab.py --exam --base-url http://127.0.0.1:8000/v1 \
      --serve-model nemotron_3_nano_omni --tag omni_nvfp4
  # 结果累积写 docs/results/slow_tower_ab_5090.json

评分口径(与 L4 2026-07-10 探针四项一致):
  json_ok   单行 JSON 可解析(parse_slow_reply 的 parsed)
  one_line  整个回复恰为一行(去首尾空白后无换行)
  aim_in    原始 aim(裁剪前)两个分量都在 [0,1000]
  dec_ok    decision 命中词表 {continue, switch, replan}
延迟为逐题墙钟(含网络),报 p50/p95;并记 nvidia-smi 服务显存。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.craftground.grpo_pixel import (DECISIONS, SLOW_SYSTEM,  # noqa: E402
                                          DEFAULT_TASK, SlowTower,
                                          parse_slow_reply)

EXAM_DIR = Path("runs/probe_slow_ab/exam")
OUT_JSON = Path("docs/results/slow_tower_ab_5090.json")


# ────────────────────────────────────────────── ① 考题采集(真实循环)

class _Recorder(SlowTower):
    """SlowTower 透传,同时把每次调用的 (rgb, state) 固化为考题文件。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n = 0

    def __call__(self, rgb, state=""):
        EXAM_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(EXAM_DIR / f"item_{self.n:03d}.png")
        (EXAM_DIR / f"item_{self.n:03d}.state.txt").write_text(state or "STATE t=0 (fresh start)")
        self.n += 1
        return super().__call__(rgb, state)


def collect(args) -> None:
    import torch
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.initial_environment_config import (Difficulty, GameMode,
                                                        WorldType)
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from sentence_transformers import SentenceTransformer

    from net.pixel_tower import PixelTowerConfig, build_pixel_tower
    from train.craftground.grpo_pixel import rollout

    from train.craftground.action_contract import CAM_BINS as _CB, V2_KEYS as _VK
    from train.craftground.grpo_pixel import IMG_HW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    cfg = PixelTowerConfig(img_hw=IMG_HW, goal_dim=384 + 2, n_keys=len(_VK),
                           camera_bins=_CB)                # 与 grpo_pixel v1 同契约
    tower = build_pixel_tower(cfg).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=True)
        tower.load_state_dict(ck["tower"])
        print(f"init from {args.init_from} (bc_step={ck.get('step')})", flush=True)
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    encode_text = lambda xs: st.encode(xs, normalize_embeddings=True)  # noqa: E731
    slow = _Recorder(args.base_url, encode_text, device, task=DEFAULT_TASK,
                     model=args.serve_model)

    env, wseed, has_tree = None, "", False
    for att in range(args.seed_tries):        # 与 grpo_pixel 同款有树 seed 筛选
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
                                     port=args.port, find_free_port=True, verbose=False)
        obs0, _ = env.reset()
        for _ in range(60):
            obs0 = env.step(no_op_v2())[0]
        has_tree = any("leaves" in h.block_name or "log" in h.block_name
                       for h in obs0["full"].height_info)
        if has_tree or att == args.seed_tries - 1:
            break
        print(f"seed={wseed} 出生点无树,换 seed", flush=True)
        env.close()
    print(f"world_seed={wseed} has_tree={has_tree}", flush=True)

    r = rollout(env, tower, slow, no_op_v2, rng, args.ticks, device, temp=1.0)
    env.close()
    meta = dict(world_seed=wseed, has_tree=has_tree, ticks=args.ticks,
                n_items=slow.n, init_from=args.init_from,
                collect_model=args.serve_model, collect_fails=slow.fails,
                inv_events=sorted(r.get("inv_events", [])) if isinstance(r, dict) else None)
    (EXAM_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"考题 {slow.n} 题 → {EXAM_DIR}", flush=True)


# ────────────────────────────────────────────── ② 重放打分

def raw_call(client, model: str, state: str, img_b64: str):
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SLOW_SYSTEM},
                  {"role": "user", "content": [
                      {"type": "text", "text": f"TASK: {DEFAULT_TASK}\n" + state},
                      {"type": "image_url", "image_url": {"url": img_b64}},
                      {"type": "text", "text": "Next subgoal."}]}],
        max_tokens=96, temperature=0.2, top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
    )
    dt = time.perf_counter() - t0
    return (r.choices[0].message.content or "").strip(), dt


def score_one(txt: str) -> dict:
    rep = parse_slow_reply(txt)
    aim_in = False
    try:
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group())
        raw_aim = [float(v) for v in list(d.get("aim", []))[:2]]
        aim_in = len(raw_aim) == 2 and all(0 <= v <= 1000 for v in raw_aim)
        dec_ok = str(d.get("decision", "")).strip().lower() in DECISIONS
    except Exception:  # noqa: BLE001
        dec_ok = False
    return dict(json_ok=rep["parsed"], one_line="\n" not in txt.strip(),
                aim_in=aim_in, dec_ok=dec_ok, subgoal=rep["subgoal"],
                aim=rep["aim"], decision=rep["decision"])


def gpu_mem_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def exam(args) -> None:
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    items = sorted(EXAM_DIR.glob("item_*.png"))
    assert items, f"{EXAM_DIR} 无考题,先 --collect"
    rows, lat = [], []
    for p in items:
        state = (p.parent / (p.stem + ".state.txt")).read_text()
        b = io.BytesIO()
        Image.open(p).convert("RGB").save(b, format="JPEG", quality=80)
        b64 = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
        try:
            txt, dt = raw_call(client, args.serve_model, state, b64)
        except Exception as e:  # noqa: BLE001
            txt, dt = f"<CALL_FAIL {type(e).__name__}>", float("nan")
        s = score_one(txt)
        s.update(item=p.stem, latency_s=dt, raw=txt[:400])
        rows.append(s)
        if not np.isnan(dt):
            lat.append(dt)
        print(f"{p.stem} {dt:5.2f}s json={s['json_ok']} line={s['one_line']} "
              f"aim={s['aim_in']} dec={s['dec_ok']} '{s['subgoal']}'", flush=True)

    n = len(rows)
    agg = dict(tag=args.tag, serve_model=args.serve_model, n=n,
               json_ok=sum(r["json_ok"] for r in rows) / n,
               one_line=sum(r["one_line"] for r in rows) / n,
               aim_in=sum(r["aim_in"] for r in rows) / n,
               dec_ok=sum(r["dec_ok"] for r in rows) / n,
               call_fail=sum(r["raw"].startswith("<CALL_FAIL") for r in rows),
               lat_p50=float(np.percentile(lat, 50)) if lat else None,
               lat_p95=float(np.percentile(lat, 95)) if lat else None,
               gpu_mem_mib=gpu_mem_mib(), ts=time.strftime("%F %T"))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    db = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    db[args.tag] = dict(agg=agg, rows=rows)
    OUT_JSON.write_text(json.dumps(db, ensure_ascii=False, indent=1))
    print(json.dumps(agg, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--exam", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--serve-model", default="nemotron_3_nano_omni")
    ap.add_argument("--tag", default="omni_nvfp4")
    ap.add_argument("--init-from", default="runs/checkpoints/bc_vpt4/best.pt")
    ap.add_argument("--ticks", type=int, default=640)   # 640/SLOW_EVERY=32 题
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seed-tries", type=int, default=8)
    ap.add_argument("--port", type=int, default=14100)
    args = ap.parse_args()
    if args.collect:
        collect(args)
    if args.exam:
        exam(args)


if __name__ == "__main__":
    main()
