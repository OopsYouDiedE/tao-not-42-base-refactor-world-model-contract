#!/usr/bin/env python3
"""下载 OpenAI VPT/BASALT 承包商数据并转换为 train/minecraft 数据契约。

数据源:openaipublic.blob.core.windows.net/minecraft-rl(公开,无需鉴权)。
索引 json 给出 basedir+relpaths(mp4);同名 .jsonl 是逐帧动作记录(20Hz,与帧 1:1)。

原始 jsonl 单帧 schema(实测 find-cave):
    {"mouse": {"dx","dy","buttons":[0=左键,1=右键],...},
     "keyboard": {"keys": ["key.keyboard.w",...]}, "isGuiOpen": bool, ...}
转换后契约(vpt_dataset._action_vec 消费):
    {"mouse": {"dx","dy"}, "keyboard": {"key_w":1,...}, "gui": bool}
    首行额外带 "task"(任务文本)。

转换后报告 |dx|/|dy| 分位数,给出建议 camera_scale(取 p95;归一化后大部分
相机运动落在 [-1,1],重尾截到边缘 bin,与 vpt_dataset 注释的标定原则一致)。

使用方法:
    python -m tests.download_vpt_data --index find-cave-Jul-28 \
        --n 8 --out runs/data/vpt_findcave --task "find a cave"
"""
import argparse
import json
import os
import random

import numpy as np
import requests

from train.minecraft.vpt_action import _KEYMAP

SNAPSHOT_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/{}.json"


def parse_args():
    p = argparse.ArgumentParser(description="下载并转换 VPT/BASALT 数据")
    p.add_argument("--index", default="find-cave-Jul-28",
                   help="索引名(find-cave-Jul-28 / make-waterfall-Jul-28 / all_6xx_Jun_29 ...)")
    p.add_argument("--n", type=int, default=8, help="下载的 clip 数(mp4+jsonl 对)")
    p.add_argument("--out", default="runs/data/vpt_findcave")
    p.add_argument("--task", default="find a cave", help="写入契约 jsonl 首行的任务文本")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-mp4-mb", type=float, default=5.0, help="小于此体积的 mp4 视为坏段,换下一个")
    p.add_argument("--min-frames", type=int, default=2000, help="jsonl 行数下限(20Hz,2000≈100s)")
    return p.parse_args()


def convert_jsonl(raw_path, out_path, task):
    """原始承包商 jsonl → 契约 jsonl。返回该 clip 的 (|dx| 列表, |dy| 列表, 帧数)。"""
    adx, ady, n = [], [], 0
    with open(raw_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            try:
                d = json.loads(line) if line else None
            except ValueError:
                d = None
            if not isinstance(d, dict):        # null/坏行 → no-op 帧(保持与视频帧对齐)
                d = {}
            mouse = d.get("mouse") or {}
            dx = float(mouse.get("dx") or 0.0)
            dy = float(mouse.get("dy") or 0.0)
            kb = {}
            for k in (d.get("keyboard") or {}).get("keys") or []:
                name = _KEYMAP.get(k)
                if name:
                    kb[name] = 1
            for b in mouse.get("buttons") or []:
                if b == 0:
                    kb["key_attack"] = 1
                elif b == 1:
                    kb["key_use"] = 1
            rec = {"mouse": {"dx": dx, "dy": dy}, "keyboard": kb,
                   "gui": bool(d.get("isGuiOpen"))}
            if n == 0:
                rec["task"] = task
            fout.write(json.dumps(rec) + "\n")
            adx.append(abs(dx))
            ady.append(abs(dy))
            n += 1
    return adx, ady, n


def fetch(url, path, chunk=1 << 20):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                f.write(c)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"📥 拉取索引 {args.index} ...")
    idx = requests.get(SNAPSHOT_URL.format(args.index), timeout=60).json()
    basedir, relpaths = idx["basedir"], list(idx["relpaths"])
    rng.shuffle(relpaths)
    print(f"   共 {len(relpaths)} 段,目标下载 {args.n} 段 → {args.out}")

    got, all_dx, all_dy = 0, [], []
    for rel in relpaths:
        if got >= args.n:
            break
        stem = os.path.basename(rel)[:-4]
        mp4_path = os.path.join(args.out, stem + ".mp4")
        jsonl_path = os.path.join(args.out, stem + ".jsonl")
        if os.path.exists(mp4_path) and os.path.exists(jsonl_path):
            got += 1
            continue
        raw_jsonl = jsonl_path + ".raw"
        try:
            fetch(basedir + rel[:-4] + ".jsonl", raw_jsonl)
            adx, ady, n_frames = convert_jsonl(raw_jsonl, jsonl_path, args.task)
            if n_frames < args.min_frames:
                raise RuntimeError(f"太短({n_frames} 帧)")
            fetch(basedir + rel, mp4_path)
            if os.path.getsize(mp4_path) < args.min_mp4_mb * 1e6:
                raise RuntimeError("mp4 太小,疑似坏段")
        except Exception as ex:                 # 坏段/网络失败:清理残留,换下一个
            print(f"   ⤫ 跳过 {stem}: {ex}")
            for pth in (mp4_path, jsonl_path):
                if os.path.exists(pth):
                    os.remove(pth)
            continue
        finally:
            if os.path.exists(raw_jsonl):
                os.remove(raw_jsonl)
        got += 1
        all_dx += adx
        all_dy += ady
        print(f"   ✓ [{got}/{args.n}] {stem}: {n_frames} 帧, "
              f"{os.path.getsize(mp4_path)/1e6:.0f}MB")

    if all_dx:
        dx = np.asarray(all_dx)
        dy = np.asarray(all_dy)
        nz = dx[dx > 0]
        print("\n📐 相机幅度统计(本次新转换的帧):")
        print(f"   |dx| p50/p90/p95/p99 = {np.percentile(dx,50):.1f}/"
              f"{np.percentile(dx,90):.1f}/{np.percentile(dx,95):.1f}/{np.percentile(dx,99):.1f}")
        print(f"   |dy| p50/p90/p95/p99 = {np.percentile(dy,50):.1f}/"
              f"{np.percentile(dy,90):.1f}/{np.percentile(dy,95):.1f}/{np.percentile(dy,99):.1f}")
        print(f"   非零 |dx| 占比 {len(nz)/max(len(dx),1):.1%}")
        scale = max(1.0, round(float(np.percentile(np.concatenate([dx, dy]), 95)), 1))
        print(f"   👉 建议 --camera_scale {scale}(p95;写入训练 CLI)")
    print(f"\n✅ 完成:{got} 段可用于 train/minecraft(--data_dir {args.out})")


if __name__ == "__main__":
    main()
