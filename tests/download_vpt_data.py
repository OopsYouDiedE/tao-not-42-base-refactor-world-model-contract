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
    p.add_argument("--roll", default="", help="滚动无限池:逗号分隔索引列表,流式下完全部索引;"
                   "配合 --max-pool-gb 磁盘滑动窗口与 <out>_seen.txt 去重(已淘汰段不再回灌)。"
                   "训练端 VPTStreamDataset 滚动目录模式(split=None)自动适应目录增删")
    p.add_argument("--max-pool-gb", type=float, default=60.0, help="池目录体积上限,超出按 mtime 淘汰最旧段")
    p.add_argument("--min-free-gb", type=float, default=25.0, help="磁盘剩余低于此值即暂停下载(安全阀)")
    return p.parse_args()


def _pool_gb(out_dir):
    return sum(os.path.getsize(os.path.join(out_dir, f))
               for f in os.listdir(out_dir)) / 1e9


def prune_pool(out_dir, max_gb):
    """磁盘滑动窗口:按 mtime 淘汰最旧的 (mp4,jsonl) 对,直到池体积 ≤ max_gb。

    训练端 VPTStreamDataset(split=None) 对被删文件静默跳过重试,删除是安全的。
    """
    pairs = sorted((os.path.getmtime(os.path.join(out_dir, f)), f)
                   for f in os.listdir(out_dir) if f.endswith(".mp4"))
    while _pool_gb(out_dir) > max_gb and pairs:
        _, f = pairs.pop(0)
        for pth in (os.path.join(out_dir, f), os.path.join(out_dir, f[:-4] + ".jsonl")):
            if os.path.exists(pth):
                os.remove(pth)
        print(f"   ♻ 淘汰最旧段 {f[:-4]}", flush=True)


def roll_pool(args):
    """滚动无限池:流式遍历全部索引,去重下载 + 磁盘滑动窗口,直到索引耗尽。"""
    import shutil
    rng = random.Random(args.seed)
    seen_path = args.out.rstrip("/") + "_seen.txt"
    seen = set()
    if os.path.exists(seen_path):
        seen = set(open(seen_path).read().split())
    got_total = 0
    for index_name in [s.strip() for s in args.roll.split(",") if s.strip()]:
        try:
            idx = requests.get(SNAPSHOT_URL.format(index_name), timeout=60).json()
        except Exception as ex:  # noqa: BLE001  索引不存在/网络失败:跳过该索引
            print(f"⤫ 索引 {index_name} 拉取失败: {ex}", flush=True)
            continue
        basedir, relpaths = idx["basedir"], list(idx["relpaths"])
        rng.shuffle(relpaths)
        print(f"📥 [{index_name}] 共 {len(relpaths)} 段", flush=True)
        for rel in relpaths:
            stem = os.path.basename(rel)[:-4]
            if stem in seen:
                continue
            if shutil.disk_usage("/").free / 1e9 < args.min_free_gb:
                print("⛔ 磁盘剩余低于安全阀,停止", flush=True)
                return
            mp4_path = os.path.join(args.out, stem + ".mp4")
            jsonl_path = os.path.join(args.out, stem + ".jsonl")
            raw_jsonl = jsonl_path + ".raw"
            try:
                fetch(basedir + rel[:-4] + ".jsonl", raw_jsonl)
                _, _, n_frames = convert_jsonl(raw_jsonl, jsonl_path, args.task)
                if n_frames < args.min_frames:
                    raise RuntimeError(f"太短({n_frames} 帧)")
                fetch(basedir + rel, mp4_path)
                if os.path.getsize(mp4_path) < args.min_mp4_mb * 1e6:
                    raise RuntimeError("mp4 太小,疑似坏段")
            except Exception as ex:  # noqa: BLE001  坏段:清残留换下一个
                print(f"   ⤫ 跳过 {stem}: {ex}", flush=True)
                for pth in (mp4_path, mp4_path + ".part", jsonl_path):
                    if os.path.exists(pth):
                        os.remove(pth)
                continue
            finally:
                if os.path.exists(raw_jsonl):
                    os.remove(raw_jsonl)
            seen.add(stem)
            with open(seen_path, "a") as f:
                f.write(stem + "\n")
            got_total += 1
            print(f"   ✓ [{got_total}] {stem}: {n_frames} 帧", flush=True)
            prune_pool(args.out, args.max_pool_gb)
    print(f"✅ 全部索引耗尽,累计 {got_total} 段", flush=True)


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
    """先写 .part 再原子 rename:滚动目录消费方(VPTStreamDataset)按 .mp4 后缀配对,
    半截文件不再以最终名可见(读端撞半截 mp4 只会打 ffmpeg 警告并换段,但白耗一次解码)。"""
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                f.write(c)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.roll:
        roll_pool(args)
        return
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
            for pth in (mp4_path, mp4_path + ".part", jsonl_path):
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
