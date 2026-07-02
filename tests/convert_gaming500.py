#!/usr/bin/env python3
"""下载 HF markov-ai/gaming-500-hours 任意游戏子集并转换为 train/minecraft 数据契约。

对外接口:命令行脚本(main);无库接口。

多游戏预训练动机:OS 级输入契约跨游戏通用(WASD/空格/Shift/Ctrl/数字键 + 鼠标
dx,dy + 左右键在 FPS/生存/开放世界游戏中语义一致),22 维动作向量全数据集共用;
--games 逗号分隔多个游戏目录,游戏进程按会话自动识别(接收输入最多的非系统进程)。

数据源:每会话 = clip.mp4(1080p/30fps 桌面录屏)+ frame_events.json(逐帧 OS 输入事件)
+ metadata.json(标题/描述)。与 VPT/BASALT(tests/download_vpt_data.py)的差异:
  - 动作是 OS 级事件(key isDown 按下/抬起、click、mouse_move 绝对坐标),需要
    键状态机 + 帧间坐标差分重建为逐帧动作 dict;
  - 录屏含桌面/浏览器片段,须按事件 bundleId(javaw.exe)过滤游戏内帧,
    只导出连续游戏内片段(--min-frames 下限);
  - 鼠标绝对坐标差分已实测可用(游戏内 34k 事件仅 0.1% 钉中心,|dx| p99=75,
    无回中大跳变);>--warp-px 的跳变按回中丢弃。

转换后契约(vpt_dataset._action_vec 消费,与 download_vpt_data 一致):
    {"mouse": {"dx","dy"}, "keyboard": {"key_w":1,...}, "gui": bool}
    首行额外带 "task"(metadata 标题)。视频重编码为 360p/30fps(降低训练期
    CPU 解码开销;高分辨率实验可用 --scale-h 换档重转)。

使用方法:
    python tests/convert_gaming500.py --n 2 --out runs/data/gaming500_mc      # 验证
    python tests/convert_gaming500.py --n 75 --out runs/data/gaming500_mc     # 全量
"""
import argparse
import json
import os
import subprocess

import numpy as np
import requests

BASE = "https://huggingface.co/datasets/markov-ai/gaming-500-hours/resolve/main"
API = "https://huggingface.co/api/datasets/markov-ai/gaming-500-hours"
# 已知游戏进程;不在表内的游戏由 detect_game_bundle 按"接收输入事件最多的非系统进程"自动识别
KNOWN_GAME_BUNDLES = {"javaw.exe", "java.exe", "minecraft.exe"}
NON_GAME_BUNDLES = {"chrome.exe", "msedge.exe", "firefox.exe", "explorer.exe",
                    "steam.exe", "steamwebhelper.exe", "discord.exe", "unknown", None}

# Windows VK keyCode → 契约键名(train/minecraft/vpt_dataset.VPT_KEYS)
VK_MAP = {
    87: "key_w", 65: "key_a", 83: "key_s", 68: "key_d",
    32: "key_space", 16: "key_sneak", 17: "key_sprint",
    81: "key_drop", 69: "key_inventory",
    **{48 + i: f"key_hotbar.{i}" for i in range(1, 10)},
}
BTN_MAP = {"left": "key_attack", "right": "key_use"}


def parse_args():
    p = argparse.ArgumentParser(description="gaming-500-hours minecraft 子集转换")
    p.add_argument("--games", default="minecraft",
                   help="逗号分隔的游戏目录名(HF 仓库顶层目录),如 'minecraft,valorant,palworld'")
    p.add_argument("--n", type=int, default=2, help="转换的会话数")
    p.add_argument("--out", default="runs/data/gaming500_mc")
    p.add_argument("--raw", default="runs/data/gaming500_raw", help="原始下载缓存目录")
    p.add_argument("--min-frames", type=int, default=900, help="游戏内片段帧数下限(30fps,900=30s)")
    p.add_argument("--warp-px", type=float, default=200.0, help="单事件位移超此值视为回中丢弃")
    p.add_argument("--scale-h", type=int, default=360, help="重编码目标高(宽等比,-2 对齐)")
    p.add_argument("--crop-bottom", type=float, default=0.05,
                   help="裁掉底部高度比例(无边框窗口录屏常带 Windows 任务栏,实测样例约 4.5%%)")
    p.add_argument("--purge-raw", action="store_true",
                   help="每会话转换完成后删除原始 clip 缓存(全量语料时必开,原片约 1-2GB/会话)")
    p.add_argument("--match", default=None,
                   help="仅转换 metadata 标题/描述匹配此正则(不区分大小写)的会话,如 'surviv|tutorial'")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def http_json(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def download(url, path, desc):
    if os.path.exists(path):
        return
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, path)
    print(f"   ↓ {desc}: {os.path.getsize(path) / 1e6:.0f}MB", flush=True)


def detect_game_bundle(fe_path):
    """扫描输入事件(key/click/mouse_move)的 bundleId 频次,取非系统进程的众数为游戏进程。

    Returns:
        game_bundles: set,该会话的游戏进程名集合(已知表命中则并入)。
    """
    import collections
    freq = collections.Counter()
    with open(fe_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                break
            for e in fr.get("events", []):
                if e.get("type") in ("key", "click", "mouse_move", "drag"):
                    b = e.get("bundleId")
                    if b not in NON_GAME_BUNDLES:
                        freq[b] += 1
    hits = {b for b in freq if b in KNOWN_GAME_BUNDLES}
    if hits:
        return hits
    return {freq.most_common(1)[0][0]} if freq else set()


def frame_actions(fe_path, warp_px, game_bundles):
    """frame_events.json → (per_frame_action 列表, in_game 布尔列表)。

    键/鼠标键走 isDown 状态机;dx/dy 为帧内游戏事件坐标差分和(回中跳变丢弃);
    in_game[t] 由 active_app 状态机维护(输入事件的 bundleId 亦可翻转状态)。
    """
    held, acts, in_game = {}, [], []
    game_active, prev_xy = False, None
    with open(fe_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                break
            dx = dy = 0.0
            for e in fr.get("events", []):
                et, bid = e.get("type"), e.get("bundleId")
                if et == "active_app":
                    game_active = bid in game_bundles
                    if not game_active:
                        held.clear()
                        prev_xy = None
                    continue
                if bid not in game_bundles:
                    continue
                game_active = True
                if et == "key":
                    name = VK_MAP.get(e.get("keyCode"))
                    if name:
                        held[name] = 1 if e.get("isDown") else 0
                elif et == "click":
                    name = BTN_MAP.get(e.get("button"))
                    if name:
                        held[name] = 1 if e.get("isDown") else 0
                elif et in ("mouse_move", "drag"):
                    xy = (e.get("x"), e.get("y"))
                    if None not in xy:
                        if prev_xy is not None:
                            ddx, ddy = xy[0] - prev_xy[0], xy[1] - prev_xy[1]
                            if abs(ddx) <= warp_px and abs(ddy) <= warp_px:
                                dx += ddx
                                dy += ddy
                        prev_xy = xy
            kb = {k: v for k, v in held.items() if v}
            acts.append({"mouse": {"dx": dx, "dy": dy}, "keyboard": kb, "gui": False})
            in_game.append(game_active)
    return acts, in_game


def segments(in_game, min_frames):
    """连续 in_game 帧区间 [(start, end)),长度 ≥ min_frames。"""
    out, s = [], None
    for i, g in enumerate(in_game + [False]):
        if g and s is None:
            s = i
        elif not g and s is not None:
            if i - s >= min_frames:
                out.append((s, i))
            s = None
    return out


def cut_video(src, dst, start_f, n_frames, fps, scale_h, crop_bottom):
    """帧精确切段 + 底部裁剪 + 等比缩放重编码(-ss 放 -i 后逐帧精确)。"""
    vf = f"crop=iw:ih*{1 - crop_bottom:.3f}:0:0,scale=-2:{scale_h}" \
        if crop_bottom > 0 else f"scale=-2:{scale_h}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-ss", f"{start_f / fps:.3f}", "-frames:v", str(n_frames),
           "-vf", vf, "-r", str(fps),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", dst]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.raw, exist_ok=True)
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    all_sessions = []
    for game in games:
        tree = http_json(f"{API}/tree/main/{game}")
        all_sessions += sorted(x["path"] for x in tree if x["type"] == "directory")
    if args.match:
        import re
        pat = re.compile(args.match, re.I)
        kept = []
        for sess in all_sessions:
            try:
                m = http_json(f"{BASE}/{sess}/metadata.json")
            except requests.RequestException:
                continue
            if pat.search(m.get("title", "") + " " + m.get("description", "")):
                kept.append(sess)
        print(f"   筛选 --match '{args.match}': {len(all_sessions)} → {len(kept)} 个会话", flush=True)
        all_sessions = kept
    sessions = all_sessions[: args.n]
    print(f"📥 {','.join(games)} 会话共 {len(all_sessions)},转换前 {len(sessions)} 个 → {args.out}",
          flush=True)

    adx, ady, n_seg, n_frames_total = [], [], 0, 0
    for si, sess in enumerate(sessions):
        sid = sess.split("/")[0] + "_" + sess.split("/")[-1][:8]
        print(f"[{si + 1}/{len(sessions)}] {sid}", flush=True)
        fe = os.path.join(args.raw, f"{sid}_frame_events.json")
        mp4 = os.path.join(args.raw, f"{sid}_clip.mp4")
        try:
            meta = http_json(f"{BASE}/{sess}/metadata.json")
            download(f"{BASE}/{sess}/frame_events.json", fe, "frame_events")
            bundles = detect_game_bundle(fe)
            if not bundles:
                print("   ⤫ 未识别到游戏进程,跳过", flush=True)
                continue
            acts, in_game = frame_actions(fe, args.warp_px, bundles)
            segs = segments(in_game, args.min_frames)
            if not segs:
                print("   ⤫ 无足够长的游戏内片段,跳过", flush=True)
                continue
            download(f"{BASE}/{sess}/clip.mp4", mp4, "clip")
        except (requests.RequestException, OSError) as ex:
            print(f"   ⤫ 下载失败: {ex}", flush=True)
            continue
        task = meta.get("title", "minecraft gameplay")
        for gi, (s, e) in enumerate(segs):
            stem = os.path.join(args.out, f"{sid}_{gi:02d}")
            try:
                cut_video(mp4, stem + ".mp4", s, e - s, 30, args.scale_h, args.crop_bottom)
            except subprocess.CalledProcessError as ex:
                print(f"   ⤫ ffmpeg 失败 seg{gi}: {ex}", flush=True)
                continue
            with open(stem + ".jsonl", "w", encoding="utf-8") as f:
                for t in range(s, e):
                    a = dict(acts[t])
                    if t == s:
                        a = {"task": task, **a}
                    f.write(json.dumps(a) + "\n")
                    adx.append(abs(a["mouse"]["dx"]))
                    ady.append(abs(a["mouse"]["dy"]))
            n_seg += 1
            n_frames_total += e - s
            print(f"   ✓ seg{gi}: 帧 {s}-{e}({(e - s) / 30:.0f}s)", flush=True)
        if args.purge_raw:
            for pth in (mp4, fe):
                if os.path.exists(pth):
                    os.remove(pth)

    if adx:
        adx, ady = np.array(adx), np.array(ady)
        print(f"\n📐 |dx| p50/p90/p95/p99 = "
              f"{np.percentile(adx, [50, 90, 95, 99]).round(1).tolist()}")
        print(f"   |dy| p50/p90/p95/p99 = "
              f"{np.percentile(ady, [50, 90, 95, 99]).round(1).tolist()}")
        print(f"   非零 |dx| 占比 {(adx > 0).mean():.1%}")
        print(f"   👉 建议 --camera_scale {max(np.percentile(adx, 95), 1.0):.0f}")
    print(f"\n✅ 完成:{n_seg} 段 / {n_frames_total / 30 / 60:.1f} 分钟游戏内数据 → {args.out}")


if __name__ == "__main__":
    main()
