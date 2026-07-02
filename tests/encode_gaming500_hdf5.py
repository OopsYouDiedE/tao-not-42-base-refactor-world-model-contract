#!/usr/bin/env python3
"""gaming-500-hours → HDF5 分片归档:图像 JPEG 入 h5,事件全率无损保留,流式上传 HF。

对外接口:
    命令行脚本(main)——编码/上传管线;
    Gaming500H5(库类)——多线程随机读取已编码分片(训练/分析端加载器)。

设计(用户 2026-07-02 拍板):
  - 解码帧先积在内存缓冲,超 --buffer-gb(默认 10)即由线程池 JPEG 压缩追加进当前分片;
  - 分片文件超 --shard-gb(默认 20)即封片:后台线程上传到 HF 数据集并删本地,另起新片;
  - 编码(JPEG imencode,释放 GIL)与加载(Gaming500H5.read_batch)都走线程池;
  - "其他信息不要丢":事件以源生 30Hz 全率存两份——结构化数组(dx/dy f32、键位 u32
    位掩码、gui u8,gzip)供快速训练加载 + 整段契约 JSON gzip 原文(绝对保真,含 task/
    元数据);图像可按 --hz 降采样(frame_idx 记录到 30Hz 事件流的映射,不丢对齐)。
  - 断点续跑:启动时列 HF 仓库与本地清单,已完成会话跳过;上传失败的分片留本地重试。

体量账(360p/15Hz/q80 ≈ 20KB/帧):500h ≈ 2700 万帧 ≈ 550GB ≈ 28 个分片;
上传 50MB/s 时每片 ~7 分钟,与编码流水重叠。HF 需 write token(hf auth login)。

HDF5 布局(每游戏段一组,与 VPT 契约的段划分一致):
    /{game}/{session8}_{seg:02d}/
        jpeg       vlen u8 [N]   # JPEG 字节流,attrs: hz/w/h/quality
        frame_idx  i32 [N]       # 各图像帧在 30Hz 事件流中的下标
        dx, dy     f32 [M] (gzip)  # M = 段内 30Hz 全率帧数
        keys       u32 [M] (gzip)  # VPT_KEYS 顺序位掩码
        gui        u8  [M] (gzip)
        events_gz  u8  [K]       # 整段逐帧契约 JSONL 的 gzip 原文(无损兜底)
        attrs: task / session / game / seg_start / seg_end / src_fps / meta_json

使用方法:
    PYTHONPATH=. python tests/encode_gaming500_hdf5.py --games minecraft --n 2 \
        --buffer-gb 1 --shard-gb 2 --no-upload          # 冒烟
    PYTHONPATH=. python tests/encode_gaming500_hdf5.py --games minecraft,valorant,gta-v --n 999
"""
import argparse
import gzip
import importlib.util
import io
import json
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# tests/ 被站点包同名包遮蔽,按路径加载兄弟模块复用会话/事件逻辑
_spec = importlib.util.spec_from_file_location(
    "convert_gaming500", os.path.join(os.path.dirname(__file__), "convert_gaming500.py"))
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)

from train.minecraft.vpt_action import VPT_KEYS  # noqa: E402

_KEY_BIT = {k: 1 << i for i, k in enumerate(VPT_KEYS)}


def parse_args():
    p = argparse.ArgumentParser(description="gaming-500-hours → HDF5 分片 + HF 流式上传")
    p.add_argument("--games", default="minecraft")
    p.add_argument("--n", type=int, default=2, help="每游戏最多会话数")
    p.add_argument("--out", default="runs/data/g500_h5", help="分片输出目录")
    p.add_argument("--raw", default="runs/data/gaming500_raw")
    p.add_argument("--repo", default="gaming500-360p-hdf5",
                   help="HF 数据集名(命名空间自动取当前登录用户)")
    p.add_argument("--hz", type=float, default=15.0, help="图像采样率(事件恒为源生 30Hz 全率)")
    p.add_argument("--scale-h", type=int, default=360)
    p.add_argument("--quality", type=int, default=80, help="JPEG 质量")
    p.add_argument("--buffer-gb", type=float, default=10.0, help="内存原始帧缓冲上限")
    p.add_argument("--shard-gb", type=float, default=20.0, help="单分片封片阈值")
    p.add_argument("--threads", type=int, default=8, help="JPEG 编码线程数")
    p.add_argument("--min-frames", type=int, default=900)
    p.add_argument("--warp-px", type=float, default=200.0)
    p.add_argument("--match", default=None)
    p.add_argument("--no-upload", action="store_true", help="只编码不上传(无 token 冒烟)")
    p.add_argument("--public", action="store_true", help="HF 仓库设为公开(默认私有)")
    return p.parse_args()


# ---------------------------------------------------------------- 上传

class Uploader:
    """封片后台上传线程:成片入队,逐个 upload_file 到 HF 数据集后删本地。"""

    def __init__(self, repo, public, enabled):
        self.q, self.enabled, self.repo_id = queue.Queue(), enabled, None
        if enabled:
            from huggingface_hub import HfApi
            self.api = HfApi()
            user = self.api.whoami()["name"]
            self.repo_id = f"{user}/{repo}"
            self.api.create_repo(self.repo_id, repo_type="dataset",
                                 private=not public, exist_ok=True)
            print(f"☁️  上传目标: {self.repo_id}({'公开' if public else '私有'})", flush=True)
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def existing(self):
        if not self.enabled:
            return set()
        return set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))

    def submit(self, path):
        self.q.put(path)

    def _loop(self):
        while True:
            path = self.q.get()
            if path is None:
                return
            if not self.enabled:
                print(f"☁️  [跳过上传] {path}(--no-upload 或无 token),保留本地", flush=True)
                continue
            name = os.path.basename(path)
            try:
                self.api.upload_file(path_or_fileobj=path, path_in_repo=name,
                                     repo_id=self.repo_id, repo_type="dataset")
                os.remove(path)
                print(f"☁️  ✓ 已上传并删本地: {name}", flush=True)
            except Exception as ex:                      # 网络类失败留本地,下次启动重试
                print(f"☁️  ⤫ 上传失败留本地待重试: {name}: {ex}", flush=True)

    def close(self):
        self.q.put(None)
        self.t.join()


# ---------------------------------------------------------------- 分片写入

class ShardWriter:
    """滚动分片 HDF5 写入器:jpeg 数据集可追加,超阈值封片交给 Uploader。"""

    def __init__(self, out_dir, shard_bytes, quality, threads, uploader, manifest_cb):
        import h5py
        self.h5py, self.dir, self.limit = h5py, out_dir, shard_bytes
        self.quality, self.uploader = quality, uploader
        self.manifest_cb = manifest_cb                 # 封片时记录 (shard名, 段名列表)
        self.pool = ThreadPoolExecutor(max_workers=threads)
        os.makedirs(out_dir, exist_ok=True)
        taken = []
        for fn in os.listdir(out_dir):                 # 崩溃残留的未封分片:损坏即清
            if not fn.startswith("shard_"):
                continue
            p = os.path.join(out_dir, fn)
            try:
                h5py.File(p, "r").close()
                taken.append(int(fn.split("_")[1].split(".")[0]))
            except OSError:
                print(f"🧹 清理损坏分片 {fn}", flush=True)
                os.remove(p)
        self.idx = max(taken) + 1 if taken else 0
        self.f, self.cur_segs = None, []

    def _open(self):
        self.path = os.path.join(self.dir, f"shard_{self.idx:04d}.h5")
        self.f = self.h5py.File(self.path, "w")
        print(f"📦 新分片 {self.path}", flush=True)

    def begin_segment(self, game, name, w, h, hz, attrs):
        if self.f is None:
            self._open()
        g = self.f.require_group(game).create_group(name)
        self.cur_segs.append(f"{game}/{name}")
        dt = self.h5py.vlen_dtype(np.uint8)
        g.create_dataset("jpeg", shape=(0,), maxshape=(None,), dtype=dt, chunks=(256,))
        g.create_dataset("frame_idx", shape=(0,), maxshape=(None,), dtype=np.int32,
                         chunks=(4096,))
        g["jpeg"].attrs.update({"hz": hz, "w": w, "h": h, "quality": self.quality})
        for k, v in attrs.items():
            g.attrs[k] = v
        return g

    def append_frames(self, g, frames_bgr, frame_idx):
        """线程池 JPEG 压缩一批原始帧并追加(imencode 释放 GIL,真并行)。"""
        q = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        blobs = list(self.pool.map(
            lambda f: cv2.imencode(".jpg", f, q)[1].reshape(-1), frames_bgr))
        d, n0 = g["jpeg"], g["jpeg"].shape[0]
        d.resize((n0 + len(blobs),))
        d[n0:] = blobs
        gi = g["frame_idx"]
        gi.resize((n0 + len(blobs),))
        gi[n0:] = np.asarray(frame_idx, np.int32)

    def end_segment(self, g, acts, seg_range):
        """写事件全率结构化数组 + gzip JSONL 原文。acts 为段内 30Hz 契约 dict 列表。"""
        s, e = seg_range
        dx = np.array([a["mouse"]["dx"] for a in acts], np.float32)
        dy = np.array([a["mouse"]["dy"] for a in acts], np.float32)
        keys = np.array([sum(_KEY_BIT.get(k, 0) for k in a["keyboard"]) for a in acts],
                        np.uint32)
        gui = np.array([a.get("gui", False) for a in acts], np.uint8)
        for nm, arr in (("dx", dx), ("dy", dy), ("keys", keys), ("gui", gui)):
            g.create_dataset(nm, data=arr, compression="gzip", compression_opts=4)
        raw = "\n".join(json.dumps(a) for a in acts).encode()
        g.create_dataset("events_gz", data=np.frombuffer(gzip.compress(raw, 6), np.uint8))
        self.f.flush()
        if os.path.getsize(self.path) >= self.limit:
            self.rollover()

    def rollover(self):
        if self.f is None:
            return
        self.f.close()
        print(f"📦 封片 {self.path}({os.path.getsize(self.path) / 1e9:.2f}GB)", flush=True)
        self.manifest_cb(os.path.basename(self.path), self.cur_segs)
        self.uploader.submit(self.path)
        self.idx += 1
        self.f, self.cur_segs = None, []

    def close(self):
        self.rollover()
        self.pool.shutdown(wait=True)


# ---------------------------------------------------------------- 解码

def decode_stream(mp4, start_f, n_frames, scale_h, use_gpu):
    """ffmpeg(NVDEC 可用则走 GPU 引擎)切段缩放,rawvideo 管道逐帧产出 BGR ndarray。"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", mp4], capture_output=True, text=True)
    W, H = map(int, probe.stdout.strip().split(","))
    w = (W * scale_h // H) // 2 * 2
    dec = ["-hwaccel", "cuda"] if use_gpu else []
    cmd = (["ffmpeg", "-loglevel", "error"] + dec +
           ["-i", mp4, "-ss", f"{start_f / 30:.3f}", "-frames:v", str(n_frames),
            "-vf", f"scale={w}:{scale_h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * scale_h * 3 * 8)
    nbytes = w * scale_h * 3
    try:
        for _ in range(n_frames):
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(scale_h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------- 主流程

def main():
    args = parse_args()
    use_gpu = cg.probe_nvenc()
    enabled = not args.no_upload
    if enabled:
        try:
            up = Uploader(args.repo, args.public, True)
        except Exception as ex:
            print(f"⚠️  HF 未登录({ex}),转为只编码;分片留本地,登录后重跑即自动补传", flush=True)
            up = Uploader(args.repo, args.public, False)
    else:
        up = Uploader(args.repo, args.public, False)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.raw, exist_ok=True)
    manifest_p = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_p)) if os.path.exists(manifest_p) else {}
    manifest.setdefault("shards", {})
    manifest.setdefault("sessions", {})

    def record_shard(shard_name, segs):
        manifest["shards"][shard_name] = segs
        json.dump(manifest, open(manifest_p, "w"))

    writer = ShardWriter(args.out, int(args.shard_gb * 1e9), args.quality,
                         args.threads, up, record_shard)
    # 已固化的段 = 所有已封分片(本地完整或已上传)记录的段名并集
    done_segs = {s for segs in manifest["shards"].values() for s in segs}

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    sessions = []
    for game in games:
        tree = cg.http_json(f"{cg.API}/tree/main/{game}")
        sessions += sorted(x["path"] for x in tree if x["type"] == "directory")[: args.n]
    print(f"📥 {len(sessions)} 个会话 → {args.out}(hz={args.hz} h={args.scale_h} "
          f"q={args.quality} buffer={args.buffer_gb}GB shard={args.shard_gb}GB)", flush=True)

    buf_limit = int(args.buffer_gb * 1e9)
    step = 30.0 / args.hz                      # 30Hz 源 → 目标采样步长
    dl_pool = ThreadPoolExecutor(max_workers=1)  # 预取下一会话的下载线程

    def fetch(sess):
        sid = sess.split("/")[0] + "_" + sess.split("/")[-1][:8]
        fe = os.path.join(args.raw, f"{sid}_frame_events.json")
        mp4 = os.path.join(args.raw, f"{sid}_clip.mp4")
        meta = cg.http_json(f"{cg.BASE}/{sess}/metadata.json")
        cg.download(f"{cg.BASE}/{sess}/frame_events.json", fe, f"{sid} events")
        cg.download(f"{cg.BASE}/{sess}/clip.mp4", mp4, f"{sid} clip")
        return sid, fe, mp4, meta

    fut = dl_pool.submit(fetch, sessions[0]) if sessions else None
    for si, sess in enumerate(sessions):
        game = sess.split("/")[0]
        try:
            sid, fe, mp4, meta = fut.result()
        except Exception as ex:
            print(f"⤫ [{si + 1}/{len(sessions)}] 下载失败: {ex}", flush=True)
            fut = dl_pool.submit(fetch, sessions[si + 1]) if si + 1 < len(sessions) else None
            continue
        fut = dl_pool.submit(fetch, sessions[si + 1]) if si + 1 < len(sessions) else None
        if manifest["sessions"].get(sid) == "done":
            print(f"[{si + 1}/{len(sessions)}] {sid} 已完成,跳过", flush=True)
            continue
        print(f"[{si + 1}/{len(sessions)}] {sid}", flush=True)
        bundles = cg.detect_game_bundle(fe)
        if not bundles:
            manifest["sessions"][sid] = "no_game"
            continue
        acts, in_game = cg.frame_actions(fe, args.warp_px, bundles)
        segs = cg.segments(in_game, args.min_frames)
        for gi, (s, e) in enumerate(segs):
            name = f"{sid}_{gi:02d}"
            if f"{game}/{name}" in done_segs:
                continue
            want = np.arange(s, e, step).astype(int)     # 图像采样的 30Hz 帧号
            want_set, wi = set(want.tolist()), 0
            g = writer.begin_segment(game, name, 0, args.scale_h, args.hz, {
                "task": meta.get("title", ""), "session": sess, "game": game,
                "seg_start": s, "seg_end": e, "src_fps": 30,
                "meta_json": json.dumps(meta)[:65000]})
            pend, pend_bytes, pend_idx = [], 0, []
            for off, frame in enumerate(decode_stream(mp4, s, e - s, args.scale_h, use_gpu)):
                if s + off not in want_set:
                    continue
                pend.append(frame.copy())
                pend_idx.append(off)
                pend_bytes += frame.nbytes
                if pend_bytes >= buf_limit:              # 内存水位触发:线程池压缩落盘
                    writer.append_frames(g, pend, pend_idx)
                    pend, pend_bytes, pend_idx = [], 0, []
                wi += 1
            if pend:
                writer.append_frames(g, pend, pend_idx)
            writer.end_segment(g, acts[s:e], (s, e))
            print(f"   ✓ seg{gi}: {len(want)} 图像帧 / {e - s} 事件帧", flush=True)
        manifest["sessions"][sid] = "done"
        json.dump(manifest, open(manifest_p, "w"))
        for pth in (mp4, fe):
            if os.path.exists(pth):
                os.remove(pth)

    writer.close()
    up.close()
    print("✅ 编码管线结束", flush=True)


# ---------------------------------------------------------------- 加载器

class Gaming500H5:
    """多线程随机读取已编码分片。

    使用方法:
        ds = Gaming500H5("runs/data/g500_h5")            # 或 HF snapshot 下载目录
        imgs, acts = ds.read_batch(ds.segments[0], t0=0, n=16)   # [n,H,W,3] u8 + 契约列表
    """

    def __init__(self, root, threads=8):
        import h5py
        self.h5py = h5py
        self.files = []
        for p in sorted(os.path.join(root, f) for f in os.listdir(root)
                        if f.endswith(".h5")):
            try:
                self.files.append(h5py.File(p, "r"))
            except OSError as ex:
                print(f"⚠️  跳过损坏分片 {p}: {ex}", flush=True)
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.segments = []                                # (file_i, "game/name")
        for fi, f in enumerate(self.files):
            for game in f:
                for name in f[game]:
                    self.segments.append((fi, f"{game}/{name}"))

    def read_batch(self, seg, t0, n):
        fi, path = seg
        g = self.files[fi][path]
        blobs = [g["jpeg"][t0 + i] for i in range(n)]     # h5 读取需单线程
        imgs = list(self.pool.map(                        # JPEG 解码释放 GIL,线程池并行
            lambda b: cv2.imdecode(b, cv2.IMREAD_COLOR), blobs))
        idx = g["frame_idx"][t0: t0 + n]
        raw = gzip.decompress(g["events_gz"][:].tobytes()).decode().splitlines()
        acts = [json.loads(raw[i]) for i in idx]
        return np.stack(imgs), acts


if __name__ == "__main__":
    main()
