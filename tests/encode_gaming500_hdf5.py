#!/usr/bin/env python3
"""gaming-500-hours → HDF5 分片归档:图像 JPEG 入 h5,事件全率无损保留,流式上传 HF。

对外接口:
    命令行脚本(main)——并行编码/上传管线;
    Gaming500H5(库类)——多线程随机读取已编码分片(训练/分析端加载器)。

设计(用户 2026-07-02 拍板;并行化与多样性同日追加):
  - **会话级并行**(--parallel,默认 3):每 worker 独立"下载→解析→解码→压缩"一个
    会话,分片写入用锁串行化;实测单流瓶颈在 ffmpeg 单核缩放与原片下载,并行 3 路
    吞吐 ×2-3(12 核 load 仅 ~5,NVDEC 余量 80%)。
  - **多样性优先**:会话顺序为游戏间轮转交错(每游戏轮流出 1 个会话),管线随时
    中断都保证"各游戏都已有样本",而非字母序头部游戏独占。
  - 解码帧积各 worker 内存缓冲,超 buffer_gb/parallel 即由共享线程池 JPEG 压缩
    追加进当前分片;分片超 --shard-gb 即**滚动**——新段开进新分片,旧分片上仍在写
    的段自然写完后立即封片(后台线程上传 HF 后删本地)。不等"全局无在写段"窗口:
    高并行下该窗口趋于不存在,等窗口会让分片无限膨胀、流式上传失效(2026-07-02 实测
    parallel=8 时 8.3GB 仍未封)。
  - "其他信息不要丢":事件以源生 30Hz 全率存两份——结构化数组(dx/dy f32、键位 u32
    位掩码、gui u8,gzip)+ 整段契约 JSONL gzip 原文;图像可按 --hz 降采样
    (frame_idx 记录到 30Hz 事件流的映射,不丢对齐)。
  - 断点续跑:manifest 记录"分片→段名"映射(封片时落盘)与"会话→段名列表"
    (会话完成时落盘);**跳过会话的唯一依据是其全部段都在已封分片里**——只有
    "sessions done"标记而分片未封即崩溃的,重启后自动重做。损坏分片启动即清。

HDF5 布局(每游戏段一组):
    /{game}/{session8}_{seg:02d}/
        jpeg       vlen u8 [N]   # JPEG 字节流,attrs: hz/w/h/quality
        frame_idx  i32 [N]       # 各图像帧在段内 30Hz 事件流中的下标
        dx, dy     f32 [M] (gzip);keys u32 [M] (gzip,VPT_KEYS 位掩码);gui u8 [M]
        events_gz  u8  [K]       # 整段逐帧契约 JSONL 的 gzip 原文(无损兜底)
        attrs: task / session / game / seg_start / seg_end / src_fps / meta_json

使用方法:
    PYTHONPATH=. python tests/encode_gaming500_hdf5.py --games minecraft --n 2 \
        --buffer-gb 1 --shard-gb 2 --no-upload          # 冒烟
    PYTHONPATH=. python tests/encode_gaming500_hdf5.py --games <全部游戏> --n 999
"""
import argparse
import gzip
import importlib.util
import json
import os
import queue
import subprocess
import threading
import time
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
    p.add_argument("--buffer-gb", type=float, default=10.0, help="内存原始帧缓冲上限(全 worker 合计)")
    p.add_argument("--shard-gb", type=float, default=20.0, help="单分片封片阈值")
    p.add_argument("--threads", type=int, default=8, help="JPEG 编码线程数(共享池)")
    p.add_argument("--parallel", type=int, default=3, help="并行编码的会话 worker 数")
    p.add_argument("--shard-prefix", default="",
                   help="分片文件名前缀(多机同写一个 HF 仓库时用于隔离,如 'cpu1_')")
    p.add_argument("--min-frames", type=int, default=900)
    p.add_argument("--seg-max-frames", type=int, default=27000,
                   help="单段最大事件帧数(30Hz,超长游戏段切块;分片超额上界="
                        "parallel×单块体积,0=不切)")
    p.add_argument("--warp-px", type=float, default=200.0)
    p.add_argument("--match", default=None)
    p.add_argument("--no-upload", action="store_true", help="只编码不上传(无 token 冒烟)")
    p.add_argument("--min-free-gb", type=float, default=30.0,
                   help="磁盘低水位:低于此值暂停接新会话,等待上传腾地或人工处置")
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
            for i in range(5):                           # 原地退避重试:瞬时网络/5xx
                try:                                     # 不该等到下次启动才补传
                    self.api.upload_file(path_or_fileobj=path, path_in_repo=name,
                                         repo_id=self.repo_id, repo_type="dataset")
                    os.remove(path)
                    print(f"☁️  ✓ 已上传并删本地: {name}", flush=True)
                    break
                except Exception as ex:
                    print(f"☁️  ⤫ 上传失败({i + 1}/5): {name}: {ex}", flush=True)
                    time.sleep(60 * (i + 1))
            else:                                        # 5 次仍败:留本地,重启补传
                print(f"☁️  ⤫ 放弃重试留本地: {name}", flush=True)

    def close(self):
        self.q.put(None)
        self.t.join()


# ---------------------------------------------------------------- 分片写入(多 worker 共享,锁串行化)

class ShardWriter:
    """滚动分片 HDF5 写入器:JPEG 压缩在锁外并行,h5 写入锁内串行。

    封片策略(高并行安全):当前分片超 --shard-gb 后,**新段一律开进新分片**,
    旧分片转入 pending,其上余段全部 end_segment 后立即封片上传。不等
    "全局无在写段"窗口——parallel≥8 时该窗口趋于不存在,分片会无限膨胀。
    """

    def __init__(self, out_dir, shard_bytes, quality, threads, uploader, manifest_cb,
                 sealed_shards=(), prefix=""):
        import h5py
        self.h5py, self.dir, self.limit = h5py, out_dir, shard_bytes
        self.quality, self.uploader = quality, uploader
        self.manifest_cb = manifest_cb
        self.prefix = prefix
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.lock = threading.Lock()
        os.makedirs(out_dir, exist_ok=True)
        taken = []
        for fn in os.listdir(out_dir):
            if not fn.startswith(prefix + "shard_"):
                continue
            p = os.path.join(out_dir, fn)
            if fn not in sealed_shards:                # 未封分片(崩溃残留):无论
                print(f"🧹 清理未封分片 {fn}", flush=True)  # 可否打开都弃置——其段
                os.remove(p)                           # 不在 manifest,必被重编,
                continue                               # 留着即孤儿
            taken.append(int(fn[len(prefix):].split("_")[1].split(".")[0]))
        self.idx = max(taken) + 1 if taken else 0
        self.cur = None          # {"f","path","segs","open"} 接收新段的当前分片
        self.pending = []        # 已滚动、余段未写完的旧分片

    def _open(self):
        path = os.path.join(self.dir, f"{self.prefix}shard_{self.idx:04d}.h5")
        self.idx += 1
        print(f"📦 新分片 {path}", flush=True)
        return {"f": self.h5py.File(path, "w"), "path": path, "segs": [], "open": 0}

    def _seal(self, sh):
        sh["f"].close()
        print(f"📦 封片 {sh['path']}({os.path.getsize(sh['path']) / 1e9:.2f}GB)",
              flush=True)
        self.manifest_cb(os.path.basename(sh["path"]), sh["segs"])
        self.uploader.submit(sh["path"])

    def _shard_of(self, g):
        fn = g.file.filename
        for sh in [self.cur] + self.pending:
            if sh is not None and sh["path"] == fn:
                return sh
        raise KeyError(f"段所属分片未登记: {fn}")

    def begin_segment(self, game, name, hz, attrs):
        with self.lock:
            if self.cur is not None:
                self.cur["f"].flush()                  # 刷盘后按真实大小判滚动
                if os.path.getsize(self.cur["path"]) >= self.limit:
                    if self.cur["open"] == 0:
                        self._seal(self.cur)
                    else:
                        self.pending.append(self.cur)  # 余段写完时在 end_segment 封
                    self.cur = None
            if self.cur is None:
                self.cur = self._open()
            g = self.cur["f"].require_group(game).create_group(name)
            self.cur["segs"].append(f"{game}/{name}")
            self.cur["open"] += 1
            dt = self.h5py.vlen_dtype(np.uint8)
            g.create_dataset("jpeg", shape=(0,), maxshape=(None,), dtype=dt, chunks=(256,))
            g.create_dataset("frame_idx", shape=(0,), maxshape=(None,), dtype=np.int32,
                             chunks=(4096,))
            g["jpeg"].attrs.update({"hz": hz, "w": 0, "h": 0, "quality": self.quality})
            for k, v in attrs.items():
                g.attrs[k] = v
            return g

    def append_frames(self, g, frames_bgr, frame_idx):
        """共享线程池 JPEG 压缩(锁外,真并行)→ 锁内追加写。"""
        q = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        blobs = list(self.pool.map(
            lambda f: cv2.imencode(".jpg", f, q)[1].reshape(-1), frames_bgr))
        with self.lock:
            if g["jpeg"].attrs.get("w", 0) == 0:
                g["jpeg"].attrs["w"] = frames_bgr[0].shape[1]
                g["jpeg"].attrs["h"] = frames_bgr[0].shape[0]
            d, n0 = g["jpeg"], g["jpeg"].shape[0]
            d.resize((n0 + len(blobs),))
            d[n0:] = blobs
            gi = g["frame_idx"]
            gi.resize((n0 + len(blobs),))
            gi[n0:] = np.asarray(frame_idx, np.int32)

    def end_segment(self, g, acts):
        """写事件全率数组 + gzip JSONL 原文;所属分片余段归零时按策略封片。"""
        dx = np.array([a["mouse"]["dx"] for a in acts], np.float32)
        dy = np.array([a["mouse"]["dy"] for a in acts], np.float32)
        keys = np.array([sum(_KEY_BIT.get(k, 0) for k in a["keyboard"]) for a in acts],
                        np.uint32)
        gui = np.array([a.get("gui", False) for a in acts], np.uint8)
        raw = "\n".join(json.dumps(a) for a in acts).encode()
        blob = np.frombuffer(gzip.compress(raw, 6), np.uint8)
        with self.lock:
            sh = self._shard_of(g)
            for nm, arr in (("dx", dx), ("dy", dy), ("keys", keys), ("gui", gui)):
                g.create_dataset(nm, data=arr, compression="gzip", compression_opts=4)
            g.create_dataset("events_gz", data=blob)
            sh["f"].flush()
            sh["open"] -= 1
            if sh["open"] > 0:
                return
            if sh in self.pending:                     # 已滚动旧片:余段清零即封
                self.pending.remove(sh)
                self._seal(sh)
            elif os.path.getsize(sh["path"]) >= self.limit:   # 当前片:超阈值即封
                self._seal(sh)
                self.cur = None

    def close(self):
        with self.lock:
            for sh in self.pending + ([self.cur] if self.cur else []):
                self._seal(sh)                         # 收尾:残片一律封存上传
            self.pending, self.cur = [], None
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=w * scale_h * 3 * 8)  # stderr 静默:读满即关管道
                                                          # 的 EPIPE 噪声无诊断价值
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

def chunk_segments(segs, max_frames, min_frames):
    """超长游戏段按 max_frames 切块;不足 min_frames 的尾块并回前块。

    Parameters
    ----------
    segs : list[tuple[int, int]]   30Hz 事件帧区间 [s, e)
    max_frames : int               单块上限(0 = 不切)
    min_frames : int               尾块下限(过短并回前块,前块 ≤ max+min)

    Returns
    -------
    list[tuple[int, int]]
    """
    if max_frames <= 0:
        return list(segs)
    out = []
    for s, e in segs:
        cuts = list(range(s, e, max_frames)) + [e]
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < min_frames:
            cuts.pop(-2)                               # 尾块过短并回前块
        out += list(zip(cuts[:-1], cuts[1:]))
    return out


def interleave_by_game(games, per_game_sessions):
    """游戏间轮转交错:g1s1,g2s1,...,gNs1,g1s2,... 保证随时中断都覆盖各游戏。"""
    order, i = [], 0
    while True:
        row = [s[i] for s in per_game_sessions.values() if i < len(s)]
        if not row:
            return order
        order += row
        i += 1


def main():
    args = parse_args()
    use_gpu = cg.probe_nvenc()
    if args.no_upload:
        up = Uploader(args.repo, args.public, False)
    else:
        try:
            up = Uploader(args.repo, args.public, True)
        except Exception as ex:
            print(f"⚠️  HF 未登录({ex}),转为只编码;分片留本地,登录后重跑即自动补传", flush=True)
            up = Uploader(args.repo, args.public, False)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.raw, exist_ok=True)
    manifest_p = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_p)) if os.path.exists(manifest_p) else {}
    manifest.setdefault("shards", {})
    manifest.setdefault("sessions", {})
    mlock = threading.Lock()

    def msave():
        json.dump(manifest, open(manifest_p, "w"))

    def record_shard(shard_name, segs):
        with mlock:
            manifest["shards"][shard_name] = segs
            msave()

    writer = ShardWriter(args.out, int(args.shard_gb * 1e9), args.quality,
                         args.threads, up, record_shard,
                         sealed_shards=set(manifest["shards"]), prefix=args.shard_prefix)
    # 已固化的段 = 所有已封分片记录的段名并集(跳过判据,比"会话 done"标记更强)
    done_segs = {s for segs in manifest["shards"].values() for s in segs}

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    per_game = {}
    for game in games:
        try:
            tree = cg.http_json(f"{cg.API}/tree/main/{game}")
        except Exception as ex:
            print(f"⤫ 列举 {game} 失败: {ex}", flush=True)
            continue
        per_game[game] = sorted(
            x["path"] for x in tree if x["type"] == "directory")[: args.n]
    sessions = interleave_by_game(games, per_game)
    print(f"📥 {len(sessions)} 个会话({len(per_game)} 游戏,轮转交错)→ {args.out}"
          f"(hz={args.hz} q={args.quality} 并行={args.parallel} "
          f"buffer={args.buffer_gb}GB shard={args.shard_gb}GB)", flush=True)

    buf_limit = int(args.buffer_gb * 1e9 / max(1, args.parallel))
    step = 30.0 / args.hz
    counter = {"done": 0}

    def session_complete(sid):
        """会话可跳过 ⇔ 其记录的全部段都已固化在已封分片里。"""
        segs = manifest["sessions"].get(sid)
        return isinstance(segs, list) and all(s in done_segs for s in segs)

    def process_session(sess):
        game = sess.split("/")[0]
        sid = game + "_" + sess.split("/")[-1][:8]
        if session_complete(sid):
            return
        while True:                                    # 磁盘低水位:上传受阻时分片
            st = os.statvfs(args.out)                  # 积压本地,暂停接新会话防爆盘
            free_gb = st.f_bavail * st.f_frsize / 1e9
            if free_gb >= args.min_free_gb:
                break
            print(f"⏸ 磁盘余 {free_gb:.0f}GB < {args.min_free_gb}GB,{sid} 暂停 5 分钟",
                  flush=True)
            time.sleep(300)
        fe = os.path.join(args.raw, f"{sid}_frame_events.json")
        mp4 = os.path.join(args.raw, f"{sid}_clip.mp4")
        try:
            meta = cg.http_json(f"{cg.BASE}/{sess}/metadata.json")
            cg.download(f"{cg.BASE}/{sess}/frame_events.json", fe, f"{sid} events")
            bundles = cg.detect_game_bundle(fe)
            if not bundles:
                with mlock:
                    manifest["sessions"][sid] = []
                    msave()
                return
            acts, in_game = cg.frame_actions(fe, args.warp_px, bundles)
            segs = chunk_segments(cg.segments(in_game, args.min_frames),
                                  args.seg_max_frames, args.min_frames)
            if not segs:
                with mlock:
                    manifest["sessions"][sid] = []
                    msave()
                return
            cg.download(f"{cg.BASE}/{sess}/clip.mp4", mp4, f"{sid} clip")
        except Exception as ex:
            print(f"⤫ {sid} 下载失败: {ex}", flush=True)
            return
        try:                                           # 单会话异常只弃当前会话,
            seg_names = []                             # 不许杀死整个管线
            for gi, (s, e) in enumerate(segs):
                name = f"{sid}_{gi:02d}"
                seg_names.append(f"{game}/{name}")
                if f"{game}/{name}" in done_segs:
                    continue
                want = np.arange(s, e, step).astype(int)
                want_set = set(want.tolist())
                g = writer.begin_segment(game, name, args.hz, {
                    "task": meta.get("title", ""), "session": sess, "game": game,
                    "seg_start": s, "seg_end": e, "src_fps": 30,
                    "meta_json": json.dumps(meta)[:65000]})
                pend, pend_bytes, pend_idx = [], 0, []
                for off, frame in enumerate(
                        decode_stream(mp4, s, e - s, args.scale_h, use_gpu)):
                    if s + off not in want_set:
                        continue
                    pend.append(frame.copy())
                    pend_idx.append(off)
                    pend_bytes += frame.nbytes
                    if pend_bytes >= buf_limit:          # 内存水位:压缩落盘
                        writer.append_frames(g, pend, pend_idx)
                        pend, pend_bytes, pend_idx = [], 0, []
                if pend:
                    writer.append_frames(g, pend, pend_idx)
                writer.end_segment(g, acts[s:e])
                print(f"   ✓ {name}: {len(want)} 图像帧 / {e - s} 事件帧", flush=True)
            with mlock:
                manifest["sessions"][sid] = seg_names
                counter["done"] += 1
                msave()
                print(f"[{counter['done']}/{len(sessions)}] ✔ {sid}", flush=True)
        except Exception as ex:
            print(f"⤫ {sid} 编码异常,弃置本会话: {type(ex).__name__}: {ex}", flush=True)
        finally:
            for pth in (mp4, fe):
                if os.path.exists(pth):
                    os.remove(pth)

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        list(ex.map(process_session, sessions))

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
