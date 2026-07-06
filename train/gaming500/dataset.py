"""gaming500-720p-hdf5 分片 → 训练样本(原生动作形态,不套 VPT 契约)。

设计动机(为何不和 VPT 对齐、变率聚合语义)见 knowledge/design_gaming500_consume.md。

对外接口:
    Gaming500Dataset —— map 式序列窗口数据集;seq_len=1 即 tokenizer 单帧模式。
    unpack_keys      —— u32 位掩码 → [..., 20] uint8 multihot(位序见 KEY_NAMES)。
    KEY_NAMES        —— 编码期写入的 20 键词表(bit 位序 = 掩码 bit 位;数据事实,无游戏语义)。
"""
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# bit i = 编码期 tests/encode_gaming500_hdf5.py 写入的第 i 位;此处复刻为**数据事实**,
# 不引入 Minecraft 语义(多游戏域不继承 VPT 键义,见 design_gaming500_consume.md §3-4)。
KEY_NAMES = [
    "key_w", "key_a", "key_s", "key_d", "key_space", "key_sneak", "key_sprint",
    "key_attack", "key_use", "key_drop", "key_inventory",
    "key_hotbar.1", "key_hotbar.2", "key_hotbar.3", "key_hotbar.4", "key_hotbar.5",
    "key_hotbar.6", "key_hotbar.7", "key_hotbar.8", "key_hotbar.9",
]
N_KEYS = len(KEY_NAMES)


def unpack_keys(mask):
    """u32 位掩码 → multihot。

    Parameters
    ----------
    mask : ndarray[uint32], Shape [..]   每元素一帧的 20 键位掩码

    Returns
    -------
    ndarray[uint8], Shape [.., 20]        bit i(=KEY_NAMES[i])按下为 1
    """
    m = np.asarray(mask, dtype=np.uint32)[..., None]
    bits = (m >> np.arange(N_KEYS, dtype=np.uint32)) & np.uint32(1)
    return bits.astype(np.uint8)


def _in_split(seg, holdout, split):
    """段名 → 是否属于目标 split。holdout 是决定性选出的 holdout 段名集合。"""
    if split == "all":
        return True
    return (seg in holdout) if split == "holdout" else (seg not in holdout)


def _resize_crop(img, size, mode, rng):
    """720p BGR 帧 → [size,size,3] BGR。mode: resize(方形缩放)/center/random(原生密度裁剪)。"""
    h, w = img.shape[:2]
    if mode == "resize":
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    side = min(h, w, max(size, 1))
    if mode == "center":
        y0, x0 = (h - side) // 2, (w - side) // 2
    else:                                              # random:原生像素随机裁剪
        y0 = int(rng.integers(0, h - side + 1))
        x0 = int(rng.integers(0, w - side + 1))
    crop = img[y0:y0 + side, x0:x0 + side]
    if side != size:                                   # 裁块非目标尺寸时再缩放兜底
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def _decode_frame(blob, size, mode, seed, want_gray=False):
    """JPEG 字节 → [3,size,size] uint8 RGB(CHW);want_gray 时附带 45×80 全帧灰度
    (step2 周边消息通道用:解码本来就发生,多一次缩放 ~0.1ms)。"""
    bgr = cv2.imdecode(blob, cv2.IMREAD_COLOR)         # 存的是 BGR
    gray = (cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (80, 45),
                       interpolation=cv2.INTER_AREA) if want_gray else None)
    bgr = _resize_crop(bgr, size, mode, np.random.default_rng(seed))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return (chw, gray) if want_gray else chw


# ---- step2 周边消息通道(fovea-twotower-step2 §1,变化通道 v0) ----
N_MSG = 11            # 8方位one-hot + log显著度 + log背景(pan代理) + valid
_MSG_PEAK_TH = 45.0   # 事件门:3×3平滑后 peak−bg 阈值(360p 校准,事件率≈8.6%)
_MSG_BG_TH = 3.0      # 准静态门:bg 中值低于此才判事件(粗代自运动补偿)


def _periph_msgs(grays):
    """[L] 张 45×80 灰度 → [L,N_MSG] float32。第 0 帧无差分,valid=0。

    帧差 3×3 平滑去单像素噪声;掩膜=游戏视口中环(排除边缘 HUD 带:小地图/
    击杀播报/弹药栏永动,会淹没真实事件)再挖去中心凹区(模型看得见);
    周边取背景中值 bg 与峰值 peak;peak−bg 过阈且 bg 准静态 → 事件,
    方位=峰值位置相对屏心的 8 扇区。"""
    L = len(grays)
    out = np.zeros((L, N_MSG), np.float32)
    cy, cx = 45 // 2, 80 // 2
    mask = np.zeros((45, 80), bool)
    mask[8:37, 10:70] = True                           # 视口中环(HUD 边带除外)
    mask[cy - 8:cy + 8, cx - 8:cx + 8] = False         # 凹区(模型看得见)不算周边
    prev = grays[0].astype(np.float32)
    for j in range(1, L):
        cur = grays[j].astype(np.float32)
        dimg = cv2.blur(np.abs(cur - prev), (3, 3))
        prev = cur
        dimg[~mask] = 0.0
        bg = float(np.median(dimg[mask]))
        pk = float(dimg.max())
        out[j, 9] = np.log1p(bg) / 4.0                 # pan 代理
        out[j, 10] = 1.0                               # valid
        if pk - bg > _MSG_PEAK_TH and bg < _MSG_BG_TH:
            y, x = divmod(int(dimg.argmax()), 80)
            ang = np.arctan2(y - cy, x - cx)
            out[j, int(((ang + np.pi) / (2 * np.pi) * 8)) % 8] = 1.0
            out[j, 8] = min(np.log1p(pk - bg) / 4.0, 1.5)
    return out


class Gaming500Dataset(Dataset):
    """已下载分片的 map 式序列窗口加载器(原生动作契约)。

    扫描 root 下所有 .h5 分片,枚举每个游戏段的图像窗口;__getitem__ 现场 JPEG 解码
    + frame_idx 变率对齐(聚合语义见 design_gaming500_consume.md §5)。h5py 句柄按进程
    惰性打开(fork 安全:worker 内首次访问才 open)。

    cache=True(仅 seq_len=1):__init__ 期多线程把**全部选中帧**解码进一个 uint8 大张量
    [N,3,H,W] 常驻内存;训练零解码,且 fork 后 worker 以 COW 共享同一张量(不复制)。
    既喂饱 CPU 内存又消灭解码瓶颈。

    样本(seq_len=L):
        img   uint8  [L,3,H,W]   RGB,H=W=img_size
        dx,dy f32    [L-1]       相邻图间位移**和**,原始像素(未归一化)
        keys  uint8  [L-1,20]    区间 OR 键掩码
        gui   uint8  [L-1]       区间内是否进过菜单
        dt    int32  [L-1]       区间跨的 30Hz 帧数(真实时距=dt/30 s)
        game,task str
    seq_len=1(tokenizer 单帧):只出 img[1,3,H,W]+game/task,不做动作对齐。

    Parameters
    ----------
    root : str               分片目录(含 shard_*.h5)
    seq_len : int            窗口图像帧数(≥1);1 = 单帧模式
    img_size : int           输出方形边长(720p 缩放/裁剪目标)
    stride : int             相邻窗口在段内的图像帧步长(默认 = seq_len,不重叠)
    crop_mode : str          resize / center / random(见 _resize_crop)
    seed : int               random 裁剪与打乱的基种
    split : str              all / train / holdout(按段名决定性分,防时间漏洩)
    holdout_frac : float     holdout 段占比(split≠all 时生效;至少留 1 段)
    cache : bool             全帧解码入内存(仅 seq_len=1;喂饱 CPU + 免训练期解码)
    cache_threads : int      建缓存的解码线程数
    """

    def __init__(self, root, seq_len=16, img_size=128, stride=None, crop_mode="resize",
                 seed=0, split="all", holdout_frac=0.03, cache=False, cache_threads=16,
                 periph=False):
        super().__init__()
        self.root = root
        self.seq_len = max(1, int(seq_len))
        self.img_size = int(img_size)
        self.stride = int(stride) if stride else self.seq_len
        self.crop_mode = crop_mode
        self.periph = bool(periph)                     # step2:附带周边消息 msg [L,N_MSG]
        self.seed = int(seed)
        self._files = {}                               # pid → {path: h5py.File}(惰性/进程隔离)
        self._lock = threading.Lock()
        self.cache = None

        import h5py
        shards = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".h5"))
        if not shards:
            raise FileNotFoundError(f"{root} 下无 .h5 分片")

        # 第 1 遍:收集所有段(只读元数据)。逐片 try-open,跳损坏片。
        all_segs = []                                  # (path, "game/name", N, task)
        n_ok = 0
        for p in shards:
            try:
                f = h5py.File(p, "r")
            except OSError as ex:
                print(f"⚠️  跳过损坏分片 {p}: {ex}", flush=True)
                continue
            n_ok += 1
            for game in f:
                for name in f[game]:
                    g = f[game][name]
                    if "jpeg" not in g or g["jpeg"].shape[0] < self.seq_len:
                        continue
                    all_segs.append((p, f"{game}/{name}", g["jpeg"].shape[0],
                                     str(g.attrs.get("task", ""))))
            f.close()
        # holdout 段:按 md5(段名) 排名取底部 k 个,k≥1(段少也保证 holdout 非空)
        holdout = set()
        if split != "all":
            ranked = sorted(all_segs,
                            key=lambda s: hashlib.md5(s[1].encode()).hexdigest())
            k = max(1, round(holdout_frac * len(ranked)))
            holdout = {s[1] for s in ranked[:k]}

        self.segments, self.windows, self.tasks = [], [], {}
        for p, seg, n, task in all_segs:
            if not _in_split(seg, holdout, split):
                continue
            self.segments.append((p, seg, n))
            self.tasks[seg] = task
            for t0 in range(0, n - self.seq_len + 1, self.stride):
                self.windows.append((p, seg, t0))
        if not self.windows:
            raise RuntimeError(
                f"{root}(split={split}): 无长度≥{self.seq_len} 的段;调小 seq_len/"
                f"holdout_frac 或补下载分片")
        print(f"[Gaming500Dataset:{split}] {n_ok} 片 / {len(self.segments)} 段 / "
              f"{len(self.windows)} 窗口 | seq_len={self.seq_len} img={self.img_size} "
              f"crop={self.crop_mode}", flush=True)

        if cache:
            self._build_cache(cache_threads)

    def _build_cache(self, threads):
        """多线程把全部窗口(seq_len=1)解码进常驻内存大张量(COW 供 worker 共享)。"""
        if self.seq_len != 1:
            raise ValueError("cache 仅支持 seq_len=1(tokenizer 单帧)")
        n, S = len(self.windows), self.img_size
        gb = n * 3 * S * S / 1e9
        print(f"🧠 建内存缓存: {n} 帧 × 3×{S}×{S} ≈ {gb:.1f}GB({threads} 线程解码)...",
              flush=True)
        self.cache = torch.empty((n, 3, S, S), dtype=torch.uint8)

        def fill(i):
            path, seg, t0 = self.windows[i]
            blob = self._handle(path)[seg]["jpeg"][t0]
            self.cache[i] = _decode_frame(blob, S, self.crop_mode, self.seed + i)

        t = __import__("time").time()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for j, _ in enumerate(ex.map(fill, range(n))):
                if (j + 1) % 20000 == 0:
                    print(f"   缓存 {j+1}/{n}", flush=True)
        # h5 句柄用完即关(缓存后训练期不再触碰磁盘)
        for f in self._files.get(os.getpid(), {}).values():
            f.close()
        self._files.clear()
        print(f"🧠 缓存就绪,耗时 {__import__('time').time()-t:.0f}s", flush=True)

    def _handle(self, path):
        """按进程惰性打开的只读 h5py 句柄(worker fork 后各自 open,避免共享损坏)。"""
        pid = os.getpid()
        with self._lock:
            fmap = self._files.setdefault(pid, {})
            f = fmap.get(path)
            if f is None:
                import h5py
                f = h5py.File(path, "r")
                fmap[path] = f
            return f

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        path, seg, t0 = self.windows[i]
        if self.cache is not None:                     # 缓存命中:纯索引,零解码
            # 必须 .clone():self.cache[i:i+1] 是 34GB 大张量的视图,若直接经
            # DataLoader 队列外传,PyTorch 张量 reduce 会把**整块底层存储**搬进
            # 共享内存(每 worker 一次)→ 内存爆炸 OOM。clone 出独立小张量只搬本样本。
            return {"img": self.cache[i:i + 1].clone(), "game": seg.split("/")[0],
                    "task": self.tasks.get(seg, "")}
        rng = np.random.default_rng(self.seed + i)     # 每样本确定性(裁剪可复现)
        g = self._handle(path)[seg]
        L = self.seq_len

        if self.periph:
            pairs = [_decode_frame(g["jpeg"][t0 + k], self.img_size, self.crop_mode,
                                   self.seed + i + k, want_gray=True) for k in range(L)]
            imgs = [p[0] for p in pairs]
            msg = torch.from_numpy(_periph_msgs([p[1] for p in pairs]))
        else:
            imgs = [_decode_frame(g["jpeg"][t0 + k], self.img_size, self.crop_mode,
                                  self.seed + i + k) for k in range(L)]
            msg = None
        img = torch.stack(imgs, 0)                     # [L,3,H,W] uint8
        sample = {"img": img, "game": seg.split("/")[0], "task": self.tasks.get(seg, "")}
        if msg is not None:
            sample["msg"] = msg                        # [L,N_MSG] float32
        if L == 1:                                     # tokenizer 单帧:不做动作对齐
            return sample

        fidx = g["frame_idx"][t0:t0 + L].astype(np.int64)   # 各图在 30Hz 流的偏移
        dx_full, dy_full = g["dx"][:], g["dy"][:]
        keys_full, gui_full = g["keys"][:], g["gui"][:]
        M = dx_full.shape[0]
        dx = np.zeros(L - 1, np.float32)
        dy = np.zeros(L - 1, np.float32)
        keys = np.zeros((L - 1, N_KEYS), np.uint8)
        gui = np.zeros(L - 1, np.uint8)
        dt = np.zeros(L - 1, np.int32)
        for j in range(L - 1):
            a, b = fidx[j], fidx[j + 1]                # 区间 (a, b]:第 j→j+1 张图间
            lo, hi = min(a + 1, M), min(b + 1, M)      # 30Hz 子帧切片,防越界截断
            dt[j] = b - a
            if hi > lo:                                # 位移可加 → 求和(design §5)
                dx[j] = dx_full[lo:hi].sum()
                dy[j] = dy_full[lo:hi].sum()
                keys[j] = np.bitwise_or.reduce(unpack_keys(keys_full[lo:hi]), axis=0)
                gui[j] = np.uint8(gui_full[lo:hi].any())
        sample.update(dx=torch.from_numpy(dx), dy=torch.from_numpy(dy),
                      keys=torch.from_numpy(keys), gui=torch.from_numpy(gui),
                      dt=torch.from_numpy(dt))
        return sample
