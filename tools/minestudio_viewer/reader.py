"""MineStudio LMDB 数据集读取器（轻量，无需 minestudio 依赖）。

数据集按模态分目录（image/action/meta_info/segmentation/event），每个模态目录下
有若干 part-* 子目录，每个 part 是一个 LMDB。核心结构：

  元数据键（pickle 编码）:
    __chunk_size__        int         每个 chunk 的帧数（如 32）
    __chunk_infos__       List[Dict]  [{'episode': str, 'episode_idx': int, 'num_frames': int}]
    __num_episodes__      int
    __num_total_frames__  int

  数据键: key = str((episode_idx, chunk_start)).encode()
          chunk_start 是 chunk_size 的整数倍；帧按 chunk 成块存储。

不同模态里同一 episode 的 episode_idx 不保证一致，必须按 episode 名对齐。
"""
from __future__ import annotations

import glob
import io
import os
import pickle
from dataclasses import dataclass, field

import lmdb
import numpy as np

MODALS = ["image", "action", "meta_info", "segmentation"]


@dataclass
class EpisodeLoc:
    """一个 episode 在某个模态 LMDB 中的定位信息。"""

    lmdb_path: str
    episode_idx: int  # LMDB 内部 key 使用的整数索引
    num_frames: int


@dataclass
class EpisodeEntry:
    """一个 episode 在各模态中的定位，以及可用帧数（取各模态最小值）。"""

    name: str
    locs: dict = field(default_factory=dict)  # modal -> EpisodeLoc
    num_frames: int = 0
    part: str = ""  # 该 episode 主要所在的 part 名（用于分组展示）


class _LmdbCache:
    """按路径缓存 LMDB env，避免重复打开。"""

    def __init__(self):
        self._envs = {}

    def env(self, path):
        e = self._envs.get(path)
        if e is None:
            e = lmdb.open(
                path, readonly=True, lock=False, subdir=True,
                readahead=False, max_readers=512, meminit=False,
            )
            self._envs[path] = e
        return e

    def close(self):
        for e in self._envs.values():
            try:
                e.close()
            except Exception:
                pass
        self._envs.clear()


def _scan_modal(base_dir, modal):
    """扫描一个模态目录下所有 part 的 LMDB，返回 {episode_name: EpisodeLoc}。"""
    result = {}
    modal_dir = os.path.join(base_dir, modal)
    if not os.path.isdir(modal_dir):
        return result
    for part in sorted(glob.glob(os.path.join(modal_dir, "part-*"))):
        if not os.path.exists(os.path.join(part, "data.mdb")):
            continue  # 尚未下载完
        try:
            env = lmdb.open(part, readonly=True, lock=False, subdir=True, max_readers=512)
        except lmdb.Error:
            continue
        with env.begin() as txn:
            raw = txn.get(b"__chunk_infos__")
            if raw is None:
                env.close()
                continue
            chunk_infos = pickle.loads(raw)
        env.close()
        for info in chunk_infos:
            result[info["episode"]] = EpisodeLoc(
                lmdb_path=part,
                episode_idx=info["episode_idx"],
                num_frames=info["num_frames"],
            )
    return result


def _part_name(path):
    return os.path.basename(path.rstrip("/"))


class MineStudioDataset:
    """读取整个 MineStudio 数据集，按 episode 名对齐各模态。"""

    def __init__(self, base_dir, chunk_size=32):
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self._cache = _LmdbCache()
        self._scan_cache: dict[str, dict] = {}
        self.episodes: list[EpisodeEntry] = []
        self._by_name: dict[str, EpisodeEntry] = {}
        self.build_index()

    def build_index(self):
        """扫描各模态，按 episode 名对齐。image 为必需模态；其余可选。"""
        self._scan_cache.clear()
        modal_maps = {m: _scan_modal(self.base_dir, m) for m in MODALS}
        # 以 image 存在的 episode 为基准（可视化必须有画面）；若 image 尚未下载，退回用并集
        base_names = set(modal_maps["image"].keys())
        if not base_names:
            base_names = set().union(*(set(m.keys()) for m in modal_maps.values()))

        entries = []
        for name in sorted(base_names):
            locs = {}
            for modal in MODALS:
                loc = modal_maps[modal].get(name)
                if loc is not None:
                    locs[modal] = loc
            if not locs:
                continue
            # 可用帧数取所有已存在模态的最小 num_frames，保证跨模态对齐安全
            nf = min(l.num_frames for l in locs.values())
            part = _part_name(locs.get("image", next(iter(locs.values()))).lmdb_path)
            entries.append(EpisodeEntry(name=name, locs=locs, num_frames=nf, part=part))

        self.episodes = entries
        self._by_name = {e.name: e for e in entries}

    # ---- 元信息查询 ----
    def num_episodes(self):
        return len(self.episodes)

    def parts(self):
        """返回所有 part 名（按 image 模态归属），用于分组下拉。"""
        seen = []
        for e in self.episodes:
            if e.part not in seen:
                seen.append(e.part)
        return sorted(seen)

    def episodes_in_part(self, part):
        return [e for e in self.episodes if e.part == part]

    def get(self, name) -> EpisodeEntry | None:
        return self._by_name.get(name)

    # ---- 全片段扫描：按 chunk 批量读取，构建 per-frame 信号，用于帧筛选 ----
    def scan_episode(self, entry: EpisodeEntry):
        """扫描整个片段，返回一个 dict：
            n            帧数
            actions      {动作键: np.ndarray(n,)}（0/1；camera 除外）
            cam_mag      np.ndarray(n,)  视角移动幅度 |pitch|+|yaw|
            gui_open     np.ndarray(n,) bool
            gui_inv      np.ndarray(n,) bool
            events       list[set[str]]  每帧"新发生"的事件名集合（对累计计数做差分）
            seg_events   list[set[str]]  每帧分割交互的 event 描述集合
            event_vocab  set[str]        全片段出现过的事件名（含分割）
        结果按 entry.name 缓存。
        """
        cached = self._scan_cache.get(entry.name)
        if cached is not None:
            return cached

        n = entry.num_frames
        cs = self.chunk_size
        actions: dict[str, np.ndarray] = {}
        cam_mag = np.zeros(n, dtype=np.float32)
        gui_open = np.zeros(n, dtype=bool)
        gui_inv = np.zeros(n, dtype=bool)
        events = [set() for _ in range(n)]
        seg_events = [set() for _ in range(n)]
        vocab: set[str] = set()

        a_loc = entry.locs.get("action")
        m_loc = entry.locs.get("meta_info")
        s_loc = entry.locs.get("segmentation")
        prev_counts: dict[str, float] = {}

        for start in range(0, n, cs):
            end = min(start + cs, n)
            length = end - start
            # 动作
            if a_loc is not None:
                cb = self._read_chunk_bytes(a_loc, start)
                if cb is not None:
                    obj = pickle.loads(cb)
                    for k, v in obj.items():
                        arr = np.asarray(v)
                        if k == "camera" and arr.ndim == 2 and arr.shape[1] >= 2:
                            mag = np.abs(arr[:length, 0]) + np.abs(arr[:length, 1])
                            cam_mag[start:end] = mag
                        elif arr.ndim == 1:
                            if k not in actions:
                                actions[k] = np.zeros(n, dtype=np.int8)
                            actions[k][start:end] = arr[:length].astype(np.int8)
            # meta_info：GUI + 事件差分
            if m_loc is not None:
                cb = self._read_chunk_bytes(m_loc, start)
                if cb is not None:
                    self._scan_meta_chunk(pickle.loads(cb), start, length, gui_open,
                                          gui_inv, events, vocab, prev_counts)
            # segmentation 交互事件
            if s_loc is not None:
                cb = self._read_chunk_bytes(s_loc, start)
                if cb is not None:
                    self._scan_seg_chunk(pickle.loads(cb), start, length, seg_events, vocab)

        result = {
            "n": n, "actions": actions, "cam_mag": cam_mag,
            "gui_open": gui_open, "gui_inv": gui_inv,
            "events": events, "seg_events": seg_events, "event_vocab": vocab,
        }
        self._scan_cache[entry.name] = result
        return result

    @staticmethod
    def _scan_meta_chunk(obj, start, length, gui_open, gui_inv, events, vocab, prev_counts):
        # obj 可能是 List[Dict]（逐帧）或 dict-of-list
        def frame_dict(i):
            if isinstance(obj, list):
                return obj[i] if i < len(obj) else None
            if isinstance(obj, dict):
                return {k: (v[i] if hasattr(v, "__getitem__") and i < len(v) else None)
                        for k, v in obj.items()}
            return None

        for i in range(length):
            fd = frame_dict(i)
            if not isinstance(fd, dict):
                continue
            gi = start + i
            if fd.get("isGuiOpen"):
                gui_open[gi] = True
            if fd.get("isGuiInventory"):
                gui_inv[gi] = True
            ev = fd.get("events") or {}
            if isinstance(ev, dict):
                for name, val in ev.items():
                    if "custom" in name:  # 累计的统计量，跳过
                        continue
                    try:
                        cur = float(val)
                    except (TypeError, ValueError):
                        continue
                    prev = prev_counts.get(name)
                    if prev is not None and cur > prev:  # 计数增加=本帧发生
                        events[gi].add(name)
                        vocab.add(name)
                    prev_counts[name] = cur

    @staticmethod
    def _scan_seg_chunk(obj, start, length, seg_events, vocab):
        if not isinstance(obj, list):
            return
        for i in range(min(length, len(obj))):
            fd = obj[i]
            if not isinstance(fd, dict):
                continue
            for _, interaction in fd.items():
                if isinstance(interaction, dict):
                    evt = interaction.get("event")
                    if evt:
                        tag = "seg:" + str(evt)
                        seg_events[start + i].add(tag)
                        vocab.add(tag)

    # ---- 底层：读取指定模态的原始 chunk bytes ----
    def _read_chunk_bytes(self, loc: EpisodeLoc, chunk_start: int):
        env = self._cache.env(loc.lmdb_path)
        key = str((loc.episode_idx, chunk_start)).encode()
        with env.begin() as txn:
            return txn.get(key)

    def _chunk_start(self, frame_idx):
        return (frame_idx // self.chunk_size) * self.chunk_size

    # ---- 单帧图像解码 ----
    def read_frame_image(self, entry: EpisodeEntry, frame_idx: int):
        """解码并返回第 frame_idx 帧的 RGB 图像 (H, W, 3) uint8；失败返回 None。"""
        loc = entry.locs.get("image")
        if loc is None:
            return None
        cs = self._chunk_start(frame_idx)
        chunk = self._read_chunk_bytes(loc, cs)
        if chunk is None:
            return None
        offset = frame_idx - cs
        return _decode_video_frame(chunk, offset)

    # ---- 单帧动作 ----
    def read_frame_action(self, entry: EpisodeEntry, frame_idx: int):
        loc = entry.locs.get("action")
        if loc is None:
            return None
        cs = self._chunk_start(frame_idx)
        chunk = self._read_chunk_bytes(loc, cs)
        if chunk is None:
            return None
        obj = pickle.loads(chunk)
        offset = frame_idx - cs
        out = {}
        for k, v in obj.items():
            arr = np.asarray(v)
            if offset < arr.shape[0]:
                out[k] = arr[offset]
        return out

    # ---- 单帧 meta_info（状态真值）----
    def read_frame_meta(self, entry: EpisodeEntry, frame_idx: int):
        loc = entry.locs.get("meta_info")
        if loc is None:
            return None
        cs = self._chunk_start(frame_idx)
        chunk = self._read_chunk_bytes(loc, cs)
        if chunk is None:
            return None
        obj = pickle.loads(chunk)
        offset = frame_idx - cs
        # meta_info chunk 是 List[Dict]（逐帧），也可能是 dict-of-list
        if isinstance(obj, list):
            return obj[offset] if offset < len(obj) else None
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    out[k] = v[offset]
                except Exception:
                    out[k] = v
            return out
        return obj

    # ---- 单帧 segmentation ----
    def read_frame_segmentation(self, entry: EpisodeEntry, frame_idx: int):
        loc = entry.locs.get("segmentation")
        if loc is None:
            return None
        cs = self._chunk_start(frame_idx)
        chunk = self._read_chunk_bytes(loc, cs)
        if chunk is None:
            return None
        obj = pickle.loads(chunk)
        offset = frame_idx - cs
        if isinstance(obj, list):
            return obj[offset] if offset < len(obj) else None
        return obj

    def close(self):
        self._cache.close()


def compute_matches(scan: dict, spec: dict) -> np.ndarray:
    """在扫描结果上按 spec 计算匹配帧索引（各条件取 AND）。

    spec 支持的键（全部可选，缺省即不约束）：
        action_keys   list[str]  这些动作键需同时=1（AND）
        action_any    bool       True 时上面改为"任一=1"（OR）
        cam_min       float      视角移动幅度 >= 阈值
        gui           str        'any'|'open'|'inventory'|'none'
        events        list[str]  这些事件/分割标签在该帧出现（任一即可）
    返回升序的帧索引 np.ndarray。
    """
    n = scan["n"]
    mask = np.ones(n, dtype=bool)

    aks = spec.get("action_keys") or []
    if aks:
        acts = scan["actions"]
        sub = [acts[k].astype(bool) for k in aks if k in acts]
        if sub:
            combined = np.zeros(n, dtype=bool)
            if spec.get("action_any"):
                for s in sub:
                    combined |= s
            else:
                combined = np.ones(n, dtype=bool)
                for s in sub:
                    combined &= s
            mask &= combined

    cam_min = spec.get("cam_min")
    if cam_min:
        mask &= scan["cam_mag"] >= float(cam_min)

    gui = spec.get("gui") or "any"
    if gui == "open":
        mask &= scan["gui_open"]
    elif gui == "inventory":
        mask &= scan["gui_inv"]
    elif gui == "none":
        mask &= ~scan["gui_open"]

    want_events = spec.get("events") or []
    if want_events:
        want = set(want_events)
        ev_mask = np.zeros(n, dtype=bool)
        events = scan["events"]
        seg_events = scan["seg_events"]
        for i in range(n):
            if (events[i] & want) or (seg_events[i] & want):
                ev_mask[i] = True
        mask &= ev_mask

    return np.nonzero(mask)[0]


# ---------- 视频 chunk 解码 ----------
# image chunk 是一段 H.264/mp4 字节流（一个 chunk 内 chunk_size 帧）。
# 解码单帧：顺序 decode 到目标 offset。chunk 很短（~32 帧），整块解码开销可接受。


def _decode_video_frame(chunk_bytes, offset):
    import av  # 延迟导入，避免非必要环境报错

    try:
        with io.BytesIO(chunk_bytes) as buf:
            container = av.open(buf, "r")
            stream = container.streams.video[0]
            i = 0
            target = None
            for frame in container.decode(stream):
                if i == offset:
                    target = frame.to_ndarray(format="rgb24")
                    break
                i += 1
            container.close()
            return target
    except Exception:
        return None


def decode_video_chunk_all(chunk_bytes):
    """解码整个 chunk 的所有帧，返回 (T, H, W, 3) uint8 RGB。"""
    import av

    frames = []
    with io.BytesIO(chunk_bytes) as buf:
        container = av.open(buf, "r")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
    return np.asarray(frames) if frames else None
