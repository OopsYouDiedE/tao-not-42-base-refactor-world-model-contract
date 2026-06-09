"""通用数据基础设施(与具体数据集无关)。

decode_uint16_range / maybe_pin_memory / numpy_to_pinned_tensor — 张量解码/锁页助手。
to_gpu / pad_instances / instance_presence — 通用批次张量操作。
CUDAPrefetcher — 后台 CUDA stream 双缓冲预取器，与训练计算时间重叠。
BaseSource / MixedSource — 数据源抽象接口与加权混合采样器。
"""
import queue
import random
import threading
import atexit
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "decode_uint16_range", "maybe_pin_memory", "numpy_to_pinned_tensor",
    "to_gpu", "pad_instances", "instance_presence",
    "CUDAPrefetcher",
    "BaseSource", "MixedSource",
]


# ==========================================================================
# 张量解码 / 锁页助手
# ==========================================================================

def decode_uint16_range(encoded, value_range):
    encoded = encoded.astype(np.float32)
    minv, maxv = np.asarray(value_range, dtype=np.float32)
    return encoded / 65535.0 * (maxv - minv) + minv


def maybe_pin_memory(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.pin_memory()
    return tensor


def numpy_to_pinned_tensor(value):
    """numpy/标量 → 锁页 tensor；dict/None 直接返回 None。"""
    if value is None or isinstance(value, dict):
        return None
    if isinstance(value, torch.Tensor):
        return maybe_pin_memory(value)
    if isinstance(value, np.ndarray):
        return maybe_pin_memory(torch.from_numpy(value))
    if np.isscalar(value):
        return maybe_pin_memory(torch.as_tensor(value))
    return None


# ==========================================================================
# 通用批次张量操作
# ==========================================================================

def to_gpu(val, device, dtype=None):
    """tensor 或 list[tensor] → device，可选 dtype 转换。"""
    if isinstance(val, (list, tuple)):
        val = torch.stack(val)
    val = val.to(device, non_blocking=True)
    return val.to(dtype) if dtype is not None else val


def pad_instances(values, device):
    """将可变长度实例列表 padding 至统一长度，保持 batch 轴完整。

    TFDS 有时会在单个样本上忽略可选实例元数据（None）；
    此函数用零填充缺失条目，避免 batch 维度缩小导致实例 ID 错位。
    """
    if isinstance(values, torch.Tensor):
        return values.to(device, non_blocking=True)
    if not values:
        return None
    present = [x for x in values if x is not None]
    if not present:
        return None
    max_len = max(len(x) for x in present)
    exemplar = present[0]
    padded = []
    for x in values:
        if x is None:
            fill_shape = (max_len, *exemplar.shape[1:])
            padded.append(torch.zeros(fill_shape, dtype=exemplar.dtype))
            continue
        pad_dims = []
        for _ in range(x.dim() - 1):
            pad_dims.extend([0, 0])
        pad_dims.extend([0, max_len - len(x)])
        padded.append(F.pad(x, tuple(pad_dims)))
    return torch.stack(padded).to(device, non_blocking=True)


def instance_presence(values, device):
    """返回 bool tensor：batch 中每项是否有非 None 的值。"""
    if isinstance(values, torch.Tensor):
        return torch.ones(values.shape[0], device=device, dtype=torch.bool)
    if not values:
        return None
    return torch.tensor([x is not None for x in values], device=device, dtype=torch.bool)


# ==========================================================================
# 异步双缓冲预取器
# ==========================================================================

class CUDAPrefetcher:
    """后台 CUDA stream 双缓冲预取器，与训练计算时间重叠。

    process_fn(batch_raw, device, target_size) → batch_gpu
    传 None 则跳过 GPU 预处理，直接透传原始批次。
    queue maxsize=2：后台最多超前准备 2 个批次。
    """

    def __init__(self, buffer, device, process_fn=None, target_size=256, wait_timeout_sec=180):
        self.buffer = buffer
        self.device = device
        self.target_size = target_size
        self.wait_timeout_sec = wait_timeout_sec
        self.process_fn = process_fn
        self._ready = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def _prefetch_loop(self):
        stream = torch.cuda.Stream(device=self.device)
        while not self._stop.is_set():
            try:
                batch_raw = self.buffer.get_batch()
            except Exception as e:
                self._ready.put((e, None))
                return
            if batch_raw is None:
                continue
            try:
                with torch.cuda.stream(stream):
                    if self.process_fn is not None:
                        batch_gpu = self.process_fn(batch_raw, self.device, self.target_size)
                    else:
                        batch_gpu = batch_raw
                event = torch.cuda.Event()
                event.record(stream)
                self._ready.put((batch_gpu, event))
            except Exception as e:
                self._ready.put((e, None))
                return

    def next(self):
        try:
            item, event = self._ready.get(timeout=self.wait_timeout_sec)
        except queue.Empty:
            raise TimeoutError(f"[Prefetcher] 等待预取批次超时 ({self.wait_timeout_sec}s)")
        if isinstance(item, Exception):
            raise item
        if event is not None:
            event.synchronize()
        return item

    def stop(self):
        self._stop.set()
        try:
            while True:
                self._ready.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=3.0)


# ==========================================================================
# 数据源抽象
# ==========================================================================

class BaseSource(ABC):
    """通用数据源接口。子类无限生成契约格式的 batch。"""
    game_id: int
    n_actions: int

    @abstractmethod
    def batches(self, batch_size: int, chunk_len: int) -> Iterator[dict]:
        """无限生成 batch。"""


class MixedSource:
    """按权重交替采样多个 BaseSource 实例。"""

    def __init__(self, sources: list[tuple[BaseSource, float]],
                 batch_size: int = 16, chunk_len: int = 16):
        self.sources = sources
        total = sum(w for _, w in sources)
        self.weights = [w / total for _, w in sources]
        self._iters = [s.batches(batch_size, chunk_len) for s, _ in sources]

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        idx = random.choices(range(len(self.sources)), weights=self.weights)[0]
        return next(self._iters[idx])
