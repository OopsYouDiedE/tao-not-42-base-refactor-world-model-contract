"""把槽位产出的帧编码成浏览器可读的图像，并保留可回放的帧历史。

这里只做图像编码、缓存与分发，不决定何时前进一个 tick。帧由执行循环逐 tick 推入，
每帧记下它属于哪个 tick，因此界面既能看实时画面，也能逐 tick 回放已经执行过的帧。

推流使用异步生成器：`publish` 发生在执行线程，读取发生在事件循环上，两侧只通过一个
递增的 `revision` 交换信息，不做跨线程唤醒，也不占用 uvicorn 的同步线程池。
"""

from __future__ import annotations

import asyncio
import io
from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np
from PIL import Image

BOUNDARY = "craftground-frame"
DEFAULT_HISTORY = 4096


def encode_frame(frame: object, *, quality: int = 70) -> bytes:
    """把一帧观察转成 JPEG 字节。

    CraftGround 的 `raw` 模式给出 numpy RGB；`zerocopy_torch` 给出 CUDA 张量，
    需要先取回主机内存。两者都在这里归一，界面层不需要知道编码模式。
    """
    array = frame if isinstance(frame, np.ndarray) else frame.detach().cpu().numpy()  # type: ignore[attr-defined]
    array = np.ascontiguousarray(array)
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


@dataclass(frozen=True)
class FrameRecord:
    """一帧画面及它对应的环境 tick。"""

    tick: int | None
    jpeg: bytes


class FrameStream:
    """单个槽位的帧历史；多个浏览器窗口可以同时看实时画面或回放。"""

    def __init__(self, *, history: int = DEFAULT_HISTORY) -> None:
        if history < 1:
            raise ValueError("history 必须大于零")
        self._lock = Lock()
        self._records: deque[FrameRecord] = deque(maxlen=history)
        self._revision = 0

    @property
    def revision(self) -> int:
        """已推入的帧总数；用于判断是否有新画面。"""
        with self._lock:
            return self._revision

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def publish(self, frame: object, *, tick: int | None = None) -> None:
        record = FrameRecord(tick, encode_frame(frame))
        with self._lock:
            self._records.append(record)
            self._revision += 1

    def clear(self) -> None:
        """清空回放历史，并通知实时流观察者历史边界已经变化。"""
        with self._lock:
            self._records.clear()
            self._revision += 1

    def latest(self) -> FrameRecord | None:
        with self._lock:
            return self._records[-1] if self._records else None

    def at(self, index: int) -> FrameRecord | None:
        """按历史下标取帧；负数从末尾数起。"""
        with self._lock:
            if not self._records:
                return None
            if -len(self._records) <= index < len(self._records):
                return self._records[index]
            return None

    def at_tick(self, tick: int) -> FrameRecord | None:
        """取某个环境 tick 的帧；同一 tick 有多帧时返回最后一帧。"""
        with self._lock:
            for record in reversed(self._records):
                if record.tick == tick:
                    return record
            return None

    def ticks(self) -> tuple[int | None, ...]:
        with self._lock:
            return tuple(record.tick for record in self._records)

    def snapshot(self) -> tuple[int, int, int | None]:
        """返回 (revision, 历史帧数, 最新帧的 tick)，供界面构建回放控件。"""
        with self._lock:
            latest = self._records[-1].tick if self._records else None
            return self._revision, len(self._records), latest

    async def multipart(self, *, fps: float = 20.0):
        """异步产出 multipart 分片；只在出现新帧时推送。

        运行在事件循环上而不是同步线程池里，因此并发的浏览器窗口和状态轮询不会把它
        饿死。`fps` 只是检查新帧的上限频率，不会凭空补帧。
        """
        if fps <= 0:
            raise ValueError("fps 必须大于零")
        interval = 1.0 / fps
        seen = -1
        while True:
            revision, _, _ = self.snapshot()
            if revision != seen:
                seen = revision
                record = self.latest()
                if record is not None:
                    yield (
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(record.jpeg)}\r\n\r\n".encode()
                        + record.jpeg
                        + b"\r\n"
                    )
            await asyncio.sleep(interval)
