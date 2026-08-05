"""帧编码、历史与分发；不需要 CraftGround。"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from trajectory_visualization.frame_stream import (
    BOUNDARY,
    FrameRecord,
    FrameStream,
    encode_frame,
)


def _frame(value: int) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


def test_encode_frame_produces_jpeg_bytes() -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    jpeg = encode_frame(frame)

    assert jpeg.startswith(b"\xff\xd8")
    assert jpeg.endswith(b"\xff\xd9")


def test_publish_tracks_revisions_and_latest_frame() -> None:
    stream = FrameStream()
    assert stream.latest() is None
    assert stream.revision == 0

    frame = _frame(128)
    stream.publish(frame, tick=7)

    assert stream.revision == 1
    assert stream.latest() == FrameRecord(7, encode_frame(frame))

    stream.publish(_frame(0))
    assert stream.revision == 2
    assert stream.count == 2


def test_history_is_bounded_and_drops_the_oldest_frames() -> None:
    stream = FrameStream(history=2)
    for tick in range(3):
        stream.publish(_frame(tick * 10), tick=tick)

    # revision 记录推入总数，历史只保留最后两帧。
    assert stream.revision == 3
    assert stream.count == 2
    assert stream.ticks() == (1, 2)
    assert stream.at_tick(0) is None


def test_history_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        FrameStream(history=0)


def test_frames_are_addressable_by_index_and_tick() -> None:
    stream = FrameStream()
    stream.publish(_frame(10), tick=0)
    stream.publish(_frame(20), tick=1)
    # 同一 tick 推两帧时，按 tick 取应拿到最后一帧。
    stream.publish(_frame(30), tick=1)

    assert stream.at(0) == FrameRecord(0, encode_frame(_frame(10)))
    assert stream.at(-1) == FrameRecord(1, encode_frame(_frame(30)))
    assert stream.at_tick(1) == FrameRecord(1, encode_frame(_frame(30)))
    assert stream.at(3) is None
    assert stream.at(-4) is None
    assert stream.at_tick(9) is None


def test_snapshot_reports_revision_count_and_latest_tick() -> None:
    stream = FrameStream()
    assert stream.snapshot() == (0, 0, None)

    stream.publish(_frame(10), tick=4)

    assert stream.snapshot() == (1, 1, 4)


def test_multipart_chunk_carries_boundary_and_jpeg() -> None:
    stream = FrameStream()
    stream.publish(_frame(128))

    async def first_chunk() -> bytes:
        return await anext(stream.multipart(fps=200.0))

    chunk = asyncio.run(first_chunk())

    assert chunk.startswith(f"--{BOUNDARY}".encode())
    assert b"Content-Type: image/jpeg" in chunk
    assert b"\xff\xd8" in chunk


def test_multipart_only_yields_when_a_new_frame_arrives() -> None:
    stream = FrameStream()
    stream.publish(_frame(0), tick=0)

    async def two_chunks() -> list[bytes]:
        chunks = stream.multipart(fps=200.0)
        first = await anext(chunks)
        pending = asyncio.ensure_future(anext(chunks))
        # 没有新帧时生成器必须继续等待，不能凭空补帧。
        await asyncio.sleep(0.05)
        assert not pending.done()
        stream.publish(_frame(200), tick=1)
        return [first, await asyncio.wait_for(pending, timeout=2.0)]

    first, second = asyncio.run(two_chunks())

    assert encode_frame(_frame(0)) in first
    assert encode_frame(_frame(200)) in second


def test_multipart_requires_a_positive_fps() -> None:
    async def drain() -> None:
        await anext(FrameStream().multipart(fps=0.0))

    with pytest.raises(ValueError):
        asyncio.run(drain())
