"""验证 MineStudio LMDB 的 episode 对齐、视频解码与动作转换。"""

import io
from pathlib import Path
import pickle

import av
import lmdb
import numpy as np

from datasets.vpt.minestudio_dataset import MineStudioLMDBDataset


def _video_chunk(frame_count: int) -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            image = np.full((16, 16, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _write_database(
    path: Path,
    chunk: bytes,
    frame_count: int,
) -> None:
    path.mkdir(parents=True)
    stream = lmdb.open(str(path), map_size=16 * 1024 * 1024, subdir=True)
    with stream.begin(write=True) as transaction:
        transaction.put(b"__chunk_size__", pickle.dumps(frame_count))
        transaction.put(b"__chunk_infos__", pickle.dumps([{
            "episode": "episode-a",
            "episode_idx": 0,
            "num_frames": frame_count,
        }]))
        transaction.put(b"__num_episodes__", pickle.dumps(1))
        transaction.put(b"__num_total_frames__", pickle.dumps(frame_count))
        transaction.put(str((0, 0)).encode(), chunk)
    stream.close()


def test_minestudio_dataset_reads_aligned_window(tmp_path: Path):
    frame_count = 4
    _write_database(
        tmp_path / "image" / "part-0",
        _video_chunk(frame_count),
        frame_count,
    )
    action = {
        "camera": np.tile(np.array([[1.0, 2.0]], dtype=np.float32), (frame_count, 1)),
        "attack": np.zeros(frame_count, dtype=np.uint8),
        "forward": np.ones(frame_count, dtype=np.uint8),
        "back": np.zeros(frame_count, dtype=np.uint8),
        "left": np.zeros(frame_count, dtype=np.uint8),
        "right": np.zeros(frame_count, dtype=np.uint8),
        "jump": np.zeros(frame_count, dtype=np.uint8),
        "sneak": np.zeros(frame_count, dtype=np.uint8),
        "sprint": np.zeros(frame_count, dtype=np.uint8),
        "use": np.zeros(frame_count, dtype=np.uint8),
        "drop": np.zeros(frame_count, dtype=np.uint8),
        "inventory": np.zeros(frame_count, dtype=np.uint8),
        **{f"hotbar.{index}": np.zeros(frame_count, dtype=np.uint8)
           for index in range(1, 10)},
    }
    _write_database(
        tmp_path / "action" / "part-9",
        pickle.dumps(action),
        frame_count,
    )
    _write_database(
        tmp_path / "meta_info" / "part-4",
        pickle.dumps([
            {"isGuiOpen": False, "isGuiInventory": False}
            for _ in range(frame_count)
        ]),
        frame_count,
    )

    dataset = MineStudioLMDBDataset(
        tmp_path, sequence_length=3, image_size=(16, 16),
        task_text="test task", camera_max_degrees=18.0,
        split="all",
    )
    sample = dataset[0]

    assert sample["img"].shape == (3, 3, 16, 16)
    assert sample["act_agg"].shape == (2, 22)
    assert np.isclose(float(sample["act_agg"][0, 0]), 2.0 / 18.0)
    assert np.isclose(float(sample["act_agg"][0, 1]), 1.0 / 18.0)
    assert float(sample["act_agg"][0, 2]) == 1.0
    assert sample["task_text"] == "test task"


def test_gui_windows_are_detected_from_metadata():
    metadata = pickle.dumps([
        {"isGuiOpen": False},
        {"isGuiOpen": True},
        {"isGuiOpen": False},
        {"isGuiOpen": False},
    ])
    assert MineStudioLMDBDataset._contains_gui([metadata], 0, 4)
