"""验证 MineStudio LMDB 的 episode 对齐、视频解码与动作转换。"""

import io
from pathlib import Path
import pickle

import av
import lmdb
import numpy as np

from datasets.minestudio.dataset import MineStudioLMDBDataset


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


def _action_chunk(frame_count: int) -> bytes:
    """返回覆盖 VPT 键位的测试动作 pickle。"""
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
    return pickle.dumps(action)


def test_minestudio_dataset_reads_aligned_window(tmp_path: Path):
    frame_count = 4
    _write_database(
        tmp_path / "image" / "part-0",
        _video_chunk(frame_count),
        frame_count,
    )
    _write_database(
        tmp_path / "action" / "part-9",
        _action_chunk(frame_count),
        frame_count,
    )
    _write_database(
        tmp_path / "meta_info" / "part-4",
        pickle.dumps([
            {"isGuiOpen": False, "isGuiInventory": False,
             "cursor_x": 0, "cursor_y": 0},
            {"isGuiOpen": True, "isGuiInventory": True,
             "cursor_x": 8, "cursor_y": 4},
            {"isGuiOpen": True, "isGuiInventory": False,
             "cursor_x": 15, "cursor_y": 15},
            {"isGuiOpen": False, "isGuiInventory": False,
             "cursor_x": 0, "cursor_y": 0},
        ]),
        frame_count,
    )

    dataset = MineStudioLMDBDataset(
        tmp_path, sequence_length=3, image_size=(16, 16),
        task_text="test task", camera_max_degrees=18.0,
        split="all",
        include_metadata_targets=True,
    )
    sample = dataset[0]

    assert sample["img"].shape == (3, 3, 16, 16)
    assert sample["act_agg"].shape == (2, 22)
    assert np.isclose(float(sample["act_agg"][0, 0]), 2.0 / 18.0)
    assert np.isclose(float(sample["act_agg"][0, 1]), 1.0 / 18.0)
    assert float(sample["act_agg"][0, 2]) == 1.0
    assert sample["task_text"] == "test task"
    assert sample["gui_open_target"].tolist() == [False, True, True]
    assert sample["gui_inventory_target"].tolist() == [False, True, False]
    assert sample["cursor_target_valid"].tolist() == [False, True, True]
    assert np.isclose(float(sample["cursor_target_xy"][1, 0]), 8.0 / 15.0)
    assert np.isclose(float(sample["cursor_target_xy"][1, 1]), 4.0 / 15.0)
    assert sample["cursor_target_xy"][2].tolist() == [1.0, 1.0]


def test_minestudio_dataset_does_not_require_metadata_by_default(tmp_path: Path):
    frame_count = 4
    _write_database(
        tmp_path / "image" / "part-0",
        _video_chunk(frame_count),
        frame_count,
    )
    _write_database(
        tmp_path / "action" / "part-9",
        _action_chunk(frame_count),
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
    assert "cursor_target_xy" not in sample
    assert dataset.metadata_shards == ()
