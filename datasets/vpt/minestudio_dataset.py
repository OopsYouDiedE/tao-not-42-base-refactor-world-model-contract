"""直接读取 MineStudio v1.1 的图像与动作 LMDB 训练窗口。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import pickle
from typing import Any

import av
import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.vpt.video_dataset import VPT_KEYS


_MINESTUDIO_OF_VPT = {
    "key_w": "forward",
    "key_a": "left",
    "key_s": "back",
    "key_d": "right",
    "key_space": "jump",
    "key_sneak": "sneak",
    "key_sprint": "sprint",
    "key_attack": "attack",
    "key_use": "use",
    "key_drop": "drop",
    "key_inventory": "inventory",
    **{f"key_hotbar.{index}": f"hotbar.{index}" for index in range(1, 10)},
}


@dataclass(frozen=True)
class _EpisodeLocation:
    episode: str
    image_database: Path
    image_episode_index: int
    image_chunk_size: int
    action_database: Path
    action_episode_index: int
    action_chunk_size: int
    metadata_database: Path
    metadata_episode_index: int
    metadata_chunk_size: int
    num_frames: int


@dataclass(frozen=True)
class _ModalityEpisode:
    database: Path
    episode_index: int
    chunk_size: int
    num_frames: int


def _database_directories(modality_directory: Path) -> list[Path]:
    if not modality_directory.exists():
        return []
    if (modality_directory / "data.mdb").is_file():
        return [modality_directory]
    return sorted(
        path for path in modality_directory.iterdir()
        if path.is_dir() and (path / "data.mdb").is_file()
    )


def _load_pickle(transaction: lmdb.Transaction, key: str) -> Any:
    value = transaction.get(key.encode())
    if value is None:
        raise RuntimeError(f"MineStudio LMDB 缺少元数据键 {key}")
    return pickle.loads(value)


def _scan_modality(modality_directory: Path) -> dict[str, _ModalityEpisode]:
    episodes: dict[str, _ModalityEpisode] = {}
    for database in _database_directories(modality_directory):
        stream = lmdb.open(
            str(database), readonly=True, lock=False, readahead=False,
            max_readers=128, subdir=True,
        )
        try:
            with stream.begin() as transaction:
                chunk_size = int(_load_pickle(transaction, "__chunk_size__"))
                chunk_infos = _load_pickle(transaction, "__chunk_infos__")
            if chunk_size < 1:
                raise RuntimeError(f"非法 MineStudio chunk_size: {chunk_size}")
            for information in chunk_infos:
                episode = str(information["episode"])
                if episode in episodes:
                    raise RuntimeError(f"MineStudio episode 重复: {episode}")
                episodes[episode] = _ModalityEpisode(
                    database=database,
                    episode_index=int(information["episode_idx"]),
                    chunk_size=chunk_size,
                    num_frames=int(information["num_frames"]),
                )
        finally:
            stream.close()
    return episodes


def minestudio_action_vector(
    action: dict[str, np.ndarray],
    camera_max_degrees: float,
) -> torch.Tensor:
    """把 MineStudio 环境动作转换为 ``[T,22]`` VPT 动作。

    MineStudio 相机顺序为 ``[pitch, yaw]`` 且单位为度；项目动作顺序为
    ``[yaw, pitch]``，并按部署端单 tick 最大角度归一化。
    """
    if camera_max_degrees < 1e-4:
        raise ValueError("camera_max_degrees 必须至少为 1e-4")
    camera = np.asarray(action["camera"], dtype=np.float32)
    if camera.ndim != 2 or camera.shape[1] != 2:
        raise RuntimeError("MineStudio camera 必须为 [T,2]")
    result = torch.zeros(camera.shape[0], 2 + len(VPT_KEYS), dtype=torch.float32)
    result[:, 0] = torch.from_numpy(camera[:, 1] / camera_max_degrees).clamp(-1.0, 1.0)
    result[:, 1] = torch.from_numpy(camera[:, 0] / camera_max_degrees).clamp(-1.0, 1.0)
    for index, vpt_key in enumerate(VPT_KEYS):
        source_key = _MINESTUDIO_OF_VPT[vpt_key]
        value = np.asarray(action[source_key], dtype=np.float32).reshape(camera.shape[0], -1)
        result[:, 2 + index] = torch.from_numpy(value[:, 0]).clamp(0.0, 1.0)
    return result


class MineStudioLMDBDataset(Dataset):
    """从一个课程阶段的本地图像/动作 LMDB 返回固定连续窗口。

    每个图像分片只与全部动作和元数据库求 episode 交集，因此不要求三种模态的
    分片编号对齐。元数据中的 GUI 状态和绝对光标位置只作为辅助监督目标，不得作为
    策略输入；GUI 中的相对光标移动仍由 ``camera`` 两轴动作监督。LMDB 句柄在
    DataLoader worker 内惰性打开，不跨进程序列化。
    """

    def __init__(
        self,
        data_directory: str | Path,
        sequence_length: int,
        image_size: tuple[int, int],
        task_text: str,
        camera_max_degrees: float,
        stride: int | None = None,
        split: str = "train",
        validation_fraction: float = 0.02,
        seed: int = 0,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length 必须至少为 2")
        if image_size[0] < 1 or image_size[1] < 1:
            raise ValueError("image_size 高宽必须大于零")
        if split not in {"train", "validation", "all"}:
            raise ValueError("split 必须为 train、validation 或 all")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction 必须位于 [0,1)")
        self.data_directory = Path(data_directory).resolve()
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.task_text = task_text
        self.camera_max_degrees = camera_max_degrees
        self.stride = stride or sequence_length
        if self.stride < 1:
            raise ValueError("stride 必须大于零")

        self.image_shards = tuple(
            path.name for path in _database_directories(self.data_directory / "image")
        )
        self.action_shards = tuple(
            path.name for path in _database_directories(self.data_directory / "action")
        )
        self.metadata_shards = tuple(
            path.name for path in _database_directories(self.data_directory / "meta_info")
        )
        images = _scan_modality(self.data_directory / "image")
        actions = _scan_modality(self.data_directory / "action")
        metadata = _scan_modality(self.data_directory / "meta_info")
        if not images:
            raise RuntimeError(f"{self.data_directory} 没有已下载的 image LMDB")
        if not actions:
            raise RuntimeError(f"{self.data_directory} 没有已下载的 action LMDB")
        if not metadata:
            raise RuntimeError(f"{self.data_directory} 没有已下载的 meta_info LMDB")

        self.episodes: list[_EpisodeLocation] = []
        self.cumulative_windows: list[int] = []
        self.total_frames = 0
        total_windows = 0
        for episode in sorted(images.keys() & actions.keys() & metadata.keys()):
            image = images[episode]
            action = actions[episode]
            episode_metadata = metadata[episode]
            if len({image.num_frames, action.num_frames, episode_metadata.num_frames}) != 1:
                continue
            digest = hashlib.sha256(f"{seed}:{episode}".encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") / float(2**64)
            is_validation = bucket < validation_fraction
            if split == "train" and is_validation:
                continue
            if split == "validation" and not is_validation:
                continue
            windows = max(
                0,
                (image.num_frames - sequence_length) // self.stride + 1,
            )
            if windows == 0:
                continue
            self.episodes.append(_EpisodeLocation(
                episode=episode,
                image_database=image.database,
                image_episode_index=image.episode_index,
                image_chunk_size=image.chunk_size,
                action_database=action.database,
                action_episode_index=action.episode_index,
                action_chunk_size=action.chunk_size,
                metadata_database=episode_metadata.database,
                metadata_episode_index=episode_metadata.episode_index,
                metadata_chunk_size=episode_metadata.chunk_size,
                num_frames=image.num_frames,
            ))
            total_windows += windows
            self.cumulative_windows.append(total_windows)
            self.total_frames += image.num_frames
        if not self.episodes:
            raise RuntimeError("图像与动作 LMDB 没有可用于该 split 的共同 episode")
        self._streams: dict[Path, lmdb.Environment] = {}

    def __len__(self) -> int:
        return self.cumulative_windows[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_streams"] = {}
        return state

    def _stream(self, path: Path) -> lmdb.Environment:
        stream = self._streams.get(path)
        if stream is None:
            stream = lmdb.open(
                str(path), readonly=True, lock=False, readahead=False,
                max_readers=128, subdir=True,
            )
            self._streams[path] = stream
        return stream

    def _read_chunks(
        self,
        database: Path,
        episode_index: int,
        chunk_size: int,
        start: int,
        length: int,
    ) -> tuple[list[bytes], int]:
        first_chunk = start // chunk_size * chunk_size
        final_frame = start + length - 1
        final_chunk = final_frame // chunk_size * chunk_size
        chunks = []
        stream = self._stream(database)
        with stream.begin() as transaction:
            for chunk_start in range(first_chunk, final_chunk + 1, chunk_size):
                key = str((episode_index, chunk_start)).encode()
                value = transaction.get(key)
                if value is None:
                    raise RuntimeError(f"MineStudio LMDB 缺少 chunk {key!r}")
                chunks.append(value)
        return chunks, start - first_chunk

    def _decode_images(
        self,
        chunks: list[bytes],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        frames = []
        source_size = None
        for chunk in chunks:
            with av.open(io.BytesIO(chunk), mode="r") as container:
                for frame in container.decode(video=0):
                    image = frame.to_ndarray(format="rgb24")
                    frame_source_size = image.shape[:2]
                    if source_size is None:
                        source_size = frame_source_size
                    elif source_size != frame_source_size:
                        raise RuntimeError("同一 MineStudio 窗口的源图像尺寸不一致")
                    if image.shape[:2] != self.image_size:
                        image = cv2.resize(
                            image, (self.image_size[1], self.image_size[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    frames.append(image)
        if not frames:
            raise RuntimeError("MineStudio 图像 chunk 无法解码")
        if source_size is None:
            raise RuntimeError("MineStudio 图像 chunk 缺少源尺寸")
        return np.stack(frames), source_size

    @staticmethod
    def _decode_actions(chunks: list[bytes]) -> dict[str, np.ndarray]:
        decoded = [pickle.loads(chunk) for chunk in chunks]
        keys = decoded[0].keys()
        return {
            key: np.concatenate([np.asarray(chunk[key]) for chunk in decoded], axis=0)
            for key in keys
        }

    @staticmethod
    def _decode_metadata(
        chunks: list[bytes],
        bias: int,
        length: int,
        source_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """返回仅供辅助监督或诊断的 GUI 与绝对光标目标。"""
        frames = []
        for chunk in chunks:
            frames.extend(pickle.loads(chunk))
        window = frames[bias:bias + length]
        if len(window) != length:
            raise RuntimeError("MineStudio meta_info chunk 长度与索引不一致")
        source_height, source_width = source_size
        cursor_xy = torch.zeros(length, 2, dtype=torch.float32)
        gui_open = torch.tensor(
            [bool(frame.get("isGuiOpen", False)) for frame in window],
            dtype=torch.bool,
        )
        gui_inventory = torch.tensor(
            [bool(frame.get("isGuiInventory", False)) for frame in window],
            dtype=torch.bool,
        )
        cursor_valid = torch.zeros(length, dtype=torch.bool)
        for index, frame in enumerate(window):
            if not bool(gui_open[index]):
                continue
            cursor_x = float(frame.get("cursor_x", float("nan")))
            cursor_y = float(frame.get("cursor_y", float("nan")))
            if not np.isfinite(cursor_x) or not np.isfinite(cursor_y):
                continue
            if not 0.0 <= cursor_x < source_width or not 0.0 <= cursor_y < source_height:
                continue
            cursor_xy[index, 0] = cursor_x / max(source_width - 1, 1)
            cursor_xy[index, 1] = cursor_y / max(source_height - 1, 1)
            cursor_valid[index] = True
        return {
            "cursor_target_xy": cursor_xy,
            "cursor_target_valid": cursor_valid,
            "gui_open_target": gui_open,
            "gui_inventory_target": gui_inventory,
        }

    def _locate_window(self, index: int) -> tuple[_EpisodeLocation, int]:
        episode_index = bisect_right(self.cumulative_windows, index)
        previous = self.cumulative_windows[episode_index - 1] if episode_index else 0
        location = self.episodes[episode_index]
        return location, (index - previous) * self.stride

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        location, start = self._locate_window(index)
        metadata_chunks, metadata_bias = self._read_chunks(
            location.metadata_database, location.metadata_episode_index,
            location.metadata_chunk_size, start, self.sequence_length,
        )
        image_chunks, image_bias = self._read_chunks(
            location.image_database, location.image_episode_index,
            location.image_chunk_size, start, self.sequence_length,
        )
        action_chunks, action_bias = self._read_chunks(
            location.action_database, location.action_episode_index,
            location.action_chunk_size, start, self.sequence_length - 1,
        )
        decoded_images, source_size = self._decode_images(image_chunks)
        images = decoded_images[image_bias:image_bias + self.sequence_length]
        metadata = self._decode_metadata(
            metadata_chunks, metadata_bias, self.sequence_length, source_size,
        )
        raw_actions = self._decode_actions(action_chunks)
        actions = minestudio_action_vector(
            {key: value[action_bias:action_bias + self.sequence_length - 1]
             for key, value in raw_actions.items()},
            self.camera_max_degrees,
        )
        if len(images) != self.sequence_length or len(actions) != self.sequence_length - 1:
            raise RuntimeError("MineStudio chunk 解码长度与索引元数据不一致")
        return {
            "img": torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),
            "act_seq": actions.unsqueeze(1),
            "act_agg": actions,
            "dt": torch.ones(self.sequence_length - 1, dtype=torch.float32),
            "task_text": self.task_text,
            "t_vec": torch.arange(self.sequence_length, dtype=torch.float32) / 20.0,
            "episode": location.episode,
            **metadata,
        }
