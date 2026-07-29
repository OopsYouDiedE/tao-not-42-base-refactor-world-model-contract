"""MineStudio LMDB 分片的模态读取与多模态 episode 对齐。

对外接口：
    EpisodeInfo — 一条 episode 在某模态分片中的定位信息。
    LMDBModalityReader — 单模态 LMDB 读取：列 episode、按帧区间取数据。
    TrajectoryReader — 多模态按 episode 名对齐读取。

MineStudio v1.1.0 布局：每个模态一组 LMDB 分片，分片内以
``str((episode_idx, chunk_id))`` 为 key，value 是固定 ``__chunk_size__`` 帧的一段数据。
``action`` 的 value 是字段到数组的 pickle 字典，``meta_info`` 是逐帧字典列表，
``image`` 的 value 是视频字节流。
不同模态的分片切分边界不同，跨模态必须按 episode 名对齐，不能按分片号配对。
"""

from __future__ import annotations

import io
import pickle
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import av
import cv2
import lmdb
import numpy as np

ModalityName = Literal["image", "action", "meta_info"]


@dataclass(frozen=True)
class EpisodeInfo:
    """一条 episode 在某模态分片中的定位信息。

    Attributes
    ----------
    episode : str
        episode 名（跨模态对齐的唯一键）。
    episode_index : int
        该分片内的 episode 序号，参与 LMDB key 构造。
    num_frames : int
        该 episode 在本模态的总帧数。
    part_directory : Path
        所属分片目录。
    """

    episode: str
    episode_index: int
    num_frames: int
    part_directory: Path


class LMDBModalityReader:
    """单模态 LMDB 分片组的读取器。

    Parameters
    ----------
    part_directories : list of Path
        同一模态下的分片目录列表，每个目录含 ``data.mdb``。
    modality : {"image", "action", "meta_info"}
        模态名，决定 value 的解码方式。
    frame_width, frame_height : int
        仅 ``image`` 模态使用：解码后缩放到的尺寸，单位像素。
    decode_workers : int
        仅 ``image`` 模态使用：单个视频块的并行解码线程数。
    cache_size : int
        已解码块的 LRU 缓存容量。一个块含 ``chunk_size`` 帧（通常 32），顺序扫描时
        同一块会被相邻窗口反复命中，缓存能省掉大量重复解码。
        ``image`` 模态单块约 ``chunk_size × H × W × 3`` 字节（32×224×224×3 ≈ 4.6MB），
        容量要按内存预算设小；``action`` / ``meta_info`` 块很小，可放宽。
    """

    def __init__(
        self,
        part_directories: list[Path],
        modality: ModalityName,
        frame_width: int = 224,
        frame_height: int = 224,
        decode_workers: int = 4,
        cache_size: int = 32,
    ) -> None:
        if not part_directories:
            raise ValueError("part_directories 不能为空")
        if cache_size < 0:
            raise ValueError("cache_size 不能为负")
        self.modality = modality
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.decode_workers = decode_workers
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[Path, int, int], Any] = OrderedDict()
        self._environments: dict[Path, Any] = {}
        self._chunk_size: int | None = None
        self._episodes: dict[str, EpisodeInfo] = {}
        for directory in part_directories:
            self._register_part(Path(directory))

    def _register_part(self, directory: Path) -> None:
        """打开一个分片，读取元数据并登记其中的 episode。"""
        if not (directory / "data.mdb").is_file():
            raise FileNotFoundError(f"分片缺少 data.mdb：{directory}")
        environment = lmdb.open(
            str(directory), readonly=True, lock=False, readahead=False, max_readers=128,
        )
        self._environments[directory] = environment
        with environment.begin(write=False) as transaction:
            chunk_size = pickle.loads(transaction.get(b"__chunk_size__"))
            chunk_infos = pickle.loads(transaction.get(b"__chunk_infos__"))
        if self._chunk_size is None:
            self._chunk_size = int(chunk_size)
        elif self._chunk_size != int(chunk_size):
            raise ValueError(
                f"分片 chunk_size 不一致：{self._chunk_size} vs {chunk_size}（{directory}）",
            )
        for info in chunk_infos:
            self._episodes[info["episode"]] = EpisodeInfo(
                episode=info["episode"],
                episode_index=int(info["episode_idx"]),
                num_frames=int(info["num_frames"]),
                part_directory=directory,
            )

    @property
    def chunk_size(self) -> int:
        """每个 LMDB value 覆盖的帧数。"""
        if self._chunk_size is None:
            raise RuntimeError("尚未登记任何分片")
        return self._chunk_size

    def episode_names(self) -> list[str]:
        """本模态可读的全部 episode 名，按名字排序。"""
        return sorted(self._episodes)

    def episode_info(self, episode: str) -> EpisodeInfo:
        """取某条 episode 的定位信息。"""
        try:
            return self._episodes[episode]
        except KeyError:
            raise KeyError(f"模态 {self.modality} 中没有 episode {episode!r}") from None

    def close(self) -> None:
        """关闭全部 LMDB 环境并清空缓存。"""
        for environment in self._environments.values():
            environment.close()
        self._environments.clear()
        self._cache.clear()

    def _decode_image_chunk(self, chunk: bytes) -> np.ndarray:
        """把一段视频字节流解码成帧数组，shape (T, H, W, 3)，dtype uint8，RGB。"""

        def convert_and_resize(frame: Any) -> np.ndarray:
            array = frame.to_ndarray(format="rgb24")
            if array.shape[0] != self.frame_height or array.shape[1] != self.frame_width:
                array = cv2.resize(
                    array,
                    (self.frame_width, self.frame_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            return array

        futures = []
        with io.BytesIO(chunk) as buffer:
            with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
                container = av.open(buffer, "r")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        futures.append(executor.submit(convert_and_resize, frame))
                frames = [future.result() for future in futures]
                container.close()
        if not frames:
            raise ValueError("视频块解码得到零帧")
        return np.stack(frames, axis=0)

    def _cached_chunk(self, info: EpisodeInfo, chunk_frame: int) -> Any:
        """取缓存中的已解码块，未命中返回 None，命中则刷新为最近使用。"""
        if self.cache_size == 0:
            return None
        key = (info.part_directory, info.episode_index, chunk_frame)
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _store_chunk(self, info: EpisodeInfo, chunk_frame: int, chunk: Any) -> None:
        """把已解码块写入缓存，超容量时淘汰最久未使用的一项。"""
        if self.cache_size == 0:
            return
        self._cache[(info.part_directory, info.episode_index, chunk_frame)] = chunk
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _decode_chunk(self, chunk: bytes) -> Any:
        """按模态解码单个 LMDB value。"""
        if self.modality == "image":
            return self._decode_image_chunk(chunk)
        return pickle.loads(chunk)

    def _merge_chunks(self, decoded: list[Any]) -> Any:
        """把连续若干块沿时间轴拼接。"""
        if self.modality == "image":
            return np.concatenate(decoded, axis=0)
        if self.modality == "meta_info":
            return [frame for chunk in decoded for frame in chunk]
        merged: dict[str, list[np.ndarray]] = {}
        for item in decoded:
            for key, value in item.items():
                merged.setdefault(key, []).append(np.asarray(value))
        return {key: np.concatenate(values, axis=0) for key, values in merged.items()}

    def read_frames(self, episode: str, start: int, length: int) -> Any:
        """读取某条 episode 的一段连续帧。

        Parameters
        ----------
        episode : str
            episode 名。
        start : int
            起始帧下标，需 ``>= 0``。
        length : int
            要读取的帧数，需 ``> 0``；越过 episode 末尾会被截断。

        Returns
        -------
        numpy.ndarray or dict
            ``image`` 模态返回 shape (T, H, W, 3)、dtype uint8 的 RGB 帧数组；
            ``action`` 返回字段名 → 长度 T 数组的字典；``meta_info`` 返回长度 T 的
            逐帧字典列表。
            实际 T 为 ``min(length, num_frames - start)``。
        """
        if start < 0:
            raise ValueError("start 不能为负")
        if length <= 0:
            raise ValueError("length 必须大于零")
        info = self.episode_info(episode)
        if start >= info.num_frames:
            raise ValueError(f"start {start} 越过 episode 长度 {info.num_frames}")
        end = min(start + length, info.num_frames)
        # LMDB key 的第二项是该块的**起始帧号**（chunk_size 的整数倍），不是块序号。
        first_chunk_frame = (start // self.chunk_size) * self.chunk_size
        last_chunk_frame = ((end - 1) // self.chunk_size) * self.chunk_size
        wanted = list(range(first_chunk_frame, last_chunk_frame + 1, self.chunk_size))
        decoded: list[Any] = [self._cached_chunk(info, frame) for frame in wanted]
        missing = [frame for frame, chunk in zip(wanted, decoded) if chunk is None]
        if missing:
            environment = self._environments[info.part_directory]
            with environment.begin(write=False) as transaction:
                for position, (chunk_frame, chunk) in enumerate(zip(wanted, decoded)):
                    if chunk is not None:
                        continue
                    key = str((info.episode_index, chunk_frame)).encode()
                    value = transaction.get(key)
                    if value is None:
                        raise KeyError(f"缺少 chunk {key!r}（{info.part_directory}）")
                    fresh = self._decode_chunk(value)
                    decoded[position] = fresh
                    self._store_chunk(info, chunk_frame, fresh)
        offset = start - first_chunk_frame
        count = end - start
        if len(decoded) == 1:
            # 单块命中：直接切片省掉逐字段 concatenate，这是顺序扫描的主路径。
            # 必须 copy——切片是缓存对象的视图，调用方若原地修改会污染缓存。
            single = decoded[0]
            if self.modality == "image":
                return np.array(single[offset:offset + count], copy=True)
            if self.modality == "meta_info":
                return [dict(frame) for frame in single[offset:offset + count]]
            return {
                key: np.array(value[offset:offset + count], copy=True)
                for key, value in single.items()
            }
        merged = self._merge_chunks(decoded)
        if self.modality == "image":
            return merged[offset:offset + count]
        if self.modality == "meta_info":
            return [dict(frame) for frame in merged[offset:offset + count]]
        return {key: value[offset:offset + count] for key, value in merged.items()}


def discover_part_directories(dataset_directory: Path, modality: str) -> list[Path]:
    """列出某数据集目录下指定模态的全部可用分片（含 ``data.mdb`` 的才算）。"""
    modality_root = Path(dataset_directory) / modality
    if not modality_root.is_dir():
        return []
    return sorted(
        directory
        for directory in modality_root.iterdir()
        if directory.is_dir() and (directory / "data.mdb").is_file()
    )


class TrajectoryReader:
    """多模态按 episode 名对齐的轨迹读取器。

    Parameters
    ----------
    dataset_directories : list of Path
        数据集根目录列表，例如 ``[Path("runs/bc_datasets/minestudio-data-10xx-v110")]``。
    modalities : list of str
        要读取的模态，必须包含 ``"action"``；``"image"`` 可选（缺失时只出动作）。
    frame_width, frame_height : int
        图像解码尺寸，单位像素。

    Notes
    -----
    只保留在所有请求模态里都存在的 episode，取交集。各模态分片切分边界不同，
    因此对齐必须走 episode 名而不是分片号。
    """

    def __init__(
        self,
        dataset_directories: list[Path],
        modalities: list[str],
        frame_width: int = 224,
        frame_height: int = 224,
    ) -> None:
        if "action" not in modalities:
            raise ValueError("modalities 必须包含 'action'")
        self.readers: dict[str, LMDBModalityReader] = {}
        for modality in modalities:
            parts: list[Path] = []
            for dataset_directory in dataset_directories:
                parts.extend(discover_part_directories(Path(dataset_directory), modality))
            if not parts:
                raise FileNotFoundError(
                    f"模态 {modality} 没有找到任何含 data.mdb 的分片；先用 "
                    f"bc_datasets.minestudio.huggingface_download 下载",
                )
            self.readers[modality] = LMDBModalityReader(
                part_directories=parts,
                modality=modality,  # type: ignore[arg-type]
                frame_width=frame_width,
                frame_height=frame_height,
            )
        common = set(self.readers["action"].episode_names())
        for modality, reader in self.readers.items():
            common &= set(reader.episode_names())
        self._episodes = sorted(common)

    def episode_names(self) -> list[str]:
        """在所有请求模态里都存在的 episode 名。"""
        return list(self._episodes)

    def episode_length(self, episode: str) -> int:
        """某条 episode 在各模态中的最短帧数（对齐后可用长度）。"""
        return min(
            reader.episode_info(episode).num_frames for reader in self.readers.values()
        )

    def read_window(self, episode: str, start: int, length: int) -> dict[str, Any]:
        """读取一个跨模态对齐的窗口。

        Returns
        -------
        dict
            模态名 → 该模态的窗口数据，各模态时间轴对齐、长度一致。
        """
        available = self.episode_length(episode) - start
        if available <= 0:
            raise ValueError(f"start {start} 越过 episode {episode} 的可用长度")
        span = min(length, available)
        return {
            modality: reader.read_frames(episode, start, span)
            for modality, reader in self.readers.items()
        }

    def close(self) -> None:
        """关闭全部模态读取器。"""
        for reader in self.readers.values():
            reader.close()
