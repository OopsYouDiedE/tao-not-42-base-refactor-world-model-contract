"""LMDB → 训练格式的流式转换：不落盘中间产物，边读边转边喂。

对外接口：
    StreamingSettings — 流式加载的时间布局与并行度配置。
    TAPStreamingDataset — PyTorch ``Dataset``，逐样本从 LMDB 转出对话。
    resolve_worker_count — 按 CPU 核心数与可用内存推算 DataLoader worker 数。
    build_streaming_dataset — 按 split 造训练 / 验证两个 Dataset。

转换方法（LMDB 到训练格式，共五步）
-----------------------------------
全量下载好的 MineStudio 数据集在盘上是按模态解耦的 LMDB 分片，训练要的是
「图像 + 指令 → 动作串」的对话。两者之间的转换在本模块里逐样本完成：

1. **建索引，不读数据。** ``MineStudioDataset`` 打开 ``action`` 与 ``image`` 两个模态，
   取 episode 名的交集（各模态分片边界不同，只能按名字对齐）。此时只读了每个分片的
   ``__chunk_infos__`` 元数据，拿到每条 episode 的帧数，一帧像素都没解码。

2. **切样本位。** 每条 episode 按 ``stride_frames`` 切出若干起始帧，构成
   ``(episode, start_frame)`` 的扁平索引表。起始帧需留足历史回溯空间，且窗口不能越过
   episode 末尾。这张表是纯整数，全量 10xx 也只有几百万条元组，常驻内存无压力。

3. **取一个窗口。** ``__getitem__`` 拿到 ``(episode, start)`` 后，从 ``action`` 模态读
   ``window_frames`` 帧（默认 20Hz 下 8 帧 = 400ms），从 ``image`` 模态读当前帧与历史帧。
   LMDB 的块 LRU 缓存让顺序扫描的相邻样本大量命中同一块，避免重复解码。

4. **编码动作。** ``encode_action_sequence`` 把这一窗口的按键与相机增量转成 TAP
   run-length 动作串：窗口内累计鼠标像素增量打头，随后每个电机 chunk 列出该 chunk
   按住的键。这就是监督目标。前序窗口同样编码一次，作为 prompt 里的动作历史。

5. **组装对话。** ``build_conversation`` 把帧与指令拼成 messages，图像在文本之前
   （Unsloth 视觉微调的硬约束），assistant 回复只含动作串。

关键差别：帧以 ``PIL.Image`` 对象直接进对话，从不写 JPEG。落盘版本要先把全部帧写成
文件再训练，全量数据下这一步的耗时与占盘都远超训练本身，且 JPEG 有损重编码会引入
不必要的画质损失。

并行度
------
解码是 CPU 密集的（H.264 解码 + resize），GPU 会等数据。``resolve_worker_count``
按核心数与可用内存两头取小：每个 worker 自己持有一份 LMDB 块缓存，内存是硬约束，
核心数是收益上限。LMDB 环境不可跨进程 fork 后共享，故每个 worker 在首次取数时
惰性打开自己的 reader。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset.extraction.minestudio import MineStudioDataset
from dataset.organization.split import HoldoutLevel, build_split
from tao.protocols.action import (
    DEFAULT_WINDOW_FRAMES,
    FRAMES_PER_SECOND,
    HISTORY_FRAME_INTERVAL,
    encode_action_sequence,
)
from train.bc.conversation_dataset import DEFAULT_INSTRUCTION, build_conversation

_BYTES_PER_GIBIBYTE = 1024**3

# 每个 worker 的内存预算估计值。一个 image 块是 chunk_size × H × W × 3 字节
# （32×224×224×3 ≈ 4.6MB），默认 32 块缓存约 150MB，加上解码临时缓冲与 Python 开销，
# 按 1GiB 留足余量。
_MEMORY_BUDGET_PER_WORKER_BYTES = 1 * _BYTES_PER_GIBIBYTE

# worker 数上限。再高时 LMDB 随机读与主进程的 collate 成为瓶颈，收益枯竭。
_MAXIMUM_WORKERS = 16


@dataclass(frozen=True)
class StreamingSettings:
    """流式加载的时间布局与并行度。

    Attributes
    ----------
    window_frames : int
        一个感知窗口的帧数。默认 8 帧 = 400ms。
    frames_per_tick : int
        每个电机 chunk 覆盖的帧数，需整除 ``window_frames``。
    history_windows : int
        除当前帧外额外给出的历史观测帧数，每隔 4 tick 回溯一帧。
        0 表示 non-history 配方。
    stride_frames : int
        相邻样本的起始帧间隔。等于 ``window_frames`` 时窗口不重叠。
    frame_width, frame_height : int
        观测帧解码尺寸，单位像素。
    include_previous_action : bool
        prompt 是否带上一窗口的动作串。
    instruction : str
        任务指令文本。
    """

    window_frames: int = DEFAULT_WINDOW_FRAMES
    frames_per_tick: int = 1
    history_windows: int = 0
    stride_frames: int = DEFAULT_WINDOW_FRAMES
    frame_width: int = 224
    frame_height: int = 224
    include_previous_action: bool = True
    instruction: str = DEFAULT_INSTRUCTION

    def __post_init__(self) -> None:
        if self.window_frames < 1:
            raise ValueError("window_frames 必须 >= 1")
        if self.frames_per_tick < 1:
            raise ValueError("frames_per_tick 必须 >= 1")
        if self.window_frames % self.frames_per_tick != 0:
            raise ValueError(
                f"window_frames {self.window_frames} 必须能被 frames_per_tick "
                f"{self.frames_per_tick} 整除",
            )
        if self.history_windows < 0:
            raise ValueError("history_windows 不能为负")
        if self.stride_frames < 1:
            raise ValueError("stride_frames 必须 >= 1")

    @property
    def history_span_frames(self) -> int:
        """历史帧向前回溯的总帧数。"""
        return self.history_windows * self.window_frames


def resolve_worker_count(
    logical_cores: int | None = None,
    available_memory_bytes: int | None = None,
    memory_budget_per_worker_bytes: int = _MEMORY_BUDGET_PER_WORKER_BYTES,
    maximum_workers: int = _MAXIMUM_WORKERS,
) -> int:
    """按 CPU 核心数与可用内存推算 DataLoader worker 数。

    Parameters
    ----------
    logical_cores : int or None
        逻辑核心数，None 时现场检测。
    available_memory_bytes : int or None
        可用内存，None 时现场检测；检测不到则只按核心数决定。
    memory_budget_per_worker_bytes : int
        单个 worker 的内存预算，用于把内存换算成 worker 数上限。
    maximum_workers : int
        硬上限。

    Returns
    -------
    int
        worker 数，至少 1。

    Notes
    -----
    留一个核心给主进程做 collate 与 GPU 提交，故取 ``核心数 - 1``。内存与核心两个
    上限取小值：worker 各自持有 LMDB 块缓存，内存不够时多开 worker 只会触发换页，
    比串行更慢。
    """
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 1)
    by_cores = max(1, cores - 1)

    if available_memory_bytes is None:
        # 调用方不提供内存数据时只按核心数决定，并受硬上限约束。
        return min(by_cores, maximum_workers)

    by_memory = max(1, int(available_memory_bytes // memory_budget_per_worker_bytes))
    return max(1, min(by_cores, by_memory, maximum_workers))


def _observation_frame_indices(start_frame: int, settings: StreamingSettings) -> list[int]:
    """当前帧 + 历史帧的下标，时间升序，最后一项是当前帧。

    当前帧取窗口起始帧：模型看到 t 时刻的画面，预测从 t 起执行的动作窗口。
    """
    indices = [
        start_frame - offset * HISTORY_FRAME_INTERVAL
        for offset in range(settings.history_windows, 0, -1)
    ]
    indices.append(start_frame)
    return [index for index in indices if index >= 0]


def _sample_positions(
    episode_frames: dict[str, int],
    episodes: list[str],
    settings: StreamingSettings,
) -> list[tuple[str, int]]:
    """为给定 episode 列表切出全部 ``(episode, 起始帧)`` 样本位。

    只用帧数计算，不读任何帧数据。
    """
    positions: list[tuple[str, int]] = []
    first_start = settings.history_span_frames
    for episode in episodes:
        total = episode_frames.get(episode, 0)
        last_start = total - settings.window_frames
        for start in range(first_start, last_start + 1, settings.stride_frames):
            positions.append((episode, start))
    return positions


class TAPStreamingDataset:
    """从 LMDB 流式产出 TAP 对话样本的 PyTorch ``Dataset``。

    Parameters
    ----------
    dataset_directories : list of Path
        MineStudio 数据集根目录列表。
    positions : list of tuple
        ``(episode, 起始帧)`` 样本位列表，由 ``build_streaming_dataset`` 切好。
    settings : StreamingSettings
        时间布局与解码尺寸。
    include_images : bool
        是否读取观测帧。False 时只出动作文本，用于 image 模态未下载的调试。

    Notes
    -----
    reader 惰性打开：``DataLoader`` 用多进程 worker 时，父进程里打开的 LMDB 环境
    不能安全地跨 fork 使用，必须在各 worker 首次取数时各开一份。
    """

    def __init__(
        self,
        dataset_directories: list[Path],
        positions: list[tuple[str, int]],
        settings: StreamingSettings,
        include_images: bool = True,
    ) -> None:
        self.dataset_directories = [Path(directory) for directory in dataset_directories]
        self.positions = positions
        self.settings = settings
        self.include_images = include_images
        self._reader: MineStudioDataset | None = None

    def __len__(self) -> int:
        """样本总数。"""
        return len(self.positions)

    def _ensure_reader(self) -> MineStudioDataset:
        """取本进程的 reader，尚未打开时打开。"""
        if self._reader is None:
            modalities = ["action", "image"] if self.include_images else ["action"]
            self._reader = MineStudioDataset(self.dataset_directories[0], modalities).updata_index()
        return self._reader

    def _encode_window(self, reader: MineStudioDataset, episode: str, start: int) -> str:
        """读一个动作窗口并编码为 TAP 动作串。"""
        window = reader.read_modality(
            "action",
            episode,
            start,
            self.settings.window_frames,
        )
        return encode_action_sequence(
            window,
            frames_per_tick=self.settings.frames_per_tick,
        ).to_text()

    def __getitem__(self, index: int) -> dict[str, list[dict[str, Any]]]:
        """产出一条对话样本。

        Returns
        -------
        dict
            ``{"messages": [...]}``，与落盘路径的 ``build_conversation`` 输出同构，
            可直接喂 ``UnslothVisionDataCollator``。
        """
        episode, start = self.positions[index]
        reader = self._ensure_reader()

        action_text = self._encode_window(reader, episode, start)
        previous_action_text = ""
        previous_start = start - self.settings.window_frames
        if previous_start >= 0:
            previous_action_text = self._encode_window(reader, episode, previous_start)

        images: list[Image.Image] = []
        if self.include_images and "image" in reader.modalities:
            for frame_index in _observation_frame_indices(start, self.settings):
                frame = reader.read_modality("image", episode, frame_index, 1)[0]
                images.append(Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB"))

        # 复用落盘路径的对话组装：图像已是内存对象，用 loaded_images 绕过按路径读盘。
        return build_conversation(
            {
                "image_paths": [],
                "action_text": action_text,
                "previous_action_text": previous_action_text,
            },
            dataset_root=Path("."),
            instruction=self.settings.instruction,
            include_previous_action=self.settings.include_previous_action,
            loaded_images=images,
        )

    def __iter__(self) -> Iterator[dict[str, list[dict[str, Any]]]]:
        """顺序迭代全部样本。顺序访问命中 LMDB 块缓存的概率最高。"""
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        """关闭本进程持有的 LMDB 环境。"""
        if self._reader is not None:
            self._reader.close()
            self._reader = None


def build_streaming_dataset(
    dataset_directories: list[Path],
    settings: StreamingSettings | None = None,
    include_images: bool = True,
    holdout_level: HoldoutLevel = "prefix",
    validation_ratio: float = 0.1,
    split_seed: int = 3407,
    maximum_samples: int | None = None,
) -> tuple[TAPStreamingDataset, TAPStreamingDataset, dict[str, Any]]:
    """打开数据集、划分 episode，造出训练与验证两个流式 Dataset。

    Parameters
    ----------
    dataset_directories : list of Path
        MineStudio 数据集根目录列表。
    settings : StreamingSettings or None
        时间布局，None 用默认（8 帧窗口 / 1 帧一 chunk / 无历史 / 窗口不重叠）。
    include_images : bool
        是否读取观测帧。视觉 SFT 必须为 True；False 仅用于无 image 模态的调试。
    holdout_level : {"prefix", "episode"}
        验证集留出粒度，见 ``dataset.organization.split.build_split``。
    validation_ratio : float
        验证集目标帧数占比。
    split_seed : int
        ``episode`` 粒度打散的稳定哈希种子。
    maximum_samples : int or None
        每个子集最多保留的样本位数，None 表示不限。用于小规模试跑。

    Returns
    -------
    tuple
        ``(训练集, 验证集, 统计信息)``。统计信息含样本数、episode 数与划分口径。

    Notes
    -----
    这里打开一次 reader 只为读帧数与做划分，随即关闭；两个 Dataset 各自惰性重开。
    LMDB 不允许同进程重复打开同一环境，故帧数经 ``episode_frames`` 传给
    ``build_split``，不让它再开一次。
    """
    resolved = settings if settings is not None else StreamingSettings()
    modalities = ["action", "image"] if include_images else ["action"]
    reader = MineStudioDataset(dataset_directories[0], modalities).updata_index()
    try:
        episode_frames = dict(reader.lengths)
    finally:
        reader.close()

    split = build_split(
        holdout_level=holdout_level,
        validation_ratio=validation_ratio,
        seed=split_seed,
        episode_frames=episode_frames,
    )

    subsets: dict[str, list[tuple[str, int]]] = {}
    for name, episodes in (
        ("train", split.train_episodes),
        ("validation", split.validation_episodes),
    ):
        positions = _sample_positions(episode_frames, episodes, resolved)
        if maximum_samples is not None:
            positions = positions[:maximum_samples]
        subsets[name] = positions

    datasets = {
        name: TAPStreamingDataset(
            dataset_directories=dataset_directories,
            positions=positions,
            settings=resolved,
            include_images=include_images,
        )
        for name, positions in subsets.items()
    }
    information: dict[str, Any] = {
        "num_train_samples": len(subsets["train"]),
        "num_validation_samples": len(subsets["validation"]),
        "num_train_episodes": len(split.train_episodes),
        "num_validation_episodes": len(split.validation_episodes),
        "holdout_level": split.holdout_level,
        "validation_prefixes": split.validation_prefixes,
        "achieved_validation_frame_ratio": split.achieved_validation_ratio,
        "frames_per_second": FRAMES_PER_SECOND,
        "window_frames": resolved.window_frames,
        "window_milliseconds": resolved.window_frames * 1000 // FRAMES_PER_SECOND,
        "frames_per_tick": resolved.frames_per_tick,
        "history_windows": resolved.history_windows,
        "stride_frames": resolved.stride_frames,
        "includes_images": include_images,
    }
    return datasets["train"], datasets["validation"], information
