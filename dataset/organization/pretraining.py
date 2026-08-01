"""把 MineStudio 轨迹批量预处理成 TAP 格式的预训练数据。

对外接口：
    PretrainingSample — 一条预训练样本的结构。
    WindowLayout — 感知窗口 / 历史帧 / 步长的时间布局。
    build_pretraining_dataset — 遍历 episode 产出 TAP 样本，写 JSONL + 帧图。
    main — 命令行入口。

产物布局::

    <输出目录>/
        samples_train.jsonl        训练集，每行一条 PretrainingSample
        samples_validation.jsonl   验证集，同格式
        frames/<episode>/<起始帧>_<历史序号>.jpg
        action_frames/<episode>/<起始帧>_<tick序号>.jpg
        dataset_info.json          时间布局、键表、划分统计

时间基准：MineStudio 为 20Hz（50ms/帧）。默认预测 **8 帧、400ms**，窗口内每帧一个
电机 tick（50ms/tick），满足滚动执行所需的最短 8 tick 动作计划。
TAP 原文的 6×33ms 是《原神》的 30Hz 电机步，这里按 Minecraft tick 换算，不照抄。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset.extraction.minestudio.reader import TrajectoryReader
from dataset.organization.split import HoldoutLevel, build_split
from tao.protocols.action import (
    DEFAULT_WINDOW_FRAMES,
    FRAMES_PER_SECOND,
    HISTORY_FRAME_INTERVAL,
    MINECRAFT_KEYMAP,
    encode_action_sequence,
    validate_action_image_alignment,
)


@dataclass(frozen=True)
class WindowLayout:
    """预训练样本的时间布局。

    Attributes
    ----------
    window_frames : int
        一个感知窗口的帧数。默认 8 帧 = 400ms。
    frames_per_tick : int
        每个电机 tick 覆盖的帧数，需整除 ``window_frames``。
    history_windows : int
        除当前帧外额外给出的历史观测帧数，每隔 4 tick 回溯一帧。
        0 表示 non-history 配方；TAP 的 history 变体代价约 3.5×。
    stride_frames : int
        相邻样本的起始帧间隔。等于 ``window_frames`` 时窗口不重叠。
    """

    window_frames: int = DEFAULT_WINDOW_FRAMES
    frames_per_tick: int = 1
    history_windows: int = 0
    stride_frames: int = DEFAULT_WINDOW_FRAMES

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
    def ticks_per_window(self) -> int:
        """一个感知窗口产出的电机 tick 数。"""
        return self.window_frames // self.frames_per_tick

    @property
    def history_span_frames(self) -> int:
        """历史帧向前回溯的总帧数。"""
        return self.history_windows * self.window_frames


@dataclass(frozen=True)
class PretrainingSample:
    """一条 TAP 格式预训练样本。

    Attributes
    ----------
    episode : str
        来源 episode 名。
    start_frame : int
        本样本动作窗口的起始帧下标。
    image_paths : list of str
        观测帧相对路径，时间升序，最后一张是当前帧；``image`` 模态缺失时为空列表。
    action_image_paths : list of str
        与动作窗口逐帧对应的图片路径，长度等于动作 tick 数。
    action_text : str
        本窗口的 TAP 动作串（监督目标）。
    previous_action_text : str
        上一个窗口的 TAP 动作串；无前序窗口时为空串。
    """

    episode: str
    start_frame: int
    image_paths: list[str]
    action_image_paths: list[str]
    action_text: str
    previous_action_text: str


def _write_frames(
    frames: np.ndarray,
    episode: str,
    start_frame: int,
    frames_directory: Path,
    jpeg_quality: int,
) -> list[str]:
    """把观测帧写成 JPEG，返回相对 ``frames_directory.parent`` 的路径列表。

    frames 为 shape (N, H, W, 3) 的 RGB uint8 数组，时间升序。
    """
    episode_directory = frames_directory / episode
    episode_directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, frame in enumerate(frames):
        filename = f"{start_frame:08d}_{index:02d}.jpg"
        target = episode_directory / filename
        # cv2 写盘要 BGR，读进来是 RGB。
        cv2.imwrite(
            str(target),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        paths.append(f"{frames_directory.name}/{episode}/{filename}")
    return paths


def _observation_frame_indices(start_frame: int, layout: WindowLayout) -> list[int]:
    """当前帧 + 历史帧的下标，时间升序，最后一项是当前帧。

    当前帧取窗口起始帧：模型看到 t 时刻的画面，预测从 t 起执行的动作窗口。
    """
    indices = [
        start_frame - offset * HISTORY_FRAME_INTERVAL
        for offset in range(layout.history_windows, 0, -1)
    ]
    indices.append(start_frame)
    return [index for index in indices if index >= 0]


def _iterate_episode_samples(
    reader: TrajectoryReader,
    episode: str,
    layout: WindowLayout,
    frames_directory: Path | None,
    jpeg_quality: int,
) -> Iterator[PretrainingSample]:
    """对单条 episode 逐窗口产出样本。"""
    has_image = "image" in reader.readers
    total_frames = reader.episode_length(episode)
    # 起始帧必须留足历史回溯空间，且窗口不能越过 episode 末尾。
    first_start = layout.history_span_frames
    action_reader = reader.readers["action"]
    for start in range(first_start, total_frames - layout.window_frames + 1, layout.stride_frames):
        window = action_reader.read_frames(episode, start, layout.window_frames)
        action = encode_action_sequence(window, frames_per_tick=layout.frames_per_tick)
        action_text = action.to_text()

        # 前序动作按时间轴真实回溯一个感知窗口，与 stride 无关。
        previous_action_text = ""
        previous_start = start - layout.window_frames
        if previous_start >= 0:
            previous_window = action_reader.read_frames(
                episode,
                previous_start,
                layout.window_frames,
            )
            previous_action_text = encode_action_sequence(
                previous_window,
                frames_per_tick=layout.frames_per_tick,
            ).to_text()

        image_paths: list[str] = []
        action_image_paths: list[str] = []
        if has_image and frames_directory is not None:
            # 动作窗口与图片窗口必须逐帧对应；历史观测单独保存，不参与动作对齐计数。
            action_frames = action_reader.read_frames(episode, start, layout.window_frames)
            aligned_images = reader.readers["image"].read_frames(
                episode,
                start,
                layout.window_frames,
            )
            validate_action_image_alignment(
                action_frames,
                aligned_images,
                expected_frames=layout.window_frames,
            )
            action_frames_directory = frames_directory.parent / "action_frames"
            action_image_paths = _write_frames(
                aligned_images,
                episode,
                start,
                action_frames_directory,
                jpeg_quality,
            )
            indices = _observation_frame_indices(start, layout)
            # 历史帧按感知步稀疏采样，逐帧单读避免把中间帧全解码进内存。
            frames = np.stack(
                [reader.readers["image"].read_frames(episode, index, 1)[0] for index in indices],
                axis=0,
            )
            image_paths = _write_frames(
                frames,
                episode,
                start,
                frames_directory,
                jpeg_quality,
            )

        yield PretrainingSample(
            episode=episode,
            start_frame=start,
            image_paths=image_paths,
            action_image_paths=action_image_paths,
            action_text=action_text,
            previous_action_text=previous_action_text,
        )


def build_pretraining_dataset(
    dataset_directories: list[Path],
    output_directory: Path,
    layout: WindowLayout | None = None,
    frame_width: int = 224,
    frame_height: int = 224,
    include_images: bool = True,
    maximum_episodes: int | None = None,
    maximum_samples: int | None = None,
    jpeg_quality: int = 90,
    holdout_level: HoldoutLevel = "prefix",
    validation_ratio: float = 0.1,
    split_seed: int = 3407,
) -> dict[str, Any]:
    """批量把 MineStudio 轨迹预处理成 TAP 格式预训练数据。

    Parameters
    ----------
    dataset_directories : list of Path
        MineStudio 数据集根目录列表。
    output_directory : Path
        输出目录，写入 ``samples_train.jsonl``、``samples_validation.jsonl``、
        ``frames/``、``split.json`` 与 ``dataset_info.json``。
    layout : WindowLayout or None
        时间布局，None 表示默认（8 帧窗口 / 1 帧一 chunk / 无历史 / 窗口不重叠）。
    frame_width, frame_height : int
        观测帧解码尺寸，单位像素。
    include_images : bool
        是否读取并落盘观测帧。False 时只产出动作文本（``image`` 模态未下载时必须为 False）。
    maximum_episodes : int or None
        最多处理的 episode 数，None 表示全部。
    maximum_samples : int or None
        最多产出的样本数，None 表示不限。两个子集各自受此上限约束。
    jpeg_quality : int
        观测帧 JPEG 质量，1–100。
    holdout_level : {"prefix", "episode"}
        验证集留出粒度，见 ``dataset.organization.split.build_split``。
    validation_ratio : float
        验证集目标帧数占比。
    split_seed : int
        ``episode`` 粒度打散的稳定哈希种子。

    Returns
    -------
    dict
        统计信息：两个子集的样本数与 episode 数、时间布局、划分口径等。

    Raises
    ------
    FileNotFoundError
        ``action`` 模态没有可用分片，或 ``include_images`` 为真但 ``image`` 分片缺失。
    """
    resolved = layout if layout is not None else WindowLayout()
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality 必须在 1–100")
    modalities = ["action", "image"] if include_images else ["action"]
    reader = TrajectoryReader(
        dataset_directories=dataset_directories,
        modalities=modalities,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    frames_directory = output_directory / "frames" if include_images else None

    # 复用已打开的 reader 读帧数：LMDB 不允许同进程重复打开同一环境。
    split = build_split(
        holdout_level=holdout_level,
        validation_ratio=validation_ratio,
        seed=split_seed,
        output_path=output_directory / "split.json",
        episode_frames={name: reader.episode_length(name) for name in reader.episode_names()},
    )
    subsets = {
        "train": split.train_episodes,
        "validation": split.validation_episodes,
    }
    if maximum_episodes is not None:
        subsets = {name: episodes[:maximum_episodes] for name, episodes in subsets.items()}

    counts: dict[str, int] = {}
    try:
        for subset_name, episodes in subsets.items():
            samples_path = output_directory / f"samples_{subset_name}.jsonl"
            num_samples = 0
            with samples_path.open("w", encoding="utf-8") as handle:
                for episode in episodes:
                    for sample in _iterate_episode_samples(
                        reader,
                        episode,
                        resolved,
                        frames_directory,
                        jpeg_quality,
                    ):
                        handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                        num_samples += 1
                        if maximum_samples is not None and num_samples >= maximum_samples:
                            break
                    if maximum_samples is not None and num_samples >= maximum_samples:
                        break
            counts[subset_name] = num_samples
    finally:
        reader.close()

    info: dict[str, Any] = {
        "num_train_samples": counts["train"],
        "num_validation_samples": counts["validation"],
        "num_train_episodes": len(subsets["train"]),
        "num_validation_episodes": len(subsets["validation"]),
        "holdout_level": split.holdout_level,
        "validation_prefixes": split.validation_prefixes,
        "achieved_validation_frame_ratio": split.achieved_validation_ratio,
        "target_validation_frame_ratio": split.target_validation_ratio,
        "frames_per_second": FRAMES_PER_SECOND,
        "window_frames": resolved.window_frames,
        "window_milliseconds": resolved.window_frames * 1000 // FRAMES_PER_SECOND,
        "frames_per_tick": resolved.frames_per_tick,
        "ticks_per_window": resolved.ticks_per_window,
        "history_windows": resolved.history_windows,
        "stride_frames": resolved.stride_frames,
        "includes_images": include_images,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "keymap": MINECRAFT_KEYMAP,
        "action_image_alignment": {
            "validated": include_images,
            "rule": "每个动作窗口的 action 帧数必须等于同一 episode、同一起点的 image 帧数",
            "tick_milliseconds": 1000 // FRAMES_PER_SECOND,
        },
    }
    (output_directory / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return info


def main() -> None:
    """命令行入口：MineStudio → TAP 预训练数据。"""
    parser = argparse.ArgumentParser(
        description="把 MineStudio 轨迹转成 TAP 预训练数据",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        nargs="+",
        required=True,
        help="MineStudio 数据集根目录",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument(
        "--window-frames",
        type=int,
        default=DEFAULT_WINDOW_FRAMES,
        help="动作预测窗口帧数；默认 8 帧 = 400ms",
    )
    parser.add_argument(
        "--frames-per-tick",
        type=int,
        default=1,
        help="每个电机 tick 的帧数",
    )
    parser.add_argument("--history-windows", type=int, default=0, help="额外历史观测帧数")
    parser.add_argument(
        "--stride-frames", type=int, default=None, help="样本步长，默认等于窗口帧数"
    )
    parser.add_argument("--frame-width", type=int, default=224, help="观测帧宽，像素")
    parser.add_argument("--frame-height", type=int, default=224, help="观测帧高，像素")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="只产出动作文本；image 分片未下载时必须开启",
    )
    parser.add_argument("--maximum-episodes", type=int, default=None, help="最多处理的 episode 数")
    parser.add_argument("--maximum-samples", type=int, default=None, help="最多产出的样本数")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="观测帧 JPEG 质量")
    parser.add_argument(
        "--holdout-level",
        default="prefix",
        choices=("prefix", "episode"),
        help="prefix：整个玩家留出，衡量跨玩家泛化；episode：按 episode 打散",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="验证集目标帧数占比",
    )
    parser.add_argument("--split-seed", type=int, default=3407, help="episode 粒度打散种子")
    arguments = parser.parse_args()

    layout = WindowLayout(
        window_frames=arguments.window_frames,
        frames_per_tick=arguments.frames_per_tick,
        history_windows=arguments.history_windows,
        stride_frames=(
            arguments.stride_frames
            if arguments.stride_frames is not None
            else arguments.window_frames
        ),
    )
    info = build_pretraining_dataset(
        dataset_directories=arguments.dataset_dir,
        output_directory=arguments.output_dir,
        layout=layout,
        frame_width=arguments.frame_width,
        frame_height=arguments.frame_height,
        include_images=not arguments.no_images,
        maximum_episodes=arguments.maximum_episodes,
        maximum_samples=arguments.maximum_samples,
        jpeg_quality=arguments.jpeg_quality,
        holdout_level=arguments.holdout_level,
        validation_ratio=arguments.validation_ratio,
        split_seed=arguments.split_seed,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
