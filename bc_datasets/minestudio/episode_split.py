"""MineStudio episode 的训练 / 验证划分。

对外接口：
    EpisodeIdentity — 从 episode 名解析出的身份信息。
    SplitResult — 划分结果与统计。
    parse_episode_identity — 解析 episode 名。
    build_split — 按 holdout 粒度划分并落盘 ``split.json``。
    load_split — 读回 ``split.json``。
    main — 命令行入口。

episode 名形如 ``lovely-persimmon-angora-02e496ce4abb-20220421-092639``：
``<前缀>-<12 位 hex>-<日期>-<时间>``。前缀是玩家 / 承包商标识（10xx 全量 19 个），
中间 hex 名义上是会话 ID，但 10xx 里 ``f153ac423f61`` 一个值就横跨全部 19 个前缀、
覆盖 442 条 episode——它是退化占位值，**不能单独作为分组键**，故本模块只提供
``prefix`` 与 ``episode`` 两级 holdout。

按前缀划分时帧数分布极偏（10xx 前 4 个前缀占 66%），随机抽组会让验证集占比在
0.1%–22% 之间乱跳，因此组数少时精确枚举求最接近目标占比的子集，而不是随机采样。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Literal

from bc_datasets.minestudio.lmdb_modal_reader import (
    ModalKernelReader,
    discover_part_directories,
)

HoldoutLevel = Literal["prefix", "episode"]

# episode 名的结构：<前缀>-<12 位 hex>-<8 位日期>-<6 位时间>。
_EPISODE_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<session>[0-9a-f]{12})-(?P<date>\d{8})-(?P<time>\d{6})$",
)

# 精确枚举的组数上限：2^20 约 105 万个子集，仍在秒级；超过则退回贪心。
_EXHAUSTIVE_GROUP_LIMIT = 20


@dataclass(frozen=True)
class EpisodeIdentity:
    """从 episode 名解析出的身份信息。

    Attributes
    ----------
    episode : str
        原始 episode 名。
    prefix : str
        玩家 / 承包商标识，如 ``lovely-persimmon-angora``。
    session : str
        名义会话 ID（12 位 hex）。10xx 中存在跨前缀复用的退化值，不可单独作分组键。
    date : str
        录制日期，``YYYYMMDD``。
    time : str
        录制时刻，``HHMMSS``。
    """

    episode: str
    prefix: str
    session: str
    date: str
    time: str


def parse_episode_identity(episode: str) -> EpisodeIdentity:
    """解析 episode 名。

    Raises
    ------
    ValueError
        名字不符合 ``<前缀>-<hex>-<日期>-<时间>`` 结构。
    """
    matched = _EPISODE_PATTERN.match(episode)
    if matched is None:
        raise ValueError(f"episode 名不符合预期结构：{episode!r}")
    return EpisodeIdentity(
        episode=episode,
        prefix=matched.group("prefix"),
        session=matched.group("session"),
        date=matched.group("date"),
        time=matched.group("time"),
    )


@dataclass
class SplitResult:
    """划分结果与统计。

    Attributes
    ----------
    holdout_level : str
        holdout 粒度：``"prefix"`` 或 ``"episode"``。
    train_episodes, validation_episodes : list of str
        两个子集的 episode 名，已排序。
    train_frames, validation_frames : int
        两个子集的总帧数。
    validation_prefixes : list of str
        验证集覆盖的前缀；``prefix`` 粒度下这些前缀完全不出现在训练集。
    achieved_validation_ratio : float
        验证集实际帧数占比。
    target_validation_ratio : float
        请求的目标占比。
    """

    holdout_level: str
    train_episodes: list[str] = field(default_factory=list)
    validation_episodes: list[str] = field(default_factory=list)
    train_frames: int = 0
    validation_frames: int = 0
    validation_prefixes: list[str] = field(default_factory=list)
    achieved_validation_ratio: float = 0.0
    target_validation_ratio: float = 0.0


def _stable_order(names: list[str], seed: int) -> list[str]:
    """按名字的稳定哈希排序，得到与输入顺序无关、可复现的打乱结果。

    用哈希而非 ``random.shuffle``：新增分片后已有 episode 的相对次序不变，
    划分结果不会因为数据集扩容而整体漂移。
    """
    return sorted(
        names,
        key=lambda name: hashlib.md5(f"{seed}:{name}".encode()).hexdigest(),
    )


def _select_groups_by_frames(
    group_frames: dict[str, int],
    target_ratio: float,
) -> list[str]:
    """选出一组 group，使其帧数占比最接近 ``target_ratio``。

    组数不超过 ``_EXHAUSTIVE_GROUP_LIMIT`` 时精确枚举全部子集取最优；否则按帧数
    降序贪心累加。返回的组名已排序。
    """
    names = sorted(group_frames)
    total = sum(group_frames.values())
    if total == 0:
        raise ValueError("总帧数为零，无法划分")
    target_frames = total * target_ratio

    if len(names) <= _EXHAUSTIVE_GROUP_LIMIT:
        best: tuple[float, tuple[str, ...]] = (float("inf"), ())
        for size in range(1, len(names)):
            for candidate in combinations(names, size):
                deviation = abs(sum(group_frames[n] for n in candidate) - target_frames)
                if deviation < best[0]:
                    best = (deviation, candidate)
        return sorted(best[1])

    selected: list[str] = []
    accumulated = 0
    for name in sorted(names, key=lambda n: -group_frames[n]):
        if accumulated + group_frames[name] <= target_frames or not selected:
            selected.append(name)
            accumulated += group_frames[name]
    return sorted(selected)


def read_episode_frames(dataset_directories: list[Path]) -> dict[str, int]:
    """读取各 episode 的帧数，只碰 ``action`` 模态的元数据，不解码任何帧。"""
    parts: list[Path] = []
    for directory in dataset_directories:
        parts.extend(discover_part_directories(Path(directory), "action"))
    if not parts:
        raise FileNotFoundError("没找到任何含 data.mdb 的 action 分片")
    reader = ModalKernelReader(parts, "action")
    try:
        return {name: reader.episode_info(name).num_frames for name in reader.episode_names()}
    finally:
        reader.close()


def build_split(
    dataset_directories: list[Path] | None = None,
    holdout_level: HoldoutLevel = "prefix",
    validation_ratio: float = 0.1,
    seed: int = 3407,
    output_path: Path | None = None,
    episode_frames: dict[str, int] | None = None,
) -> SplitResult:
    """划分 episode 为训练集与验证集。

    Parameters
    ----------
    dataset_directories : list of Path or None
        MineStudio 数据集根目录列表。只读 ``action`` 模态的元数据，不解码帧。
        给定 ``episode_frames`` 时忽略此参数。
    holdout_level : {"prefix", "episode"}
        ``"prefix"``：整个玩家的数据全进验证集，衡量跨玩家泛化——同一玩家的运镜与
        按键习惯不会同时出现在两边，是严格口径。
        ``"episode"``：按 episode 打散，同一玩家可同时出现在两边，验证集数字更好看
        但只衡量同分布内插。
    validation_ratio : float
        验证集目标帧数占比，取值 ``(0, 1)``。
    seed : int
        ``episode`` 粒度打散用的稳定哈希种子。``prefix`` 粒度是精确枚举，与种子无关。
    output_path : Path or None
        给定时把结果写成 JSON。
    episode_frames : dict of str to int or None
        预先读好的 episode → 帧数。调用方已持有打开的 ``ModalKernelReader`` 时必须
        走这条路径：LMDB 不允许同进程重复打开同一环境。

    Returns
    -------
    SplitResult
        划分结果。``prefix`` 粒度下 ``achieved_validation_ratio`` 可能偏离目标——
        组数少且帧数分布偏，这是数据本身的约束，不是实现缺陷。

    Raises
    ------
    ValueError
        ``validation_ratio`` 越界，或划分后任一子集为空。
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio 需在 (0, 1)，实际 {validation_ratio}")

    if episode_frames is not None:
        frames = dict(episode_frames)
    elif dataset_directories is not None:
        frames = read_episode_frames(dataset_directories)
    else:
        raise ValueError("必须给定 dataset_directories 或 episode_frames 之一")
    if not frames:
        raise ValueError("没有任何 episode 可划分")
    episodes = sorted(frames)

    identities = {name: parse_episode_identity(name) for name in episodes}

    if holdout_level == "prefix":
        group_frames: Counter[str] = Counter()
        for name in episodes:
            group_frames[identities[name].prefix] += frames[name]
        held_out = set(_select_groups_by_frames(dict(group_frames), validation_ratio))
        validation = [n for n in episodes if identities[n].prefix in held_out]
    elif holdout_level == "episode":
        ordered = _stable_order(episodes, seed)
        target_frames = sum(frames.values()) * validation_ratio
        validation, accumulated = [], 0
        for name in ordered:
            if accumulated >= target_frames:
                break
            validation.append(name)
            accumulated += frames[name]
    else:
        raise ValueError(f"未知 holdout_level：{holdout_level!r}")

    validation_set = set(validation)
    train = [name for name in episodes if name not in validation_set]
    if not train or not validation:
        raise ValueError(
            f"划分后有空子集（train={len(train)} val={len(validation)}），"
            f"调整 validation_ratio 或换 holdout_level",
        )

    total_frames = sum(frames.values())
    validation_frames = sum(frames[name] for name in validation)
    result = SplitResult(
        holdout_level=holdout_level,
        train_episodes=sorted(train),
        validation_episodes=sorted(validation),
        train_frames=total_frames - validation_frames,
        validation_frames=validation_frames,
        validation_prefixes=sorted({identities[n].prefix for n in validation}),
        achieved_validation_ratio=validation_frames / total_frames,
        target_validation_ratio=validation_ratio,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return result


def load_split(path: Path) -> SplitResult:
    """读回 ``build_split`` 写出的 JSON。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitResult(**payload)


def main() -> None:
    """命令行入口：划分 episode 并打印统计。"""
    parser = argparse.ArgumentParser(description="划分 MineStudio episode 为训练 / 验证集")
    parser.add_argument(
        "--dataset-dir", type=Path, nargs="+", required=True, help="MineStudio 数据集根目录",
    )
    parser.add_argument(
        "--holdout-level", default="prefix", choices=("prefix", "episode"),
        help="prefix：整个玩家留出，衡量跨玩家泛化；episode：按 episode 打散",
    )
    parser.add_argument(
        "--validation-ratio", type=float, default=0.1, help="验证集目标帧数占比",
    )
    parser.add_argument("--seed", type=int, default=3407, help="episode 粒度打散的种子")
    parser.add_argument(
        "--output", type=Path, default=None, help="划分结果 JSON 的输出路径",
    )
    arguments = parser.parse_args()

    result = build_split(
        dataset_directories=arguments.dataset_dir,
        holdout_level=arguments.holdout_level,
        validation_ratio=arguments.validation_ratio,
        seed=arguments.seed,
        output_path=arguments.output,
    )
    total_frames = result.train_frames + result.validation_frames
    print(f"holdout 粒度      {result.holdout_level}")
    print(
        f"训练集            {len(result.train_episodes):>5} episodes  "
        f"{result.train_frames:>10} frames  "
        f"{result.train_frames / total_frames:6.2%}  "
        f"{result.train_frames / 20 / 3600:6.1f} h",
    )
    print(
        f"验证集            {len(result.validation_episodes):>5} episodes  "
        f"{result.validation_frames:>10} frames  "
        f"{result.achieved_validation_ratio:6.2%}  "
        f"{result.validation_frames / 20 / 3600:6.1f} h",
    )
    print(f"目标验证占比      {result.target_validation_ratio:.2%}")
    print(f"验证集前缀        {', '.join(result.validation_prefixes)}")
    if arguments.output is not None:
        print(f"已写出            {arguments.output}")


if __name__ == "__main__":
    main()
