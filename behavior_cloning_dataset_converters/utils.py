"""行为克隆数据集转换器共用的确定性划分工具。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Literal

HoldoutLevel = Literal["group", "episode"]
_EXHAUSTIVE_GROUP_LIMIT = 20


@dataclass
class SplitResult:
    """记录训练集与验证集的确定性划分结果。

    Attributes:
        holdout_level: 划分粒度，由具体数据集转换器定义其业务含义。
        train_episodes: 训练集 episode 名称。
        validation_episodes: 验证集 episode 名称。
        train_frames: 训练集总帧数。
        validation_frames: 验证集总帧数。
        validation_groups: 验证集包含的分组名称。
        achieved_validation_ratio: 实际验证集帧数占比。
        target_validation_ratio: 请求的验证集帧数占比。
    """

    holdout_level: str
    train_episodes: list[str] = field(default_factory=list)
    validation_episodes: list[str] = field(default_factory=list)
    train_frames: int = 0
    validation_frames: int = 0
    validation_groups: list[str] = field(default_factory=list)
    achieved_validation_ratio: float = 0.0
    target_validation_ratio: float = 0.0

    @property
    def validation_prefixes(self) -> list[str]:
        """返回旧版 v110 合同使用的验证集前缀名称。"""
        return self.validation_groups


def _stable_order(names: list[str], seed: int) -> list[str]:
    return sorted(names, key=lambda name: hashlib.md5(f"{seed}:{name}".encode()).hexdigest())


def _select_groups_by_frames(group_frames: dict[str, int], target_ratio: float) -> list[str]:
    names = sorted(group_frames)
    total = sum(group_frames.values())
    if total == 0:
        raise ValueError("total frame count is zero")
    target = total * target_ratio
    if len(names) <= _EXHAUSTIVE_GROUP_LIMIT:
        best: tuple[float, tuple[str, ...]] = (float("inf"), ())
        for size in range(1, len(names)):
            for candidate in combinations(names, size):
                deviation = abs(sum(group_frames[name] for name in candidate) - target)
                best = min(best, (deviation, candidate))
        return sorted(best[1])
    selected: list[str] = []
    accumulated = 0
    for name in sorted(names, key=lambda value: -group_frames[value]):
        if accumulated + group_frames[name] <= target or not selected:
            selected.append(name)
            accumulated += group_frames[name]
    return sorted(selected)


def build_grouped_split(
    *,
    episode_frames: dict[str, int],
    episode_groups: dict[str, str] | None = None,
    holdout_level: HoldoutLevel = "group",
    validation_ratio: float = 0.1,
    seed: int = 3407,
    output_path: Path | None = None,
    result_holdout_level: str | None = None,
) -> SplitResult:
    """按任意调用方提供的分组执行确定性训练集与验证集划分。

    Args:
        episode_frames: episode 名称到帧数的映射。
        episode_groups: episode 名称到分组名称的映射。按组划分时必填。
        holdout_level: 使用 ``group`` 或 ``episode`` 粒度划分。
        validation_ratio: 目标验证集帧数占比。
        seed: episode 粒度稳定排序使用的种子。
        output_path: 可选的划分结果 JSON 输出路径。
        result_holdout_level: 写入结果的业务粒度名称，默认使用 ``holdout_level``。

    Returns:
        完整的训练集与验证集划分统计。

    Raises:
        ValueError: 输入为空、比例无效、分组不完整或产生空子集。
    """
    if not 0 < validation_ratio < 1 or not episode_frames:
        raise ValueError("validation_ratio must be in (0, 1) and episodes cannot be empty")
    if any(frames < 0 for frames in episode_frames.values()):
        raise ValueError("episode frame counts cannot be negative")

    episodes = sorted(episode_frames)
    validation_groups: list[str] = []
    if holdout_level == "group":
        if episode_groups is None or set(episode_groups) != set(episodes):
            raise ValueError("episode_groups must contain exactly every episode")
        group_frames: dict[str, int] = {}
        for episode in episodes:
            group = episode_groups[episode]
            group_frames[group] = group_frames.get(group, 0) + episode_frames[episode]
        validation_groups = _select_groups_by_frames(group_frames, validation_ratio)
        selected_groups = set(validation_groups)
        validation = [episode for episode in episodes if episode_groups[episode] in selected_groups]
    elif holdout_level == "episode":
        validation = []
        accumulated = 0
        target = sum(episode_frames.values()) * validation_ratio
        for episode in _stable_order(episodes, seed):
            if accumulated >= target:
                break
            validation.append(episode)
            accumulated += episode_frames[episode]
    else:
        raise ValueError(f"unknown holdout level: {holdout_level!r}")

    validation_set = set(validation)
    train = [episode for episode in episodes if episode not in validation_set]
    if not train or not validation:
        raise ValueError("split produced an empty subset")
    validation_frames = sum(episode_frames[episode] for episode in validation)
    total_frames = sum(episode_frames.values())
    if total_frames == 0:
        raise ValueError("total frame count is zero")
    result = SplitResult(
        holdout_level=result_holdout_level or holdout_level,
        train_episodes=sorted(train),
        validation_episodes=sorted(validation),
        train_frames=total_frames - validation_frames,
        validation_frames=validation_frames,
        validation_groups=validation_groups,
        achieved_validation_ratio=validation_frames / total_frames,
        target_validation_ratio=validation_ratio,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def load_split(path: Path) -> SplitResult:
    """从 JSON 文件读取通用划分结果。"""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if "validation_prefixes" in values and "validation_groups" not in values:
        values["validation_groups"] = values.pop("validation_prefixes")
    return SplitResult(**values)
