"""MineStudio 数据下载与 LMDB 读取。"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name != "TrajectoryReader":
        raise AttributeError(name)
    from dataset.minestudio.reader import TrajectoryReader

    return TrajectoryReader

__all__ = ["TrajectoryReader"]
