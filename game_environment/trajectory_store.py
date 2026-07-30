"""持久化轨迹的任意帧查询接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryFrame:
    trajectory_id: str
    frame_id: int
    tick: int
    path: Path
    segment: int | None
    action: dict[str, Any]
    reason: str


class TrajectoryStore:
    """通过稳定 ID 在后续对话轮次中定位已保存的真实 RGB 帧。"""

    def __init__(self, run_directory: Path):
        self.run_directory = run_directory.resolve()
        manifest_path = self.run_directory / "trajectory_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.snapshot_id = payload["snapshot_id"]
        self._frames = {
            item["trajectory_id"]: tuple(item["frames"])
            for item in payload["trajectories"]
        }

    def frame(self, trajectory_id: str, frame_id: int) -> TrajectoryFrame:
        frames = self._trajectory(trajectory_id)
        if frame_id < 0 or frame_id >= len(frames):
            raise IndexError(f"frame_id 超出范围: {frame_id}")
        return self._decode(frames[frame_id])

    def at_or_before_tick(self, trajectory_id: str, tick: int) -> TrajectoryFrame:
        frames = self._trajectory(trajectory_id)
        eligible = [item for item in frames if int(item["tick"]) <= tick]
        if not eligible:
            raise IndexError(f"tick 之前没有图像: {tick}")
        return self._decode(eligible[-1])

    def frames(self, trajectory_id: str) -> tuple[TrajectoryFrame, ...]:
        return tuple(self._decode(item) for item in self._trajectory(trajectory_id))

    def _trajectory(self, trajectory_id: str) -> tuple[dict[str, Any], ...]:
        try:
            return self._frames[trajectory_id]
        except KeyError as error:
            raise KeyError(f"未知 trajectory_id: {trajectory_id}") from error

    def _decode(self, item: dict[str, Any]) -> TrajectoryFrame:
        path = (self.run_directory / item["path"]).resolve()
        if self.run_directory not in path.parents:
            raise ValueError("轨迹帧路径越出运行目录")
        return TrajectoryFrame(
            trajectory_id=item["trajectory_id"],
            frame_id=int(item["frame_id"]),
            tick=int(item["tick"]),
            path=path,
            segment=item.get("segment"),
            action=dict(item.get("action", {})),
            reason=item["reason"],
        )
