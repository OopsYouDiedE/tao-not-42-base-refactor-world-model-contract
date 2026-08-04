"""CraftGround 快速存档、倒档与多实例同步接口。"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

_SNAPSHOT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class CraftGroundCommandEnvironment(Protocol):
    """内存快照机制所需的最小 CraftGround 接口。"""

    def add_command(self, command: str) -> None: ...

    def step(self, action: Any) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class SnapshotRegion:
    minimum: tuple[int, int, int]
    maximum: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(low > high for low, high in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("快照区域 minimum 不能大于 maximum")

    def command_coordinates(self) -> str:
        return " ".join(str(value) for value in (*self.minimum, *self.maximum))

    @classmethod
    def around_player(
        cls,
        position: tuple[float, float, float] | list[float],
        *,
        horizontal_radius: int = 24,
        minimum_y: int = -64,
        maximum_y: int = 319,
    ) -> SnapshotRegion:
        if len(position) != 3:
            raise ValueError("position 必须包含 x、y、z")
        if horizontal_radius < 1:
            raise ValueError("horizontal_radius 必须大于零")
        center_x = int(float(position[0]) // 1)
        center_z = int(float(position[2]) // 1)
        return cls(
            (center_x - horizontal_radius, minimum_y, center_z - horizontal_radius),
            (center_x + horizontal_radius, maximum_y, center_z + horizontal_radius),
        )


@dataclass(frozen=True)
class MemorySnapshot:
    """同一快照 ID 在每个常驻 JVM 中对应一份本地内存快照。"""

    snapshot_id: str
    region: SnapshotRegion

    def __post_init__(self) -> None:
        _validate_snapshot_id(self.snapshot_id)


@dataclass(frozen=True)
class ResetTimings:
    wall_ms: float
    worker_ms: tuple[float, ...]


class MemorySnapshotCoordinator:
    """并行控制一组常驻 CraftGround 环境的内存快照。"""

    def __init__(
        self,
        environments: Iterable[CraftGroundCommandEnvironment],
        *,
        noop_action: Callable[[], Any] | None = None,
        synchronization_ticks: int = 2,
    ) -> None:
        self.environments = tuple(environments)
        if not self.environments:
            raise ValueError("至少需要一个 CraftGround 环境")
        if synchronization_ticks < 1:
            raise ValueError("synchronization_ticks 必须大于零")
        self.noop_action = noop_action or _craftground_noop
        self.synchronization_ticks = synchronization_ticks

    def capture_all(self, snapshot_id: str, region: SnapshotRegion) -> MemorySnapshot:
        """让全部 JVM 以同一 ID 保存各自当前状态。"""
        _validate_snapshot_id(snapshot_id)
        self._broadcast(f"memorysnapshot save {snapshot_id} {region.command_coordinates()}")
        return MemorySnapshot(snapshot_id, region)

    def reset_all(self, snapshot: MemorySnapshot | str) -> ResetTimings:
        """并行倒档全部环境；墙钟耗时由最慢实例决定。"""
        snapshot_id = snapshot.snapshot_id if isinstance(snapshot, MemorySnapshot) else snapshot
        _validate_snapshot_id(snapshot_id)
        started = time.perf_counter()
        worker_ms = self._broadcast(f"memorysnapshot load {snapshot_id}")
        return ResetTimings((time.perf_counter() - started) * 1000.0, worker_ms)

    def reset_one(
        self,
        environment: CraftGroundCommandEnvironment,
        snapshot: MemorySnapshot | str,
    ) -> float:
        """把一个已分配给 SubAgent 的环境恢复到指定存档。"""
        snapshot_id = snapshot.snapshot_id if isinstance(snapshot, MemorySnapshot) else snapshot
        _validate_snapshot_id(snapshot_id)
        return self._send(environment, f"memorysnapshot load {snapshot_id}")

    def _broadcast(self, command: str) -> tuple[float, ...]:
        with ThreadPoolExecutor(
            max_workers=len(self.environments),
            thread_name_prefix="craftground-snapshot",
        ) as executor:
            return tuple(executor.map(lambda env: self._send(env, command), self.environments))

    def _send(self, environment: CraftGroundCommandEnvironment, command: str) -> float:
        started = time.perf_counter()
        environment.add_command(command)
        for _ in range(self.synchronization_ticks):
            environment.step(self.noop_action())
        return (time.perf_counter() - started) * 1000.0


def _validate_snapshot_id(snapshot_id: str) -> None:
    if not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("snapshot_id 只能包含字母、数字、点、下划线和连字符")


def _craftground_noop() -> Any:
    from craftground.environment.action_space import no_op_v2

    return no_op_v2()
