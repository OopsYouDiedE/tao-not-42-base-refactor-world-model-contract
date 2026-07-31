"""CraftGround 内存快照的一键保存与多环境恢复接口。"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

_SNAPSHOT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class CraftGroundCommandEnvironment(Protocol):
    def add_command(self, command: str) -> None: ...

    def step(self, action: Any) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class SnapshotRegion:
    minimum: tuple[int, int, int]
    maximum: tuple[int, int, int]

    def command_coordinates(self) -> str:
        coordinates = (*self.minimum, *self.maximum)
        return " ".join(str(coordinate) for coordinate in coordinates)


@dataclass(frozen=True)
class MemorySnapshot:
    snapshot_id: str
    region: SnapshotRegion


@dataclass(frozen=True)
class ResetTimings:
    wall_ms: float
    worker_ms: tuple[float, ...]


class MemorySnapshotCoordinator:
    """向一组常驻 CraftGround 环境广播内存快照命令。"""

    def __init__(
        self,
        environments: Iterable[CraftGroundCommandEnvironment],
        *,
        noop_action: Callable[[], Any] | None = None,
        synchronization_ticks: int = 2,
    ):
        self.environments = tuple(environments)
        if not self.environments:
            raise ValueError("至少需要一个 CraftGround 环境")
        if synchronization_ticks < 1:
            raise ValueError("synchronization_ticks 必须大于零")
        self.noop_action = noop_action or _craftground_noop
        self.synchronization_ticks = synchronization_ticks

    def capture_all(self, snapshot_id: str, region: SnapshotRegion) -> MemorySnapshot:
        """让所有环境以同一 ID 保存各自的当前内存状态。"""
        _validate_snapshot_id(snapshot_id)
        command = f"memorysnapshot save {snapshot_id} {region.command_coordinates()}"
        self._broadcast(command)
        return MemorySnapshot(snapshot_id=snapshot_id, region=region)

    def reset_all(self, snapshot: MemorySnapshot | str) -> ResetTimings:
        """只凭快照句柄或 ID 并行恢复全部环境。"""
        snapshot_id = snapshot.snapshot_id if isinstance(snapshot, MemorySnapshot) else snapshot
        _validate_snapshot_id(snapshot_id)
        wall_start = time.perf_counter()
        worker_ms = self._broadcast(f"memorysnapshot load {snapshot_id}")
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return ResetTimings(wall_ms=wall_ms, worker_ms=worker_ms)

    def _broadcast(self, command: str) -> tuple[float, ...]:
        with ThreadPoolExecutor(max_workers=len(self.environments)) as executor:
            timings = executor.map(
                lambda environment: self._send(environment, command), self.environments
            )
            return tuple(timings)

    def _send(self, environment: CraftGroundCommandEnvironment, command: str) -> float:
        start = time.perf_counter()
        environment.add_command(command)
        for _ in range(self.synchronization_ticks):
            environment.step(self.noop_action())
        return (time.perf_counter() - start) * 1000.0


def _validate_snapshot_id(snapshot_id: str) -> None:
    if not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("snapshot_id 只能包含字母、数字、点、下划线和连字符")


def _craftground_noop() -> Any:
    from craftground.environment.action_space import no_op_v2

    return no_op_v2()
