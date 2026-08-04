"""基于内存快照的多 SubAgent、多核并行推演。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

from .snapshot_pool import EnvironmentPool
from .snapshots import (
    CraftGroundCommandEnvironment,
    MemorySnapshot,
    MemorySnapshotCoordinator,
)

PayloadT = TypeVar("PayloadT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class RolloutRequest(Generic[PayloadT, OutputT]):
    request_id: str
    subagent_id: str
    snapshot: MemorySnapshot | str
    payload: PayloadT
    simulate: Callable[[CraftGroundCommandEnvironment, PayloadT], OutputT]


@dataclass(frozen=True)
class RolloutResult(Generic[OutputT]):
    request_id: str
    subagent_id: str
    environment_slot: int
    waited_ms: float
    restore_ms: float
    rollout_ms: float
    output: OutputT


class ParallelRolloutRunner:
    """把同一存档分配给多个 SubAgent，并在有限环境槽位上并行推演。"""

    def __init__(
        self,
        coordinator: MemorySnapshotCoordinator,
        *,
        max_workers: int | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.pool = EnvironmentPool(coordinator.environments)
        self.max_workers = max_workers or self.pool.capacity
        if self.max_workers < 1:
            raise ValueError("max_workers 必须大于零")

    def run(
        self,
        requests: Iterable[RolloutRequest[PayloadT, OutputT]],
        *,
        wait_timeout: float | None = None,
    ) -> tuple[RolloutResult[OutputT], ...]:
        """并行执行请求，并按输入顺序返回；超额请求在池外等待。"""
        request_list = tuple(requests)
        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="craftground-rollout",
        ) as executor:
            return tuple(
                executor.map(lambda request: self._run_one(request, wait_timeout), request_list)
            )

    def _run_one(
        self,
        request: RolloutRequest[PayloadT, OutputT],
        wait_timeout: float | None,
    ) -> RolloutResult[OutputT]:
        lease = self.pool.acquire(wait_timeout)
        with lease as environment:
            restore_ms = self.coordinator.reset_one(environment, request.snapshot)
            started = time.perf_counter()
            output = request.simulate(environment, request.payload)
            rollout_ms = (time.perf_counter() - started) * 1000.0
            return RolloutResult(
                request.request_id,
                request.subagent_id,
                lease.slot,
                lease.waited_ms,
                restore_ms,
                rollout_ms,
                output,
            )
