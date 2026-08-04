"""将常驻 CraftGround 实例安全分配给并发 SubAgent。"""

from __future__ import annotations

import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Condition
from typing import Generic, TypeVar

EnvironmentT = TypeVar("EnvironmentT")


class EnvironmentPoolTimeout(TimeoutError):
    """等待空闲 CraftGround 实例超过调用方给定时限。"""


@dataclass(frozen=True)
class EnvironmentLease(AbstractContextManager[EnvironmentT], Generic[EnvironmentT]):
    """环境独占租约；退出上下文时自动归还。"""

    _pool: EnvironmentPool[EnvironmentT]
    environment: EnvironmentT
    slot: int
    waited_ms: float

    def __enter__(self) -> EnvironmentT:
        return self.environment

    def __exit__(self, *exc_info: object) -> None:
        self._pool.release(self)


class EnvironmentPool(Generic[EnvironmentT]):
    """有界环境池；请求超出槽位数时按到达顺序等待。"""

    def __init__(self, environments: list[EnvironmentT] | tuple[EnvironmentT, ...]) -> None:
        if not environments:
            raise ValueError("环境池至少需要一个 CraftGround 环境")
        self._environments = tuple(environments)
        self._available = deque(range(len(self._environments)))
        self._leased: dict[int, EnvironmentLease[EnvironmentT]] = {}
        self._condition = Condition()

    @property
    def capacity(self) -> int:
        return len(self._environments)

    @property
    def available(self) -> int:
        with self._condition:
            return len(self._available)

    def acquire(self, timeout: float | None = None) -> EnvironmentLease[EnvironmentT]:
        """取得独占槽位；无空位时等待，超时则抛出明确异常。"""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout 不能为负数")
        started = time.monotonic()
        deadline = None if timeout is None else started + timeout
        with self._condition:
            while not self._available:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise EnvironmentPoolTimeout("等待空闲 CraftGround 环境超时")
                self._condition.wait(remaining)
            slot = self._available.popleft()
            lease = EnvironmentLease(
                self,
                self._environments[slot],
                slot,
                (time.monotonic() - started) * 1000.0,
            )
            self._leased[slot] = lease
            return lease

    def release(self, lease: EnvironmentLease[EnvironmentT]) -> None:
        with self._condition:
            if lease._pool is not self or self._leased.get(lease.slot) is not lease:
                raise ValueError("租约不属于此环境池，或已经归还")
            del self._leased[lease.slot]
            self._available.append(lease.slot)
            self._condition.notify()
