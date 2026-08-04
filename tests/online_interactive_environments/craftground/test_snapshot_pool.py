from __future__ import annotations

import threading
import time

import pytest

from online_interactive_environments.craftground import (
    EnvironmentPool,
    EnvironmentPoolTimeout,
)


def test_excess_request_waits_until_slot_is_released() -> None:
    pool = EnvironmentPool(["environment"])
    first = pool.acquire()
    acquired = threading.Event()
    result: list[float] = []

    def wait_for_slot() -> None:
        lease = pool.acquire(timeout=0.5)
        result.append(lease.waited_ms)
        acquired.set()
        pool.release(lease)

    worker = threading.Thread(target=wait_for_slot)
    worker.start()
    time.sleep(0.04)
    assert not acquired.is_set()
    pool.release(first)
    worker.join(timeout=0.5)

    assert acquired.is_set()
    assert result[0] >= 30
    assert pool.available == 1


def test_waiting_for_slot_has_explicit_timeout() -> None:
    pool = EnvironmentPool([object()])
    lease = pool.acquire()
    with pytest.raises(EnvironmentPoolTimeout, match="超时"):
        pool.acquire(timeout=0.01)
    pool.release(lease)


def test_context_manager_releases_slot_after_error() -> None:
    pool = EnvironmentPool([object()])
    with pytest.raises(RuntimeError), pool.acquire():
        raise RuntimeError("rollout failed")
    assert pool.available == 1


def test_old_lease_cannot_release_reacquired_slot() -> None:
    pool = EnvironmentPool([object()])
    old_lease = pool.acquire()
    pool.release(old_lease)
    current_lease = pool.acquire()

    with pytest.raises(ValueError):
        pool.release(old_lease)

    assert pool.available == 0
    with pytest.raises(EnvironmentPoolTimeout):
        pool.acquire(timeout=0)
    pool.release(current_lease)
    assert pool.available == 1
