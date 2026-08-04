from __future__ import annotations

import threading
import time
from typing import Any

from online_interactive_environments.craftground import (
    MemorySnapshotCoordinator,
    ParallelRolloutRunner,
    RolloutRequest,
)


class FakeEnvironment:
    def __init__(self, name: str) -> None:
        self.name = name
        self.commands: list[str] = []
        self.active = False

    def add_command(self, command: str) -> None:
        self.commands.append(command)

    def step(self, action: Any) -> tuple[None]:
        return (None,)


def test_same_snapshot_is_assigned_to_multiple_subagents_in_parallel() -> None:
    environments = [FakeEnvironment("env-0"), FakeEnvironment("env-1")]
    coordinator = MemorySnapshotCoordinator(
        environments,
        noop_action=lambda: None,
        synchronization_ticks=1,
    )
    runner = ParallelRolloutRunner(coordinator, max_workers=4)
    lock = threading.Lock()
    concurrent = 0
    peak_concurrent = 0

    def simulate(environment: FakeEnvironment, payload: int) -> str:
        nonlocal concurrent, peak_concurrent
        with lock:
            assert not environment.active
            environment.active = True
            concurrent += 1
            peak_concurrent = max(peak_concurrent, concurrent)
        time.sleep(0.03)
        with lock:
            environment.active = False
            concurrent -= 1
        return f"{environment.name}:{payload}"

    requests = [
        RolloutRequest(f"request-{index}", f"subagent-{index}", "shared", index, simulate)
        for index in range(4)
    ]
    results = runner.run(requests, wait_timeout=0.5)

    assert [result.request_id for result in results] == [f"request-{index}" for index in range(4)]
    assert peak_concurrent == 2
    assert sum(result.waited_ms >= 20 for result in results) >= 2
    assert all(
        environment.commands == ["memorysnapshot load shared"] * 2 for environment in environments
    )
