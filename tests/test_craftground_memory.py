import time
from typing import Any

import pytest

from game_environment.craftground_memory import MemorySnapshotCoordinator, SnapshotRegion


class FakeEnvironment:
    def __init__(self, latency: float = 0.0):
        self.commands: list[str] = []
        self.actions: list[Any] = []
        self.latency = latency

    def add_command(self, command: str) -> None:
        self.commands.append(command)

    def step(self, action: Any) -> tuple[None]:
        time.sleep(self.latency)
        self.actions.append(action)
        return (None,)


def test_capture_and_reset_use_snapshot_id_only() -> None:
    environments = [FakeEnvironment(), FakeEnvironment()]
    coordinator = MemorySnapshotCoordinator(environments, noop_action=lambda: "noop")
    region = SnapshotRegion((0, 63, 0), (8, 68, 8))

    snapshot = coordinator.capture_all("branch-42", region)
    timings = coordinator.reset_all(snapshot)

    for environment in environments:
        assert environment.commands == [
            "memorysnapshot save branch-42 0 63 0 8 68 8",
            "memorysnapshot load branch-42",
        ]
        assert environment.actions == ["noop"] * 4
    assert timings.wall_ms >= 0
    assert len(timings.worker_ms) == 2


def test_reset_all_broadcasts_in_parallel() -> None:
    environments = [FakeEnvironment(0.03) for _ in range(4)]
    coordinator = MemorySnapshotCoordinator(
        environments,
        noop_action=lambda: None,
        synchronization_ticks=1,
    )

    timings = coordinator.reset_all("golden")

    assert timings.wall_ms < 100
    assert all(environment.commands == ["memorysnapshot load golden"] for environment in environments)


@pytest.mark.parametrize("snapshot_id", ["", "has space", "../escape", "name/child"])
def test_snapshot_id_rejects_unsafe_values(snapshot_id: str) -> None:
    coordinator = MemorySnapshotCoordinator([FakeEnvironment()], noop_action=lambda: None)
    with pytest.raises(ValueError, match="snapshot_id"):
        coordinator.reset_all(snapshot_id)
