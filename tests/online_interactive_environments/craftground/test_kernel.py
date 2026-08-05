"""内核测试；全部连真实 CraftGround JVM，不使用环境替身。"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from online_interactive_environments import ActionTick
from online_interactive_environments.craftground import (
    EnvironmentKernel,
    EnvironmentPoolTimeout,
    RolloutRequest,
    SnapshotRegion,
)

pytestmark = pytest.mark.craftground

REGION = SnapshotRegion((-16, -64, -16), (16, 319, 16))


@pytest.fixture(scope="module")
def kernel() -> Iterator[EnvironmentKernel]:
    """整个模块共用两个槽位；JVM 启动昂贵，不按用例重启。"""
    with EnvironmentKernel.launch(
        slots=2,
        port_base=18740,
        image_width=160,
        image_height=90,
        use_shared_memory=False,
    ) as launched:
        yield launched


def test_launch_gives_every_slot_an_independent_runtime(kernel: EnvironmentKernel) -> None:
    described = kernel.describe()

    assert kernel.capacity == 2
    assert described["action_backend"] == "keyboard_and_mouse_only"
    assert described["action_space"] == "V2_MINERL_HUMAN"
    assert len({slot["runtime_path"] for slot in described["slots"]}) == 2
    assert len({slot["port"] for slot in described["slots"]}) == 2
    for slot in described["slots"]:
        assert Path(slot["runtime_path"]).is_dir()


def test_launch_leaves_the_world_loaded_rather_than_on_the_loading_screen() -> None:
    """`reset` 返回时地形仍在加载；launch 必须 warmup 到真实画面。"""
    import numpy as np

    with EnvironmentKernel.launch(
        slots=1,
        port_base=18790,
        image_width=160,
        image_height=90,
        use_shared_memory=False,
    ) as kernel:
        loaded = kernel.handles()[0].observe()["rgb"]
        # 加载界面几乎是均匀暗色；加载完成后画面有明显的亮度分布。
        assert float(np.asarray(loaded).std()) > 8.0


def test_warmup_rejects_negative_ticks(kernel: EnvironmentKernel) -> None:
    with pytest.raises(ValueError, match="warmup_ticks"):
        kernel.handles()[0].warmup(-1)


def test_handle_applies_ticks_and_returns_environment_facts(kernel: EnvironmentKernel) -> None:
    with kernel.lease() as handle:
        assert not hasattr(handle, "step")
        outcome = handle.apply(ActionTick(("W", "MouseMove", "0", "300")))

        assert outcome.inputs == ("W", "MouseMove", "0", "300")
        assert outcome.native_action["forward"] is True
        assert outcome.native_action["camera_pitch"] == pytest.approx(45.0)
        assert outcome.step_elapsed_ms > 0.0
        assert outcome.observation["rgb"].shape == (90, 160, 3)
        assert handle.observe() is outcome.observation


def test_consecutive_ticks_change_the_rendered_frame(kernel: EnvironmentKernel) -> None:
    import numpy as np

    with kernel.lease() as handle:
        first = handle.apply(ActionTick(("MouseMove", "0", "0"))).observation["rgb"].copy()
        for _ in range(4):
            latest = handle.apply(ActionTick(("MouseMove", "60", "0"))).observation["rgb"]

        assert not np.array_equal(first, latest)


def test_hotbar_state_is_per_slot_and_cleared_by_reset(kernel: EnvironmentKernel) -> None:
    first, second = kernel.handles()
    first.apply(ActionTick(("4",)))

    assert first.selected_hotbar == 4
    assert second.selected_hotbar == 1

    kernel.capture("hotbar-root", region=REGION, as_root=True)
    kernel.reset()

    assert first.selected_hotbar == 1


def test_preview_adapter_does_not_mutate_slot_device_state(kernel: EnvironmentKernel) -> None:
    handle = kernel.handles()[0]
    handle.apply(ActionTick(("3",)))
    preview = handle.preview_adapter()
    preview.convert(ActionTick(("7",)))

    assert preview.selected_hotbar == 7
    assert handle.selected_hotbar == 3


def test_capture_and_reset_restore_every_slot(kernel: EnvironmentKernel) -> None:
    snapshot = kernel.capture("restore-root", region=REGION, as_root=True)
    timings = kernel.reset(snapshot)

    assert snapshot.snapshot_id == "restore-root"
    assert kernel.root_snapshot is snapshot
    assert timings.wall_ms > 0.0
    assert len(timings.worker_ms) == 2


def test_rollout_gives_each_subagent_an_exclusive_slot(kernel: EnvironmentKernel) -> None:
    kernel.capture("rollout-root", region=REGION, as_root=True)
    lock = threading.Lock()
    busy: set[int] = set()
    peak = 0
    concurrent = 0

    def simulate(handle: object, payload: int) -> str:
        nonlocal peak, concurrent
        slot = handle.slot  # type: ignore[attr-defined]
        with lock:
            assert slot not in busy
            busy.add(slot)
            concurrent += 1
            peak = max(peak, concurrent)
        for _ in range(3):
            handle.apply(ActionTick(("W",)))  # type: ignore[attr-defined]
        with lock:
            busy.discard(slot)
            concurrent -= 1
        return f"slot-{slot}:{payload}"

    requests = [
        RolloutRequest(f"request-{index}", f"subagent-{index}", index, simulate)
        for index in range(4)
    ]
    results = kernel.rollout(requests, wait_timeout=120.0)

    assert [result.request_id for result in results] == [f"request-{index}" for index in range(4)]
    assert peak == 2
    assert all(result.restore_ms > 0.0 for result in results)
    assert all(result.rollout_ms > 0.0 for result in results)


def test_lease_times_out_when_every_slot_is_busy(kernel: EnvironmentKernel) -> None:
    with kernel.lease(), kernel.lease(), pytest.raises(EnvironmentPoolTimeout):
        kernel.lease(timeout=0.01)


def test_reset_without_a_root_snapshot_is_rejected() -> None:
    with (
        EnvironmentKernel.launch(
            slots=1,
            port_base=18760,
            image_width=160,
            image_height=90,
            use_shared_memory=False,
        ) as kernel,
        pytest.raises(RuntimeError, match="根快照"),
    ):
        kernel.reset()


def test_close_is_idempotent_and_blocks_further_control() -> None:
    kernel = EnvironmentKernel.launch(
        slots=1,
        port_base=18770,
        image_width=160,
        image_height=90,
        use_shared_memory=False,
    )
    kernel.close()
    kernel.close()

    with pytest.raises(RuntimeError, match="内核已关闭"):
        kernel.lease()
