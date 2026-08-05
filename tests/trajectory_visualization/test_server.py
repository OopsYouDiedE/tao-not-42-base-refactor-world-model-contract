"""控制台测试；内核为真实 CraftGround，界面通过 HTTP 操控它。"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from online_interactive_environments.craftground import (
    DEFAULT_ACTION_SEQUENCE,
    EnvironmentKernel,
)
from trajectory_visualization import create_app

pytestmark = pytest.mark.craftground


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with EnvironmentKernel.launch(
        slots=2,
        port_base=18800,
        image_width=160,
        image_height=90,
        use_shared_memory=False,
    ) as kernel:
        kernel.capture("console-root", horizontal_radius=8, as_root=True)
        with TestClient(create_app(kernel)) as test_client:
            yield test_client


def _wait_until_idle(client: TestClient, slot: int, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get(f"/api/instances/{slot}").json()
        if not state["running"]:
            return state
        time.sleep(0.05)
    raise AssertionError(f"槽位 {slot} 未在 {timeout}s 内结束执行")


def test_page_lists_every_instance(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "CraftGround 实例控制台" in response.text
    assert "slot-0" in response.text
    assert "slot-1" in response.text
    assert "W Space MouseLeft x60" in response.text


def test_describe_reports_kernel_and_session_state(client: TestClient) -> None:
    described = client.get("/api/instances").json()

    assert described["action_backend"] == "keyboard_and_mouse_only"
    assert described["default_sequence"] == DEFAULT_ACTION_SEQUENCE
    assert described["root_snapshot"] == "console-root"
    assert [instance["slot"] for instance in described["instances"]] == [0, 1]
    first = described["instances"][0]
    assert first["instance_id"] == "slot-0"
    assert first["underflow"] == "wait"
    assert first["max_overrun_ticks"] == 0
    assert first["initialization"]["port"] == 18800
    assert first["stats"]["executed_ticks"] == 0


def test_submitting_the_default_sequence_advances_the_slot(client: TestClient) -> None:
    submitted = client.post("/api/instances/0/submit", json={"sequence": DEFAULT_ACTION_SEQUENCE})

    assert submitted.status_code == 200
    accepted = submitted.json()
    # 偏移 60 加 60 + 40 个动作 tick；Observe 自身不占 tick。
    assert accepted["start_tick"] == 60
    assert accepted["accepted_ticks"] == 100

    state = _wait_until_idle(client, 0)
    assert state["stats"]["executed_ticks"] == 100
    assert state["stats"]["observe_ticks"] == 1
    assert state["stats"]["overrun_ticks"] == 0
    assert state["current_tick"] == 160
    assert state["buffered_ticks"] == 0
    assert state["last_error"] is None


def test_stream_returns_a_frame_of_the_configured_size(client: TestClient) -> None:
    with client.stream("GET", "/api/instances/0/stream") as response:
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]
        chunk = next(response.iter_bytes(chunk_size=8192))

    assert b"Content-Type: image/jpeg" in chunk
    assert b"\xff\xd8" in chunk


def test_invalid_sequence_is_rejected_without_advancing(client: TestClient) -> None:
    before = client.get("/api/instances/1").json()

    rejected = client.post(
        "/api/instances/1/submit",
        json={"sequence": "Device Gamepad\nTick 0\n<action>A</action>"},
    )

    assert rejected.status_code == 400
    assert "设备" in rejected.json()["detail"]
    after = client.get("/api/instances/1").json()
    assert after["current_tick"] == before["current_tick"]
    assert after["stats"]["executed_ticks"] == before["stats"]["executed_ticks"]


def test_control_writes_underflow_and_budget_into_the_compiler(client: TestClient) -> None:
    updated = client.post(
        "/api/instances/1/control",
        json={"underflow": "repeat_last", "max_overrun_ticks": 3},
    ).json()

    assert updated["underflow"] == "repeat_last"
    assert updated["max_overrun_ticks"] == 3
    assert updated["unlimited_overrun"] is False

    unlimited = client.post(
        "/api/instances/1/control",
        json={"underflow": "noop", "unlimited_overrun": True},
    ).json()

    assert unlimited["underflow"] == "noop"
    assert unlimited["max_overrun_ticks"] is None
    assert unlimited["unlimited_overrun"] is True


def test_overrun_budget_bounds_execution_past_the_queue(client: TestClient) -> None:
    client.post("/api/instances/1/reset", json={"world": False})
    client.post(
        "/api/instances/1/control",
        json={"underflow": "repeat_last", "max_overrun_ticks": 4},
    )

    client.post(
        "/api/instances/1/submit",
        json={"sequence": "Device KeyboardMouse\nTick 0\n<action>W x2</action>"},
    )
    state = _wait_until_idle(client, 1)

    assert state["stats"]["executed_ticks"] == 6
    assert state["stats"]["overrun_ticks"] == 4
    assert state["overrun_exhausted"] is True


def test_reset_clears_statistics(client: TestClient) -> None:
    client.post(
        "/api/instances/1/control",
        json={"underflow": "wait", "max_overrun_ticks": 0},
    )
    reset = client.post("/api/instances/1/reset", json={"world": False}).json()

    assert reset["current_tick"] == 0
    assert reset["buffered_ticks"] == 0
    assert reset["stats"]["executed_ticks"] == 0
    assert reset["stats"]["submitted_ticks"] == 0
    assert reset["underflow"] == "wait"


def test_unknown_slot_is_reported(client: TestClient) -> None:
    assert client.get("/api/instances/9").status_code == 404
