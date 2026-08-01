import pytest

from game_environment import RollingActionQueue, replan_remaining_ticks
from tao.protocols.action import ActionTick


def _chunks(count: int) -> tuple[ActionTick, ...]:
    return tuple(ActionTick(keys=("W",)) for _ in range(count))


@pytest.mark.parametrize(
    ("length", "remaining"),
    [(8, 4), (12, 4), (16, 4), (20, 5), (40, 10)],
)
def test_replan_threshold_has_four_tick_floor(length: int, remaining: int) -> None:
    assert replan_remaining_ticks(length) == remaining


def test_short_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少需要 8 tick"):
        replan_remaining_ticks(7)


def test_queue_preserves_old_prefix_and_replaces_future() -> None:
    queue = RollingActionQueue()
    queue.submit("old", _chunks(12), start_tick=0, current_tick=0)
    result = queue.submit("new", _chunks(8), start_tick=6, current_tick=3)

    assert result.accepted_ticks == 8
    assert [queue.pop(tick).plan_id for tick in range(3, 6)] == ["old"] * 3
    assert queue.pop(6).plan_id == "new"


def test_late_plan_drops_expired_prefix() -> None:
    queue = RollingActionQueue()
    result = queue.submit("late", _chunks(8), start_tick=10, current_tick=13)

    assert result.expired_ticks == 3
    assert result.accepted_ticks == 5
    assert queue.pop(13).plan_index == 3


def test_status_requests_replan_at_dynamic_threshold() -> None:
    queue = RollingActionQueue()
    result = queue.submit("plan", _chunks(20), start_tick=10, current_tick=10)

    assert result.replan_tick == 25
    assert queue.status(24)["should_replan"] is False
    assert queue.status(25)["should_replan"] is True
