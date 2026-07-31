import pytest

from game_environment import (
    CraftGroundActionAdapter,
    lumine_chunk_to_v2_action,
    scroll_hotbar_slot,
)
from lumine.action_codec import decode_lumine_action


def _no_op_v2() -> dict[str, bool | float]:
    action: dict[str, bool | float] = {
        "forward": False,
        "left": False,
        "use": False,
        "attack": False,
        "camera_yaw": 0.0,
        "camera_pitch": 0.0,
    }
    action.update({f"hotbar.{slot}": False for slot in range(1, 10)})
    return action


def _chunk_action(keys: tuple[str, ...], mouse: tuple[int, int]) -> dict[str, bool | float]:
    return lumine_chunk_to_v2_action(keys, mouse, action_factory=_no_op_v2)


def test_project_action_chunk_maps_to_craftground_v2() -> None:
    chunk = decode_lumine_action(
        "<|action_start|> ; Mouse -20 10 W A MouseRight <|action_end|>"
    ).chunks[0]
    action = _chunk_action(chunk.keys, chunk.mouse)
    assert action["forward"] is True
    assert action["left"] is True
    assert action["use"] is True
    assert action["camera_yaw"] == -3.0
    assert action["camera_pitch"] == 1.5
    assert action["attack"] is False


def test_absent_key_releases_on_next_tick() -> None:
    chunks = decode_lumine_action("<|action_start|> ; W ; Mouse 2 -1 <|action_end|>").chunks
    first = _chunk_action(chunks[0].keys, chunks[0].mouse)
    second = _chunk_action(chunks[1].keys, chunks[1].mouse)
    assert first["forward"] is True
    assert second["forward"] is False
    assert second["camera_yaw"] == 0.3
    assert second["camera_pitch"] == -0.15


def test_positive_scroll_moves_up_and_wraps_across_hotbar() -> None:
    adapter = CraftGroundActionAdapter(action_factory=_no_op_v2)
    adapter.reset(1)
    action = adapter.convert((), (0, 0), 5)
    assert action["hotbar.5"] is True
    assert adapter.selected_hotbar == 5


def test_negative_scroll_moves_down_and_tracks_subsequent_ticks() -> None:
    adapter = CraftGroundActionAdapter(action_factory=_no_op_v2)
    adapter.reset(8)
    first = adapter.convert((), (0, 0), -3)
    second = adapter.convert((), (0, 0), 1)
    assert first["hotbar.2"] is True
    assert second["hotbar.1"] is True
    assert adapter.selected_hotbar == 1


def test_explicit_hotbar_key_updates_scroll_origin() -> None:
    adapter = CraftGroundActionAdapter(action_factory=_no_op_v2)
    explicit = adapter.convert(("7",), (0, 0))
    scrolled = adapter.convert((), (0, 0), 2)
    assert explicit["hotbar.7"] is True
    assert scrolled["hotbar.5"] is True


def test_scroll_and_explicit_hotbar_key_are_rejected_in_same_tick() -> None:
    adapter = CraftGroundActionAdapter(action_factory=_no_op_v2)
    with pytest.raises(ValueError, match="同时"):
        adapter.convert(("3",), (0, 0), 1)


def test_scroll_requires_known_hotbar_for_stateless_conversion() -> None:
    with pytest.raises(ValueError, match="当前快捷栏槽位"):
        lumine_chunk_to_v2_action((), (0, 0), 1, action_factory=_no_op_v2)


@pytest.mark.parametrize(
    ("selected", "scroll", "expected"),
    [(1, 1, 9), (9, -1, 1), (4, 5, 8), (4, -5, 9)],
)
def test_scroll_hotbar_slot_wraps(selected: int, scroll: int, expected: int) -> None:
    assert scroll_hotbar_slot(selected, scroll) == expected
