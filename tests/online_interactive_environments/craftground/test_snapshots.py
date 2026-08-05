from __future__ import annotations

import pytest

from online_interactive_environments.craftground import SnapshotRegion


def test_snapshot_region_rejects_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="minimum"):
        SnapshotRegion((1, 0, 0), (0, 1, 1))


def test_snapshot_region_is_centered_on_actual_player_position() -> None:
    region = SnapshotRegion.around_player((106.8, 70.0, -32.2), horizontal_radius=8)

    assert region.minimum == (98, -64, -41)
    assert region.maximum == (114, 319, -25)


def test_snapshot_region_serializes_command_coordinates() -> None:
    region = SnapshotRegion((0, 63, 0), (8, 68, 8))

    assert region.command_coordinates() == "0 63 0 8 68 8"


@pytest.mark.parametrize("horizontal_radius", [0, -1])
def test_snapshot_region_rejects_non_positive_radius(horizontal_radius: int) -> None:
    with pytest.raises(ValueError, match="horizontal_radius"):
        SnapshotRegion.around_player((0.0, 70.0, 0.0), horizontal_radius=horizontal_radius)
