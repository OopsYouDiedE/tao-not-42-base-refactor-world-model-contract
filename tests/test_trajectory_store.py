import json
from pathlib import Path

import pytest

from game_environment import TrajectoryStore


def _store(tmp_path: Path) -> TrajectoryStore:
    frame_dir = tmp_path / "trajectory_0"
    frame_dir.mkdir()
    (frame_dir / "frame_0000.png").write_bytes(b"png")
    (frame_dir / "frame_0001.png").write_bytes(b"png")
    manifest = {
        "snapshot_id": "snapshot-a",
        "trajectories": [{
            "trajectory_id": "T1",
            "frames": [
                {
                    "trajectory_id": "T1", "frame_id": 0, "tick": 0,
                    "segment": None, "action": {}, "reason": "trajectory_start",
                    "path": "trajectory_0/frame_0000.png",
                },
                {
                    "trajectory_id": "T1", "frame_id": 1, "tick": 5,
                    "segment": 0, "action": {"forward": True}, "reason": "periodic",
                    "path": "trajectory_0/frame_0001.png",
                },
            ],
        }],
    }
    (tmp_path / "trajectory_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return TrajectoryStore(tmp_path)


def test_get_exact_frame(tmp_path: Path) -> None:
    frame = _store(tmp_path).frame("T1", 1)
    assert frame.tick == 5
    assert frame.action == {"forward": True}
    assert frame.path.name == "frame_0001.png"


def test_get_latest_frame_at_or_before_tick(tmp_path: Path) -> None:
    frame = _store(tmp_path).at_or_before_tick("T1", 4)
    assert frame.frame_id == 0
    assert frame.tick == 0


def test_reject_unknown_trajectory(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="未知 trajectory_id"):
        _store(tmp_path).frame("missing", 0)
