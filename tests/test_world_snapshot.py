from pathlib import Path

import pytest

from game_environment.world_snapshot import WorldSnapshotStore, discover_world_dir


def _make_world(path: Path, *, level: bytes = b"level") -> None:
    (path / "region").mkdir(parents=True)
    (path / "playerdata").mkdir()
    (path / "level.dat").write_bytes(level)
    (path / "region" / "r.0.0.mca").write_bytes(b"region")
    (path / "playerdata" / "player.dat").write_bytes(b"player")
    (path / "session.lock").write_bytes(b"lock")


def test_snapshot_is_immutable_verified_and_restorable(tmp_path: Path) -> None:
    source = tmp_path / "source" / "world-a"
    _make_world(source)
    store = WorldSnapshotStore(tmp_path / "snapshots")

    manifest = store.capture("branch-point", source, display_name="Training World")
    (source / "level.dat").write_bytes(b"mutated")
    restored, verified = store.restore("branch-point", tmp_path / "work" / "saves")

    assert verified == manifest
    assert (restored / "level.dat").read_bytes() == b"level"
    assert (restored / "playerdata" / "player.dat").is_file()
    assert not (restored / "session.lock").exists()
    with pytest.raises(FileExistsError):
        store.capture("branch-point", source, display_name="Training World")


def test_tampered_snapshot_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source" / "world-a"
    _make_world(source)
    store = WorldSnapshotStore(tmp_path / "snapshots")
    store.capture("branch-point", source, display_name="Training World")
    (tmp_path / "snapshots" / "branch-point" / "world" / "level.dat").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="校验失败"):
        store.restore("branch-point", tmp_path / "work" / "saves")


def test_discover_world_requires_an_unambiguous_world(tmp_path: Path) -> None:
    saves = tmp_path / "saves"
    _make_world(saves / "world-a")
    assert discover_world_dir(saves) == (saves / "world-a").resolve()
    _make_world(saves / "world-b")
    with pytest.raises(RuntimeError, match="找到 2 个"):
        discover_world_dir(saves)
