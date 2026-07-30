"""Minecraft 世界目录的不可变快照与工作副本恢复。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SNAPSHOT_VERSION = 1
_IGNORED_WORLD_FILES = {"session.lock"}


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    display_name: str
    world_folder: str
    created_at: str
    files: dict[str, str]
    state_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = SNAPSHOT_VERSION


class WorldSnapshotStore:
    """管理经过哈希校验的不可变世界快照。"""

    def __init__(self, root: os.PathLike[str] | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        snapshot_id: str,
        world_dir: os.PathLike[str] | str,
        *,
        display_name: str,
        state_digest: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SnapshotManifest:
        """归档已由游戏进程 flush 的完整世界目录。"""
        _validate_name(snapshot_id, "snapshot_id")
        source = Path(world_dir).resolve()
        _validate_world_dir(source)
        target = self.root / snapshot_id
        if target.exists():
            raise FileExistsError(f"快照已存在：{target}")

        staging = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=self.root))
        try:
            copied_world = staging / "world"
            shutil.copytree(source, copied_world, ignore=_ignore_world_files)
            manifest = SnapshotManifest(
                snapshot_id=snapshot_id,
                display_name=display_name,
                world_folder=source.name,
                created_at=datetime.now(timezone.utc).isoformat(),
                files=_hash_tree(copied_world),
                state_digest=state_digest,
                metadata=dict(metadata or {}),
            )
            _atomic_json_dump(staging / "snapshot.json", asdict(manifest))
            os.replace(staging, target)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def manifest(self, snapshot_id: str) -> SnapshotManifest:
        _validate_name(snapshot_id, "snapshot_id")
        with (self.root / snapshot_id / "snapshot.json").open(encoding="utf-8") as manifest_file:
            data = json.load(manifest_file)
        if data.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"不支持的快照版本：{data.get('version')}")
        return SnapshotManifest(**data)

    def verify(self, snapshot_id: str) -> SnapshotManifest:
        manifest = self.manifest(snapshot_id)
        if _hash_tree(self.root / snapshot_id / "world") != manifest.files:
            raise ValueError(f"快照文件校验失败：{snapshot_id}")
        return manifest

    def restore(
        self,
        snapshot_id: str,
        saves_dir: os.PathLike[str] | str,
        *,
        slot_name: str | None = None,
        replace: bool = False,
    ) -> tuple[Path, SnapshotManifest]:
        """原子地生成一个可供新游戏进程加载的工作副本。"""
        manifest = self.verify(snapshot_id)
        name = slot_name or manifest.world_folder
        _validate_name(name, "slot_name")
        saves = Path(saves_dir).resolve()
        saves.mkdir(parents=True, exist_ok=True)
        target = (saves / name).resolve()
        if target.parent != saves:
            raise ValueError("工作世界必须是 saves_dir 的直接子目录")
        if target.exists() and not replace:
            raise FileExistsError(f"工作世界已存在：{target}")

        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=saves))
        shutil.rmtree(staging)
        try:
            shutil.copytree(self.root / snapshot_id / "world", staging)
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target, manifest


def discover_world_dir(saves_dir: os.PathLike[str] | str, preferred_folder: str | None = None) -> Path:
    root = Path(saves_dir).resolve()
    if preferred_folder is not None:
        _validate_name(preferred_folder, "preferred_folder")
        world = root / preferred_folder
        _validate_world_dir(world)
        return world
    worlds = sorted(path for path in root.iterdir() if path.is_dir() and (path / "level.dat").is_file())
    if len(worlds) != 1:
        raise RuntimeError(f"无法唯一定位工作世界：{root} 下找到 {len(worlds)} 个")
    return worlds[0]


def _hash_tree(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path.name in _IGNORED_WORLD_FILES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.relative_to(root).as_posix()] = digest.hexdigest()
    return files


def _ignore_world_files(_directory: str, names: Sequence[str]) -> list[str]:
    return [name for name in names if name in _IGNORED_WORLD_FILES]


def _validate_name(value: str, field_name: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field_name} 必须是不含路径分隔符的非空名称：{value!r}")


def _validate_world_dir(path: Path) -> None:
    if not path.is_dir() or not (path / "level.dat").is_file():
        raise FileNotFoundError(f"不是完整 Minecraft 世界目录：{path}")


def _atomic_json_dump(path: Path, data: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
