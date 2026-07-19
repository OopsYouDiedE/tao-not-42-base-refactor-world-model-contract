"""管理 CraftGround 完整世界目录的不可变快照与工作副本。

对外接口：SnapshotManifest、WorldSnapshotStore、discover_world_dir。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SNAPSHOT_VERSION = 1
_IGNORED_WORLD_FILES = {"session.lock"}


@dataclass(frozen=True)
class SnapshotManifest:
    """不可变世界快照清单。

    Attributes:
        snapshot_id: 快照标识；标量字符串，Dtype 为 str。
        display_name: Minecraft 世界显示名；标量字符串，Dtype 为 str。
        world_folder: 工作副本目录名；标量字符串，Dtype 为 str。
        files: 相对路径到 SHA-256 的映射；Shape 为 [N]，键值 Dtype 为 str。
        state_digest: 保存时完整观测的可选指纹；标量字符串或 None。
        metadata: 调用方附加的 JSON 数据；标量映射。
    """

    snapshot_id: str
    display_name: str
    world_folder: str
    created_at: str
    files: Dict[str, str]
    state_digest: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = SNAPSHOT_VERSION


class WorldSnapshotStore:
    """保存完整 Minecraft 世界目录并恢复到独立工作槽。

    Args:
        root: 快照库根目录；标量路径，Dtype 为 path-like。
    """

    def __init__(self, root: os.PathLike[str] | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        snapshot_id: str,
        world_dir: os.PathLike[str] | str,
        *,
        display_name: str,
        state_digest: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SnapshotManifest:
        """复制一个已 flush 的完整世界为不可变快照。

        Args:
            snapshot_id: 快照标识；标量字符串，Dtype 为 str。
            world_dir: 含 ``level.dat`` 的世界目录；标量路径。
            display_name: 冷启动选择世界所需显示名；标量字符串。
            state_digest: 可选完整观测指纹；标量字符串或 None。
            metadata: 可选 JSON 元数据；标量映射。

        Returns:
            SnapshotManifest 标量对象。
        """
        validate_id(snapshot_id, "snapshot_id")
        source = Path(world_dir).resolve()
        _validate_world_dir(source)
        target = self.root / snapshot_id
        if target.exists():
            raise FileExistsError(f"快照已存在：{target}")

        tmp = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=self.root))
        try:
            copied = tmp / "world"
            shutil.copytree(source, copied, ignore=_ignore_world_files)
            manifest = SnapshotManifest(
                snapshot_id=snapshot_id,
                display_name=display_name,
                world_folder=source.name,
                created_at=datetime.now(timezone.utc).isoformat(),
                files=_hash_tree(copied),
                state_digest=state_digest,
                metadata=dict(metadata or {}),
            )
            atomic_json_dump(tmp / "snapshot.json", asdict(manifest))
            os.replace(tmp, target)
            return manifest
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def manifest(self, snapshot_id: str) -> SnapshotManifest:
        """读取快照清单并校验协议版本。

        Args:
            snapshot_id: 快照标识；标量字符串，Dtype 为 str。

        Returns:
            SnapshotManifest 标量对象。
        """
        validate_id(snapshot_id, "snapshot_id")
        with (self.root / snapshot_id / "snapshot.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"不支持的快照版本：{data.get('version')}")
        return SnapshotManifest(**data)

    def verify(self, snapshot_id: str) -> SnapshotManifest:
        """逐文件校验快照 SHA-256。

        Args:
            snapshot_id: 快照标识；标量字符串，Dtype 为 str。

        Returns:
            校验通过的 SnapshotManifest 标量对象。
        """
        manifest = self.manifest(snapshot_id)
        if _hash_tree(self.root / snapshot_id / "world") != manifest.files:
            raise ValueError(f"快照文件校验失败：{snapshot_id}")
        return manifest

    def restore(
        self,
        snapshot_id: str,
        saves_dir: os.PathLike[str] | str,
        *,
        slot_name: Optional[str] = None,
        replace: bool = False,
    ) -> Tuple[Path, SnapshotManifest]:
        """把快照恢复为 CraftGround 冷启动可选的完整世界。

        Args:
            snapshot_id: 快照标识；标量字符串，Dtype 为 str。
            saves_dir: 当前独立环境的 ``run/saves``；标量路径。
            slot_name: 工作副本目录名；标量字符串或 None。
            replace: 是否替换已有同名工作副本；标量 bool。

        Returns:
            ``(world_path, manifest)``；标量路径与清单对象。
        """
        manifest = self.verify(snapshot_id)
        name = slot_name or manifest.world_folder
        validate_id(name, "slot_name")
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


def discover_world_dir(
    saves_dir: os.PathLike[str] | str,
    preferred_folder: Optional[str] = None,
) -> Path:
    """定位 CraftGround ``run/saves`` 下的完整世界目录。

    Args:
        saves_dir: ``MinecraftEnv/run/saves``；标量路径。
        preferred_folder: 已知目录名；标量字符串或 None。

    Returns:
        含 ``level.dat`` 的绝对 Path 标量。
    """
    root = Path(saves_dir).resolve()
    if preferred_folder is not None:
        validate_id(preferred_folder, "preferred_folder")
        candidate = root / preferred_folder
        _validate_world_dir(candidate)
        return candidate
    worlds = sorted(p for p in root.iterdir() if p.is_dir() and (p / "level.dat").is_file())
    if len(worlds) != 1:
        raise RuntimeError(f"无法唯一定位工作世界：{root} 下找到 {len(worlds)} 个")
    return worlds[0]


def atomic_json_dump(path: Path, data: Mapping[str, Any]) -> None:
    """原子写入 JSON 文件；输入为标量路径与映射，返回 None。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def validate_id(value: str, field_name: str) -> None:
    """校验单段目录标识；输入为两个标量字符串，返回 None。"""
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field_name} 必须是不含路径分隔符的非空名称：{value!r}")


def _hash_tree(root: Path) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in _IGNORED_WORLD_FILES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.relative_to(root).as_posix()] = digest.hexdigest()
    return files


def _ignore_world_files(_directory: str, names: Sequence[str]) -> Iterable[str]:
    return [name for name in names if name in _IGNORED_WORLD_FILES]


def _validate_world_dir(path: Path) -> None:
    if not path.is_dir() or not (path / "level.dat").is_file():
        raise FileNotFoundError(f"不是完整 Minecraft 世界目录：{path}")
