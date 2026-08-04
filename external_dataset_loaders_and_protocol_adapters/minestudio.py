"""Lazy-loading MineStudio LMDB reader and explicit dataset downloader."""

from __future__ import annotations

import io
import pickle
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Self

import numpy as np

from online_interactive_environments import ActionSequence, ActionTick
from shared_tools.datasets import download_dataset_snapshot

REPOS = {
    name: f"CraftJarvis/minestudio-data-{name}-v110"
    for name in ("6xx", "7xx", "8xx", "9xx", "10xx")
}
MODALITIES = ("action", "image", "meta_info", "event", "segmentation", "motion")
MOUSE_DELTA_LIMIT = 999
DEGREES_PER_PIXEL = 0.15
MINECRAFT_KEYMAP = {
    "forward": "W",
    "back": "S",
    "left": "A",
    "right": "D",
    "jump": "Space",
    "sneak": "Shift",
    "sprint": "Ctrl",
    "attack": "MouseLeft",
    "use": "MouseRight",
    "drop": "Q",
    "inventory": "E",
    **{f"hotbar.{slot}": str(slot) for slot in range(1, 10)},
}


def _clamp(value: float, limit: int) -> int:
    return max(-limit, min(limit, int(np.rint(value))))


def encode_minestudio_actions(
    actions: dict[str, np.ndarray],
    frames_per_tick: int = 1,
    keymap: dict[str, str] | None = None,
    degrees_per_pixel: float = DEGREES_PER_PIXEL,
    *,
    offset: int = 0,
) -> ActionSequence:
    """将 MineStudio 动作窗口直接转换为项目标准动作序列。"""
    if frames_per_tick < 1 or degrees_per_pixel <= 0:
        raise ValueError("frames_per_tick 和 degrees_per_pixel 必须为正数")
    if offset < 0:
        raise ValueError("offset 必须为非负整数")
    if "camera" not in actions:
        raise ValueError("actions 缺少 camera 字段")
    camera = np.asarray(actions["camera"], dtype=np.float64)
    if camera.ndim != 2 or camera.shape[1] != 2 or camera.shape[0] == 0:
        raise ValueError(f"camera 必须是非空 shape (T, 2)，实际为 {camera.shape}")
    if camera.shape[0] % frames_per_tick:
        raise ValueError("动作帧数不能被 frames_per_tick 整除")

    mapping = MINECRAFT_KEYMAP if keymap is None else keymap
    present = [(field, name) for field, name in mapping.items() if field in actions]
    held = None
    if present:
        fields = [np.asarray(actions[field]).astype(bool) for field, _ in present]
        if any(field.shape[0] != camera.shape[0] for field in fields):
            raise ValueError("按键帧数与 camera 帧数不一致")
        matrix = np.stack(fields)
        held = matrix.reshape(len(present), -1, frames_per_tick).any(axis=2).T

    names = [name for _, name in present]
    ticks = []
    grouped_camera = camera.reshape(-1, frames_per_tick, 2).sum(axis=1)
    for index, (pitch, yaw) in enumerate(grouped_camera):
        inputs = (
            []
            if held is None
            else [name for name, active in zip(names, held[index], strict=True) if active]
        )
        mouse = (
            _clamp(yaw / degrees_per_pixel, MOUSE_DELTA_LIMIT),
            _clamp(pitch / degrees_per_pixel, MOUSE_DELTA_LIMIT),
        )
        if mouse != (0, 0):
            inputs[0:0] = ["MouseMove", str(mouse[0]), str(mouse[1])]
        ticks.append(ActionTick(tuple(inputs)))
    return ActionSequence("KeyboardMouse", offset, tuple(ticks))


def load(
    dataset: str, modalities: list[str], output: Path = Path("runs/datasets")
) -> MineStudioDataset:
    if dataset not in REPOS:
        raise ValueError(f"unknown MineStudio dataset: {dataset!r}")
    unknown = set(modalities) - set(MODALITIES)
    if unknown:
        raise ValueError("unknown modalities: " + ", ".join(sorted(unknown)))
    target = Path(output) / REPOS[dataset].split("/")[-1]
    download_dataset_snapshot(
        REPOS[dataset],
        target,
        allow_patterns=tuple(f"{name}/*" for name in modalities),
    )
    return MineStudioDataset(target, modalities)


class MineStudioDataset:
    def __init__(self, root: Path, modalities: list[str]) -> None:
        self.root, self.modalities = Path(root), list(modalities)
        self._dbs: dict[str, dict[Path, Any]] = {}
        self._episodes: dict[str, dict[str, tuple[Path, int, int]]] = {}
        self._chunks: dict[str, int] = {}
        self._cache: OrderedDict[tuple[str, Path, int, int], Any] = OrderedDict()
        self.keys: list[str] = []
        self.lengths: dict[str, int] = {}
        self._ends: list[int] = []

    def update_index(self) -> MineStudioDataset:
        try:
            import lmdb
        except ImportError as error:
            raise RuntimeError("lmdb is required to read MineStudio data") from error
        self.close()
        self.keys, self.lengths, self._ends = [], {}, []
        for modality in self.modalities:
            dbs, episodes, chunk_size = {}, {}, 0
            for part in sorted((self.root / modality).glob("part-*")):
                if not (part / "data.mdb").is_file():
                    continue
                database = lmdb.open(str(part), readonly=True, lock=False, readahead=False)
                with database.begin() as transaction:
                    size = int(pickle.loads(transaction.get(b"__chunk_size__")))
                    infos = pickle.loads(transaction.get(b"__chunk_infos__"))
                if chunk_size and size != chunk_size:
                    raise ValueError(f"inconsistent chunk size for {modality}")
                chunk_size, dbs[part] = size, database
                for info in infos:
                    key = info["episode"]
                    if key in episodes:
                        raise ValueError(f"duplicate episode: {key}")
                    episodes[key] = part, int(info["episode_idx"]), int(info["num_frames"])
            if not dbs:
                raise FileNotFoundError(self.root / modality)
            self._dbs[modality], self._episodes[modality], self._chunks[modality] = (
                dbs,
                episodes,
                chunk_size,
            )
        common = set.intersection(*(set(episodes) for episodes in self._episodes.values()))
        self.keys = sorted(common)
        self.lengths = {
            key: min(self._episodes[name][key][2] for name in self.modalities) for key in self.keys
        }
        total = 0
        for key in self.keys:
            total += self.lengths[key]
            self._ends.append(total)
        return self

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect_right(self._ends, index)
        key = self.keys[position]
        frame = index - (self._ends[position - 1] if position else 0)
        values = self.read(key, frame, 1)
        return {
            "key": key,
            "frame": frame,
            **{
                name: (
                    {field: item[0] for field, item in value.items()}
                    if name == "action"
                    else value[0]
                )
                for name, value in values.items()
            },
        }

    def read(self, key: str, start: int, length: int) -> dict[str, Any]:
        return {name: self.read_modality(name, key, start, length) for name in self.modalities}

    def read_modality(self, modality: str, key: str, start: int, length: int) -> Any:
        if key not in self.lengths or start < 0:
            raise (
                KeyError(key) if key not in self.lengths else ValueError("start cannot be negative")
            )
        length = min(length, self.lengths[key] - start)
        part, episode, _ = self._episodes[modality][key]
        size, end = self._chunks[modality], start + length
        first = start // size * size
        chunks = [self._chunk(modality, part, episode, frame) for frame in range(first, end, size)]
        offset = start - first
        if modality == "image":
            return np.concatenate(chunks)[offset : offset + length]
        if modality == "meta_info":
            return [item for chunk in chunks for item in chunk][offset : offset + length]
        return {
            field: np.concatenate([chunk[field] for chunk in chunks])[offset : offset + length]
            for field in chunks[0]
        }

    def _chunk(self, modality: str, part: Path, episode: int, frame: int) -> Any:
        cache_key = modality, part, episode, frame
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        with self._dbs[modality][part].begin() as transaction:
            value = transaction.get(str((episode, frame)).encode())
        if value is None:
            raise KeyError(cache_key)
        if modality == "image":
            try:
                import av
            except ImportError as error:
                raise RuntimeError("PyAV is required to decode MineStudio images") from error
            with av.open(io.BytesIO(value)) as container:
                decoded = np.stack(
                    [item.to_ndarray(format="rgb24") for item in container.decode(video=0)]
                )
        else:
            decoded = pickle.loads(value)
        self._cache[cache_key] = decoded
        if len(self._cache) > 32:
            self._cache.popitem(last=False)
        return decoded

    def close(self) -> None:
        for databases in self._dbs.values():
            for database in databases.values():
                database.close()
        self._dbs.clear()
        self._episodes.clear()
        self._cache.clear()

    def __enter__(self) -> Self:
        return self.update_index()

    def __exit__(self, *_: object) -> None:
        self.close()
