"""下载一个 MineStudio 组，并按全局帧序号读取。"""

from __future__ import annotations

import io
import pickle
import shutil
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import av
import lmdb
import numpy as np
from huggingface_hub import snapshot_download

REPOS = {
    name: f"CraftJarvis/minestudio-data-{name}-v110"
    for name in ("6xx", "7xx", "8xx", "9xx", "10xx")
}
MODALITIES = ("action", "image", "meta_info", "event", "segmentation", "motion")


def load(
    datasets: list[str],
    modalities: list[str],
    force_remove_other_dataset_in_this_group: bool = False,
    output: Path = Path("runs/datasets"),
) -> MineStudioDataset:
    if len(datasets) != 1:
        raise ValueError("一次只能下载一个数据集")
    repo = REPOS[datasets[0]]
    root = Path(output)
    target = root / repo.split("/")[-1]
    if force_remove_other_dataset_in_this_group:
        for path in root.glob("minestudio-data-*-v110"):
            if path != target:
                shutil.rmtree(path)
        for name in MODALITIES:
            if name not in modalities and (target / name).exists():
                shutil.rmtree(target / name)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=[f"{name}/*" for name in modalities],
    )
    return MineStudioDataset(target, modalities)


class MineStudioDataset:
    def __init__(self, root: Path, modalities: list[str]) -> None:
        self.root, self.modalities = Path(root), modalities
        self._dbs: dict[str, dict[Path, Any]] = {}
        self._episodes: dict[str, dict[str, tuple[Path, int, int]]] = {}
        self._chunks: dict[str, int] = {}
        self._cache: OrderedDict[tuple[str, Path, int, int], Any] = OrderedDict()
        self.keys: list[str] = []
        self.lengths: dict[str, int] = {}
        self._ends: list[int] = []

    def updata_index(self) -> MineStudioDataset:
        self.close()
        self.keys, self.lengths, self._ends = [], {}, []
        for modality in self.modalities:
            dbs, episodes, chunk_size = {}, {}, 0
            for part in sorted((self.root / modality).glob("part-*")):
                if not (part / "data.mdb").is_file():
                    continue
                db = lmdb.open(str(part), readonly=True, lock=False, readahead=False)
                with db.begin() as transaction:
                    size = int(pickle.loads(transaction.get(b"__chunk_size__")))
                    infos = pickle.loads(transaction.get(b"__chunk_infos__"))
                if chunk_size and size != chunk_size:
                    raise ValueError(f"{modality} 的 chunk 大小不一致")
                chunk_size, dbs[part] = size, db
                for info in infos:
                    key = info["episode"]
                    if key in episodes:
                        raise ValueError(f"重复轨迹：{key}")
                    episodes[key] = (part, int(info["episode_idx"]), int(info["num_frames"]))
            if not dbs:
                raise FileNotFoundError(self.root / modality)
            self._dbs[modality] = dbs
            self._episodes[modality] = episodes
            self._chunks[modality] = chunk_size
        common = set.intersection(*(set(items) for items in self._episodes.values()))
        self.keys = sorted(common)
        self.lengths = {
            key: min(self._episodes[name][key][2] for name in self.modalities)
            for key in self.keys
        }
        total = 0
        self._ends = []
        for key in self.keys:
            total += self.lengths[key]
            self._ends.append(total)
        return self

    update_index = updata_index

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
            **{name: self._first(name, value) for name, value in values.items()},
        }

    @staticmethod
    def _first(modality: str, value: Any) -> Any:
        if modality == "action":
            return {key: item[0] for key, item in value.items()}
        return value[0]

    def read(self, key: str, start: int, length: int) -> dict[str, Any]:
        length = min(length, self.lengths[key] - start)
        return {name: self._read_modality(name, key, start, length) for name in self.modalities}

    def read_modality(self, modality: str, key: str, start: int, length: int) -> Any:
        return self._read_modality(modality, key, start, min(length, self.lengths[key] - start))

    def _read_modality(self, modality: str, key: str, start: int, length: int) -> Any:
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
        cache_key = (modality, part, episode, frame)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        with self._dbs[modality][part].begin() as transaction:
            value = transaction.get(str((episode, frame)).encode())
        if value is None:
            raise KeyError(cache_key)
        if modality == "image":
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
        for dbs in self._dbs.values():
            for db in dbs.values():
                db.close()
        self._dbs.clear()
        self._episodes.clear()
        self._cache.clear()
