"""行为克隆 conversation 与 MineStudio 流式数据集。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from behavior_cloning_dataset_converters import build_split
from external_dataset_loaders_and_protocol_adapters import (
    MineStudioDataset,
    encode_minestudio_actions,
)
from online_interactive_environments import format_action_sequence, parse_action_sequence_strict

DEFAULT_INSTRUCTION = (
    "You control a Minecraft player. Given the chronological observations, return only one "
    "standard-input-action/v1 sequence using Device KeyboardMouse, Tick lines, and <action> "
    "blocks. Every accepted tick is one 50 ms environment step."
)


def build_conversation(
    sample: dict[str, Any], root: Path = Path("."), *, instruction: str = DEFAULT_INSTRUCTION
) -> dict[str, Any]:
    from PIL import Image

    images = sample.get("images")
    if images is None:
        images = []
        for value in sample.get("image_paths", ()):
            path = root / value
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as source:
                images.append(source.convert("RGB").copy())
    if not images:
        raise ValueError("behavior-cloning sample requires at least one image")
    action_text = str(sample["action_text"])
    parse_action_sequence_strict(action_text)
    prompt = str(sample.get("prompt") or instruction)
    previous = sample.get("previous_action_text")
    if previous:
        parse_action_sequence_strict(str(previous))
        prompt += f"\nPreviously executed sequence:\n{previous}"
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
        ]
    }


def load_conversations(path: Path, *, maximum_samples: int | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    if source.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("h5py is required to load HDF5 behavior-cloning data") from error
        with h5py.File(source, "r") as archive:
            values = archive["conversations"]
            rows = [
                json.loads(item.decode() if isinstance(item, bytes) else item) for item in values
            ]
    else:
        with source.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    if maximum_samples is not None:
        rows = rows[:maximum_samples]
    return [build_conversation(row, source.parent) for row in rows]


@dataclass(frozen=True)
class StreamingSettings:
    window_frames: int = 8
    frames_per_tick: int = 1
    stride_frames: int = 8
    history_frames: int = 0
    image_width: int = 224
    image_height: int = 224

    def __post_init__(self) -> None:
        if min(self.window_frames, self.frames_per_tick, self.stride_frames) < 1:
            raise ValueError("window, tick and stride sizes must be positive")
        if self.window_frames % self.frames_per_tick:
            raise ValueError("window_frames must be divisible by frames_per_tick")


class MineStudioBehaviorCloningDataset:
    def __init__(
        self, root: Path, positions: list[tuple[str, int]], settings: StreamingSettings
    ) -> None:
        self.root, self.positions, self.settings = Path(root), positions, settings
        self._reader: MineStudioDataset | None = None

    def __len__(self) -> int:
        return len(self.positions)

    def _get_reader(self) -> MineStudioDataset:
        if self._reader is None:
            self._reader = MineStudioDataset(self.root, ["action", "image"]).update_index()
        return self._reader

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        episode, start = self.positions[index]
        reader = self._get_reader()
        actions = reader.read_modality("action", episode, start, self.settings.window_frames)
        sequence = encode_minestudio_actions(actions, self.settings.frames_per_tick)
        frame_indices = list(range(max(0, start - self.settings.history_frames), start + 1))
        images = []
        for frame_index in frame_indices:
            frame = reader.read_modality("image", episode, frame_index, 1)[0]
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB")
            images.append(image.resize((self.settings.image_width, self.settings.image_height)))
        return build_conversation(
            {"images": images, "action_text": format_action_sequence(sequence)}
        )

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None


def build_streaming_datasets(
    root: Path,
    *,
    settings: StreamingSettings | None = None,
    validation_ratio: float = 0.1,
    holdout_level: str = "prefix",
    seed: int = 3407,
    maximum_samples: int | None = None,
) -> tuple[MineStudioBehaviorCloningDataset, MineStudioBehaviorCloningDataset, dict[str, Any]]:
    resolved = settings or StreamingSettings()
    reader = MineStudioDataset(Path(root), ["action", "image"]).update_index()
    try:
        lengths = dict(reader.lengths)
    finally:
        reader.close()
    split = build_split(
        episode_frames=lengths,
        holdout_level=holdout_level,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    def positions(episodes: list[str]) -> list[tuple[str, int]]:
        values = [
            (episode, start)
            for episode in episodes
            for start in range(
                resolved.history_frames,
                lengths[episode] - resolved.window_frames + 1,
                resolved.stride_frames,
            )
        ]
        return values if maximum_samples is None else values[:maximum_samples]

    train, validation = positions(split.train_episodes), positions(split.validation_episodes)
    return (
        MineStudioBehaviorCloningDataset(root, train, resolved),
        MineStudioBehaviorCloningDataset(root, validation, resolved),
        {
            "train_samples": len(train),
            "validation_samples": len(validation),
            "holdout_level": split.holdout_level,
            "protocol": "standard-input-action/v1",
        },
    )
