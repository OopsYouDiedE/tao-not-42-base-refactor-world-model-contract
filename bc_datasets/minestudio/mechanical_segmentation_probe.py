"""随机抽样验证动作序列的机械切分与左右外拓。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bc_datasets.minestudio.lmdb_modal_reader import TrajectoryReader


KEYS = ("forward", "back", "left", "right", "jump", "attack", "use")


def frame_label(actions: dict[str, np.ndarray], index: int) -> tuple:
    keys = tuple(name for name in KEYS if int(actions[name][index]) != 0)
    pitch, yaw = actions["camera"][index]
    direction = (int(np.sign(pitch)), int(np.sign(yaw)))
    return keys, direction


def mechanical_segments(actions: dict[str, np.ndarray], minimum_length: int = 4) -> list[tuple[int, int]]:
    labels = [frame_label(actions, index) for index in range(len(actions["camera"]))]
    boundaries = [0]
    for index in range(1, len(labels)):
        if labels[index] != labels[index - 1]:
            boundaries.append(index)
    boundaries.append(len(labels))
    raw = [(left, right) for left, right in zip(boundaries, boundaries[1:]) if right - left >= minimum_length]
    merged: list[tuple[int, int]] = []
    for left, right in raw:
        if merged and left - merged[-1][1] < minimum_length:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    return merged


def intent_label(actions: dict[str, np.ndarray], index: int) -> tuple:
    """返回可容忍短暂停顿的动作意图标签。"""
    keys = tuple(name for name in KEYS if int(actions[name][index]) != 0)
    pitch, yaw = actions["camera"][index]
    return keys, (int(np.sign(pitch)), int(np.sign(yaw)))


def intent_preserving_segments(
    actions: dict[str, np.ndarray], minimum_length: int = 4, bridge_frames: int = 2,
) -> list[tuple[int, int]]:
    """合并短暂零输入与短过渡，避免切断完整复合动作意图。"""
    labels = [intent_label(actions, index) for index in range(len(actions["camera"]))]
    runs: list[list[object]] = []
    for index, label in enumerate(labels):
        if runs and runs[-1][2] == label:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1, label])
    changed = True
    while changed:
        changed = False
        for index in range(1, len(runs) - 1):
            if runs[index][1] - runs[index][0] > bridge_frames:
                continue
            if runs[index - 1][2] == runs[index + 1][2]:
                runs[index - 1][1] = runs[index + 1][1]
                del runs[index:index + 2]
                changed = True
                break
    return [(int(left), int(right)) for left, right, _ in runs if int(right) - int(left) >= minimum_length]


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--expand", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    reader = TrajectoryReader([args.dataset_dir], ["action", "image"], frame_width=320, frame_height=180)
    records = []
    try:
        for sample_index in range(args.samples):
            episode = rng.choice(reader.episode_names())
            length = reader.episode_length(episode)
            start = rng.randint(0, length - 81)
            actions = reader.readers["action"].read_frames(episode, start, 80)
            segments = intent_preserving_segments(actions)
            if not segments:
                continue
            left, right = rng.choice(segments)
            expanded_left, expanded_right = max(0, left - args.expand), min(80, right + args.expand)
            labels = [intent_label(actions, i) for i in range(80)]
            core_labels = labels[left:right]
            purity = sum(label == core_labels[0] for label in core_labels) / len(core_labels)
            expansion_changes = sum(labels[i] != labels[i - 1] for i in range(expanded_left + 1, expanded_right))
            frames = reader.readers["image"].read_frames(episode, start + expanded_left, expanded_right - expanded_left)
            canvas = Image.new("RGB", (640, 180), "black")
            canvas.paste(Image.fromarray(frames[0]), (0, 0)); canvas.paste(Image.fromarray(frames[-1]), (320, 0))
            draw = ImageDraw.Draw(canvas); draw.text((4, 4), f"before {start+expanded_left}", fill="white"); draw.text((324, 4), f"after {start+expanded_right-1}", fill="white")
            image_path = args.output_dir / f"sample_{sample_index:02d}.jpg"; canvas.save(image_path, quality=92)
            records.append({"episode": episode, "start_frame": start, "core": [start + left, start + right], "expanded": [start + expanded_left, start + expanded_right], "label": core_labels[0], "core_purity": purity, "expansion_label_changes": expansion_changes, "image": str(image_path)})
    finally:
        reader.close()
    (args.output_dir / "report.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(records), "mean_core_purity": float(np.mean([r["core_purity"] for r in records])) if records else 0.0, "mean_expansion_changes": float(np.mean([r["expansion_label_changes"] for r in records])) if records else 0.0, "report": str(args.output_dir / "report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
