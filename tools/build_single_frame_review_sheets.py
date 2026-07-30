"""为单帧意图题生成六轮候选终点审核联系表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from datasets.minestudio_data.load import TrajectoryReader


OFFSETS = (4, 8, 12, 20, 40, 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-sheet", type=int, default=5)
    arguments = parser.parse_args()

    questions = [
        json.loads(line)
        for line in (arguments.dataset_dir / "questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    questions = [row for row in questions if row["task_type"] == "single_frame_intent_to_action"]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    reader = TrajectoryReader([arguments.raw_dataset_dir], ["action", "image"], 320, 180)
    try:
        for sheet_start in range(0, len(questions), arguments.per_sheet):
            batch = questions[sheet_start:sheet_start + arguments.per_sheet]
            sheet = Image.new("RGB", (7 * 320, len(batch) * 220), "white")
            draw = ImageDraw.Draw(sheet)
            for row_index, question in enumerate(batch):
                episode = question["source"]["episode"]
                start = question["source"]["image_frames"][0]
                episode_length = reader.episode_length(episode)
                frames = [start] + [min(start + offset, episode_length - 1) for offset in OFFSETS]
                images = reader.readers["image"].read_frames(episode, min(frames), max(frames) - min(frames) + 1)
                for column, frame in enumerate(frames):
                    image = Image.fromarray(images[frame - min(frames)]).convert("RGB")
                    sheet.paste(image, (column * 320, row_index * 220))
                    label = "start" if column == 0 else f"+{OFFSETS[column - 1]}"
                    draw.rectangle((column * 320, row_index * 220, column * 320 + 90, row_index * 220 + 18), fill="black")
                    draw.text((column * 320 + 3, row_index * 220 + 3), label, fill="white")
                draw.text((3, row_index * 220 + 184), question["id"], fill="black")
                draw.text((3, row_index * 220 + 199), question["inputs"].get("intent", ""), fill="black")
            sheet.save(arguments.output_dir / f"single_{sheet_start:03d}_{sheet_start + len(batch) - 1:03d}.jpg", quality=90)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
