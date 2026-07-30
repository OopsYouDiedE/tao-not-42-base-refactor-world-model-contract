"""为已有 CraftGround 图像轨迹补建逐帧 tick 清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(run_directory: Path) -> dict:
    candidates = json.loads(
        (run_directory / "policy_candidates.json").read_text(encoding="utf-8")
    )
    manifest = {
        "snapshot_id": "advantage_batch4_start",
        "task": candidates["task"],
        "trajectories": [],
    }
    for trajectory_index, trajectory in enumerate(candidates["trajectories"]):
        frame_id = 0
        tick = 0
        frames = [{
            "trajectory_id": trajectory["id"],
            "frame_id": 0,
            "tick": 0,
            "segment": None,
            "action": {},
            "reason": "trajectory_start",
            "path": f"trajectory_{trajectory_index}/frame_0000.png",
        }]
        for segment_index, segment in enumerate(trajectory["segments"]):
            for _ in range(int(segment["ticks"])):
                tick += 1
                if tick % 5 == 0:
                    frame_id += 1
                    frames.append({
                        "trajectory_id": trajectory["id"],
                        "frame_id": frame_id,
                        "tick": tick,
                        "segment": segment_index,
                        "action": segment["action"],
                        "reason": "periodic",
                        "path": f"trajectory_{trajectory_index}/frame_{frame_id:04d}.png",
                    })
            frame_id += 1
            frames.append({
                "trajectory_id": trajectory["id"],
                "frame_id": frame_id,
                "tick": tick,
                "segment": segment_index,
                "action": segment["action"],
                "reason": "segment_end",
                "path": f"trajectory_{trajectory_index}/frame_{frame_id:04d}.png",
            })
        missing = [item["path"] for item in frames if not (run_directory / item["path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"缺少轨迹图片: {missing}")
        manifest["trajectories"].append({
            "trajectory_id": trajectory["id"],
            "description": trajectory["description"],
            "frames": frames,
        })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="补建 CraftGround 逐帧清单")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run_directory = args.run.resolve()
    manifest = build(run_directory)
    destination = run_directory / "trajectory_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
