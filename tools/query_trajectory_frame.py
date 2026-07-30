"""按轨迹 ID 和 frame/tick 查询真实 RGB 图片。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game_environment.trajectory_store import TrajectoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="查询已保存的 CraftGround 轨迹帧")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trajectory", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--frame", type=int)
    selector.add_argument("--tick", type=int)
    args = parser.parse_args()

    store = TrajectoryStore(args.run)
    if args.frame is not None:
        frame = store.frame(args.trajectory, args.frame)
    else:
        frame = store.at_or_before_tick(args.trajectory, args.tick)
    print(json.dumps({
        "snapshot_id": store.snapshot_id,
        "trajectory_id": frame.trajectory_id,
        "frame_id": frame.frame_id,
        "tick": frame.tick,
        "segment": frame.segment,
        "action": frame.action,
        "reason": frame.reason,
        "path": str(frame.path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
