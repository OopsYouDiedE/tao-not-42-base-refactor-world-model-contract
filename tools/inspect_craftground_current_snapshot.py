"""读取未布置场景的真实 CraftGround 当前世界，为课程生成提供证据。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from game_environment import CraftGroundActionAdapter, save_rgb, step_commands

STAT_BLOCKS = ("oak_log", "birch_log", "spruce_log", "stone", "coal_ore", "iron_ore")


def build_environment(runtime: Path) -> Any:
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.screen_encoding_modes import ScreenEncodingMode

    config = InitialEnvironmentConfig(
        image_width=640,
        image_height=360,
        seed="424242",
        render_distance=6,
        simulation_distance=6,
        request_raycast=True,
        requires_surrounding_blocks=True,
        mined_stat_keys=list(STAT_BLOCKS),
        screen_encoding_mode=ScreenEncodingMode.RAW,
    )
    return CraftGroundEnvironment(
        config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        env_path=str(runtime),
        port=18700,
        find_free_port=True,
        cleanup_world=False,
        verbose=False,
    )


def collect(observation: Any) -> dict[str, Any]:
    full = observation["full"]
    raycast = full.raycast_result
    target = getattr(raycast, "target_block", None)
    items = Counter()
    for stack in full.inventory:
        if int(stack.count) > 0 and str(stack.translation_key) != "item.minecraft.air":
            items[str(stack.translation_key)] += int(stack.count)
    blocks = [
        {
            "translation_key": str(block.translation_key),
            "x": int(block.x),
            "y": int(block.y),
            "z": int(block.z),
        }
        for block in full.surrounding_blocks
    ]
    nearby_counts = Counter(entry["translation_key"] for entry in blocks)
    return {
        "position": {"x": round(float(full.x), 3), "y": round(float(full.y), 3), "z": round(float(full.z), 3), "yaw": round(float(full.yaw), 3), "pitch": round(float(full.pitch), 3)},
        "raycast": {
            "type": str(raycast.type),
            "block": str(getattr(target, "translation_key", "")),
            "position": None if target is None else {"x": int(target.x), "y": int(target.y), "z": int(target.z)},
        },
        "inventory": dict(sorted(items.items())),
        "mined_statistics": {str(key): int(value) for key, value in full.mined_statistics.items()},
        "nearby_block_counts": dict(sorted(nearby_counts.items())),
        "nearby_resource_blocks": [entry for entry in blocks if any(name in entry["translation_key"] for name in ("log", "ore", "stone"))][:200],
        "nearby_block_record_count": len(blocks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="勘察真实 CraftGround 当前世界")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    environment = build_environment(args.runtime.resolve())
    try:
        environment.reset(options={"fast_reset": False})
        observation = step_commands(environment, (), ticks=12)
        save_rgb(observation, output / "initial.png")
        scans = [{"direction": "初始朝向", "observation": collect(observation), "image": "initial.png"}]
        adapter = CraftGroundActionAdapter()
        for index, direction in enumerate(("右转90度", "右转180度", "右转270度"), start=1):
            observation = environment.step(adapter.convert((), (600, 0)))[0]
            image_name = f"scan_{index}.png"
            save_rgb(observation, output / image_name)
            scans.append({"direction": direction, "observation": collect(observation), "image": image_name})
        result = {"scans": scans}
        (output / "observation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
