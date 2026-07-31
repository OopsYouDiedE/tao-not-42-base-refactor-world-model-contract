"""验证单个常驻 CraftGround 进程中的 Minecraft 内存结构快照。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from game_environment import (
    MemorySnapshotCoordinator,
    SnapshotRegion,
    build_environment,
    step_commands,
)

EXPECTED_MARKERS = {
    "MEMORY_BLOCK_OK",
    "MEMORY_WATER_OK",
    "MEMORY_CHEST_OK",
    "MEMORY_FURNACE_ITEMS_OK",
    "MEMORY_FURNACE_BURN_OK",
    "MEMORY_FURNACE_COOK_OK",
}

SETUP_COMMANDS = (
    "gamerule doDaylightCycle false",
    "gamerule doWeatherCycle false",
    "gamerule doMobSpawning false",
    "gamerule randomTickSpeed 0",
    "fill 0 60 0 8 70 8 minecraft:air",
    "fill 0 63 0 8 63 8 minecraft:bedrock",
    "tp @s 4 64 4 0 0",
    "setblock 1 64 1 minecraft:stone",
    "setblock 2 64 1 minecraft:oak_planks",
    "setblock 3 64 1 minecraft:chest",
    "item replace block 3 64 1 container.0 with minecraft:diamond 7",
    "item replace block 3 64 1 container.1 with minecraft:apple 5",
    "setblock 4 64 1 minecraft:furnace[facing=south]",
    "item replace block 4 64 1 container.0 with minecraft:raw_iron 3",
    "item replace block 4 64 1 container.1 with minecraft:coal 2",
    "tick freeze",
    "data merge block 4 64 1 {BurnTime:200s,CookTime:40s,CookTimeTotal:200s}",
    "fill 5 64 0 7 64 2 minecraft:stone",
    "setblock 6 65 1 minecraft:water",
    "kill @e[tag=memory_snapshot_probe]",
    (
        "summon minecraft:cow 7 64 7 "
        '{NoAI:1b,PersistenceRequired:1b,Health:7.0f,Tags:["memory_snapshot_probe"]}'
    ),
)

MUTATION_COMMANDS = (
    "setblock 1 64 1 minecraft:air",
    "setblock 2 64 1 minecraft:lava",
    "setblock 3 64 1 minecraft:air",
    "setblock 4 64 1 minecraft:blast_furnace",
    "fill 5 64 0 7 66 2 minecraft:air",
    "kill @e[tag=memory_snapshot_probe]",
)

ASSERTION_COMMANDS = (
    "execute if block 1 64 1 minecraft:stone run say MEMORY_BLOCK_OK",
    "execute if block 6 65 1 minecraft:water run say MEMORY_WATER_OK",
    (
        "execute if data block 3 64 1 "
        '{Items:[{Slot:0b,id:"minecraft:diamond",count:7},'
        '{Slot:1b,id:"minecraft:apple",count:5}]} run say MEMORY_CHEST_OK'
    ),
    (
        "execute if data block 4 64 1 "
        '{Items:[{Slot:0b,id:"minecraft:raw_iron",count:3},'
        '{Slot:1b,id:"minecraft:coal",count:2}]} run say MEMORY_FURNACE_ITEMS_OK'
    ),
    "execute if data block 4 64 1 {BurnTime:200s} run say MEMORY_FURNACE_BURN_OK",
    "execute if data block 4 64 1 {CookTime:40s} run say MEMORY_FURNACE_COOK_OK",
)


def _messages(observation: Any) -> list[str]:
    return [message.message for message in observation["full"].chat_messages]


def verify(runtime: Path) -> dict[str, Any]:
    environment = build_environment(runtime, image_width=160, image_height=90, port=18200)
    try:
        environment.reset(options={"fast_reset": False})
        step_commands(environment, SETUP_COMMANDS, ticks=20)
        coordinator = MemorySnapshotCoordinator([environment])
        capture_start = time.perf_counter()
        snapshot = coordinator.capture_all(
            "relative_advantage",
            SnapshotRegion((0, 63, 0), (8, 68, 8)),
        )
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
        step_commands(environment, MUTATION_COMMANDS, ticks=10)
        reset_timings = coordinator.reset_all(snapshot)
        restored_observation = step_commands(environment, (), ticks=8)
        assertion_messages: list[str] = []
        for assertion in ASSERTION_COMMANDS:
            asserted_observation = step_commands(environment, (assertion,), ticks=2)
            assertion_messages.extend(_messages(asserted_observation))
    finally:
        environment.close()

    messages = assertion_messages
    found_markers = sorted(
        marker for marker in EXPECTED_MARKERS if any(marker in message for message in messages)
    )
    report = {
        "capture_ms": capture_ms,
        "restore_messages": _messages(restored_observation),
        "reset_wall_ms": reset_timings.wall_ms,
        "reset_worker_ms": list(reset_timings.worker_ms),
        "reset_under_one_second": reset_timings.wall_ms < 1000.0,
        "assertion_messages": messages,
        "expected_markers": sorted(EXPECTED_MARKERS),
        "found_markers": found_markers,
        "restored_exactly": set(found_markers) == EXPECTED_MARKERS,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 CraftGround 内存结构快照恢复")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.runtime.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["restored_exactly"] or not report["reset_under_one_second"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
