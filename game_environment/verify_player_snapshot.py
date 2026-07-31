"""连续验证 CraftGround 世界与玩家快照的同起点恢复合同。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from game_environment import MemorySnapshotCoordinator, SnapshotRegion, build_environment, step_commands


def _inventory(full: Any) -> list[dict[str, int | str]]:
    return [
        {
            "raw_id": int(stack.raw_id),
            "translation_key": stack.translation_key,
            "count": int(stack.count),
            "durability": int(stack.durability),
            "max_durability": int(stack.max_durability),
        }
        for stack in full.inventory
    ]


def _state(observation: Any) -> dict[str, Any]:
    full = observation["full"]
    return {
        "x": float(full.x),
        "y": float(full.y),
        "z": float(full.z),
        "yaw": float(full.yaw),
        "pitch": float(full.pitch),
        "health": float(full.health),
        "food_level": float(full.food_level),
        "saturation_level": float(full.saturation_level),
        "experience": int(full.experience),
        "velocity": [float(full.velocity_x), float(full.velocity_y), float(full.velocity_z)],
        "status_effects": sorted(
            (effect.translation_key, int(effect.amplifier), int(effect.duration))
            for effect in full.status_effects
        ),
        "inventory": _inventory(full),
    }


def _compare(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    scalar_tolerance = {
        "x": 0.01,
        "y": 0.01,
        "z": 0.01,
        "yaw": 0.01,
        "pitch": 0.01,
        "health": 0.01,
        "food_level": 0.01,
        "saturation_level": 0.01,
    }
    differences: dict[str, Any] = {}
    for key, tolerance in scalar_tolerance.items():
        delta = abs(actual[key] - expected[key])
        if delta > tolerance:
            differences[key] = {"expected": expected[key], "actual": actual[key], "delta": delta}
    for key in ("experience", "inventory"):
        if actual[key] != expected[key]:
            differences[key] = {"expected": expected[key], "actual": actual[key]}
    # 效果持续时间会随同步 tick 减少；类型和倍率必须一致。
    expected_effects = [(name, amplifier) for name, amplifier, _ in expected["status_effects"]]
    actual_effects = [(name, amplifier) for name, amplifier, _ in actual["status_effects"]]
    if actual_effects != expected_effects:
        differences["status_effects"] = {"expected": expected_effects, "actual": actual_effects}
    return differences


def verify(runtime: Path, attempts: int = 8) -> dict[str, Any]:
    environment = build_environment(runtime, image_width=320, image_height=180, port=18900)
    try:
        environment.reset(options={"fast_reset": False})
        baseline_observation = step_commands(
            environment,
            (
                "gamemode survival @s",
                "gamerule doDaylightCycle false",
                "gamerule doWeatherCycle false",
                "fill -2 80 -2 2 83 2 minecraft:air",
                "fill -2 79 -2 2 79 2 minecraft:stone",
                "clear @s",
                "give @s minecraft:diamond_pickaxe[damage=37] 1",
                "give @s minecraft:bread 13",
                "experience set @s 7 levels",
                "effect give @s minecraft:night_vision 600 1 true",
                "tp @s 0.5 80 0.5 37 -21",
            ),
            ticks=8,
        )
        expected = _state(baseline_observation)
        coordinator = MemorySnapshotCoordinator([environment], synchronization_ticks=3)
        snapshot = coordinator.capture_all("player_state_eight_restore", SnapshotRegion((-2, 79, -2), (2, 83, 2)))
        attempts_report: list[dict[str, Any]] = []
        for index in range(attempts):
            step_commands(
                environment,
                (
                    "clear @s",
                    "effect clear @s",
                    "experience set @s 0 levels",
                    f"tp @s {20 + index} 95 {20 + index} 180 45",
                    "damage @s 3 minecraft:generic",
                ),
                ticks=4,
            )
            timings = coordinator.reset_all(snapshot)
            restored = step_commands(environment, (), ticks=3)
            actual = _state(restored)
            differences = _compare(expected, actual)
            attempts_report.append(
                {
                    "attempt": index + 1,
                    "reset_wall_ms": timings.wall_ms,
                    "position": [actual["x"], actual["y"], actual["z"]],
                    "differences": differences,
                    "passed": not differences,
                }
            )
        return {
            "attempts": attempts,
            "expected": expected,
            "results": attempts_report,
            "all_passed": all(item["passed"] for item in attempts_report),
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="连续八次验证玩家状态快照")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    report = verify(args.runtime.resolve(), args.attempts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"attempts": report["attempts"], "all_passed": report["all_passed"]}, ensure_ascii=False))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
