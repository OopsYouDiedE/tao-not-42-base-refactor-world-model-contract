"""在 WSL CraftGround 中创建可分发给并行环境的固定基准存档。"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from online_interactive_environments.craftground import (
    create_environment,
    directory_sha256,
)

BASELINE_COMMANDS = (
    "gamerule doDaylightCycle false",
    "gamerule doWeatherCycle false",
    "gamerule doMobSpawning false",
    "gamerule randomTickSpeed 0",
    "time set day",
    "weather clear",
    "kill @e[type=!minecraft:player]",
    "clear @p",
    "spawnpoint @p",
    "save-all flush",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state(info: Any) -> dict[str, Any]:
    full = info["full"]
    return {
        "position": [float(full.x), float(full.y), float(full.z)],
        "yaw": float(full.yaw),
        "pitch": float(full.pitch),
        "health": float(full.health),
        "inventory": [str(value) for value in full.inventory],
    }


def create_baseline_world(
    output_directory: Path,
    *,
    port: int,
    warmup_ticks: int = 20,
) -> Path:
    """创建固定世界、等待完整落盘并导出不可变基准副本。"""
    from craftground.environment.action_space import no_op_v2

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = create_environment(
        port=port,
        instance_id=f"baseline-source-{uuid.uuid4().hex}",
        use_shared_memory=False,
        cleanup_world=False,
        verbose=False,
    )
    runtime_path = Path(environment.tao_runtime_path)
    source_world = runtime_path / "run" / "saves" / "New World"
    destination = output_directory / "baseline-world"
    started_at = None
    monotonic_started_at = None
    info = None
    error = None
    executed_ticks = 0
    try:
        _observation, info = environment.reset(options={"fast_reset": False})
        started_at = _utc_now()
        monotonic_started_at = time.perf_counter()
        for _ in range(warmup_ticks):
            _observation, _, _, _, info = environment.step(no_op_v2())
            executed_ticks += 1
        environment.add_commands(list(BASELINE_COMMANDS))
        for _ in range(8):
            _observation, _, _, _, info = environment.step(no_op_v2())
            executed_ticks += 1
        if not (source_world / "level.dat").is_file():
            raise FileNotFoundError(f"运行中的世界缺少 level.dat: {source_world}")
        if destination.exists():
            raise FileExistsError(f"基准存档输出已存在: {destination}")
        shutil.copytree(
            source_world,
            destination,
            ignore=shutil.ignore_patterns("session.lock"),
        )
        if not (destination / "level.dat").is_file():
            raise RuntimeError("导出的基准存档缺少 level.dat")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        environment.close()
        finished_at = _utc_now()
        duration_seconds = (
            time.perf_counter() - monotonic_started_at if monotonic_started_at is not None else 0.0
        )
        report = {
            "wall_clock_started_at": started_at,
            "wall_clock_finished_at": finished_at,
            "wall_clock_duration_seconds": duration_seconds,
            "environment_transport_backend": "socket",
            "action_protocol": "backend-native-command-plus-noop/v1",
            "commands": list(BASELINE_COMMANDS),
            "executed_ticks": executed_ticks,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "exception": error,
            "source_runtime_path": str(runtime_path),
            "final_state": _state(info) if info is not None else None,
        }
        (output_directory / "baseline-creation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "world_directory_name": "New World",
        "world_display_name": "New World",
        "source_world_path": str(source_world),
        "baseline_world_path": str(destination),
        "baseline_world_sha256": directory_sha256(destination),
        "created_at": _utc_now(),
    }
    manifest_path = output_directory / "baseline-world-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    arguments = parser.parse_args()
    print(
        create_baseline_world(
            arguments.output,
            port=arguments.port,
            warmup_ticks=arguments.warmup_ticks,
        )
    )


if __name__ == "__main__":
    main()
