"""在 CraftGround 后端执行十秒原生动作并保存测试记录。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_interactive_environments.craftground import create_environment  # noqa: E402


def _observation_summary(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"type": type(observation).__name__}
    summary: dict[str, Any] = {"keys": sorted(observation)}
    for key in ("position", "yaw", "pitch", "health", "food_level", "is_dead"):
        value = observation.get(key)
        if value is not None and isinstance(value, (bool, int, float, str, list, tuple)):
            summary[key] = value
    rgb = observation.get("rgb")
    if rgb is not None:
        summary["rgb_shape"] = list(rgb.shape)
    return summary


def _save_rgb(observation: Any, path: Path) -> bool:
    if not isinstance(observation, dict) or observation.get("rgb") is None:
        return False
    Image.fromarray(observation["rgb"]).save(path)
    return True


def _info_summary(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"type": type(info).__name__}
    summary: dict[str, Any] = {"keys": sorted(info)}
    summary.update(
        (key, value)
        for key, value in info.items()
        if isinstance(value, (bool, int, float, str)) and len(str(value)) <= 1_000
    )
    return summary


def _action_for_tick(tick: int) -> tuple[str, dict[str, bool | float]]:
    from craftground.environment.action_space import no_op_v2

    action = no_op_v2()
    phase = tick % 40
    if phase < 20:
        action.update(forward=True, sprint=True)
        name = "forward+sprint"
    elif phase < 30:
        action.update(camera_yaw=4.0)
        name = "turn-right"
    else:
        action.update(forward=True, jump=True)
        name = "forward+jump"
    return name, action


def run(
    output_directory: Path,
    duration_seconds: float,
    *,
    runtime_path: Path | None = None,
    port: int = 18300,
    use_shared_memory: bool = True,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = create_environment(
        runtime_path=runtime_path,
        port=port,
        use_shared_memory=use_shared_memory,
        verbose=False,
    )
    record: dict[str, Any] = {
        "test_kind": "craftground_v2_native_backend_action_test",
        "protocol_closed_loop": False,
        "requested_duration_seconds": duration_seconds,
        "ticks": [],
    }
    try:
        observation, reset_info = environment.reset(options={"fast_reset": False})
        _save_rgb(observation, output_directory / "start.png")
        record["reset_info"] = _info_summary(reset_info)
        record["start_observation"] = _observation_summary(observation)
        started_at = time.monotonic()
        record["wall_clock_started_at"] = datetime.now(timezone.utc).isoformat()
        deadline = started_at + duration_seconds
        tick = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        while time.monotonic() < deadline and not (terminated or truncated):
            action_name, action = _action_for_tick(tick)
            step_started_at = time.monotonic()
            observation, reward, terminated, truncated, info = environment.step(action)
            completed_at = time.monotonic()
            total_reward += float(reward)
            record["ticks"].append(
                {
                    "tick": tick,
                    "action": action_name,
                    "started_offset_seconds": step_started_at - started_at,
                    "completed_offset_seconds": completed_at - started_at,
                    "step_duration_seconds": completed_at - step_started_at,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _info_summary(info),
                }
            )
            tick += 1
        finished_at = time.monotonic()
        record.update(
            wall_clock_finished_at=datetime.now(timezone.utc).isoformat(),
            actual_duration_seconds=finished_at - started_at,
            completed_ticks=tick,
            total_reward=total_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            end_observation=_observation_summary(observation),
        )
        _save_rgb(observation, output_directory / "end.png")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record_path = output_directory / "result.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        environment.close()
    return record_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--runtime-path", type=Path)
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--socket-ipc", action="store_true")
    arguments = parser.parse_args()
    print(
        run(
            arguments.output,
            arguments.duration,
            runtime_path=arguments.runtime_path,
            port=arguments.port,
            use_shared_memory=not arguments.socket_ipc,
        )
    )


if __name__ == "__main__":
    main()
