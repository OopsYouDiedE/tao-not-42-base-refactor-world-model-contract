"""为多轮视觉策略提供常驻 CraftGround 回合服务。"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from game_environment import MemorySnapshotCoordinator, SnapshotRegion
from datasets.action_codec import DEGREES_PER_PIXEL, MINECRAFT_KEYMAP, decode_lumine_action
from tools.capture_craftground_advantage_scene import SCENE_COMMANDS, build_environment
from tools.run_craftground_image_advantage_batch import RESET_PLAYER_COMMANDS, _action, _save_rgb, _step_commands


class ClosedLoopSession:
    def __init__(self, runtime: Path, output: Path, max_ticks: int, max_turns: int):
        self.output = output
        self.max_ticks = max_ticks
        self.max_turns = max_turns
        self.environment = build_environment(runtime)
        self.lock = threading.Lock()
        self.environment.reset(options={"fast_reset": False})
        initial = _step_commands(self.environment, SCENE_COMMANDS, ticks=20)
        self.output.mkdir(parents=True, exist_ok=True)
        _save_rgb(initial, self.output / "initial.png")
        self.coordinator = MemorySnapshotCoordinator([self.environment])
        self.snapshot = self.coordinator.capture_all(
            "closed_loop_start",
            SnapshotRegion((0, 63, 0), (8, 68, 10)),
        )
        self.trajectory_id: str | None = None
        self.tick = 0
        self.turn = 0
        self.frame = 0
        self.simulation_wall_ms = 0.0
        self.records: list[dict[str, Any]] = []

    def reset(self, trajectory_id: str) -> dict[str, Any]:
        with self.lock:
            reset = self.coordinator.reset_all(self.snapshot)
            observation = _step_commands(self.environment, RESET_PLAYER_COMMANDS, ticks=8)
            self.trajectory_id = trajectory_id
            self.tick = 0
            self.turn = 0
            self.frame = 0
            self.simulation_wall_ms = 0.0
            self.records = []
            destination = self._frame_path()
            _save_rgb(observation, destination)
            return self._status(destination, reset_wall_ms=reset.wall_ms)

    def step_text(self, action_text: str, model_metadata: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.trajectory_id is None:
                raise ValueError("必须先 reset")
            if self.turn >= self.max_turns:
                raise ValueError("模型指令次数预算已经耗尽")
            decoded = decode_lumine_action(action_text)
            allowed_ticks = min(len(decoded.chunks), self.max_ticks - self.tick)
            if allowed_ticks < 1:
                raise ValueError("模拟 tick 预算已经耗尽")
            started = time.perf_counter()
            observation = None
            frame_paths: list[str] = []
            executed_chunks: list[dict[str, Any]] = []
            for chunk in decoded.chunks[:allowed_ticks]:
                action = _chunk_action(chunk.keys, chunk.mouse)
                observation = self.environment.step(action)[0]
                self.tick += 1
                self.frame += 1
                destination = self._frame_path()
                _save_rgb(observation, destination)
                frame_paths.append(destination.name)
                executed_chunks.append({
                    "keys": list(chunk.keys),
                    "mouse": list(chunk.mouse),
                    "scroll": chunk.scroll,
                })
            simulation_ms = (time.perf_counter() - started) * 1000.0
            self.simulation_wall_ms += simulation_ms
            self.turn += 1
            record = {
                "turn": self.turn,
                "start_tick": self.tick - allowed_ticks,
                "end_tick": self.tick,
                "ticks": allowed_ticks,
                "action_text": action_text,
                "chunks": executed_chunks,
                "model": model_metadata,
                "simulation_wall_ms": simulation_ms,
                "frames": frame_paths,
            }
            self.records.append(record)
            self._write_trajectory()
            return self._status(destination)

    def _frame_path(self) -> Path:
        return self.output / self.trajectory_id / f"frame_{self.frame:04d}.png"

    def _status(self, frame_path: Path, **extra: Any) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "trajectory_id": self.trajectory_id,
            "turn": self.turn,
            "tick": self.tick,
            "remaining_turns": self.max_turns - self.turn,
            "remaining_ticks": self.max_ticks - self.tick,
            "simulation_seconds": self.tick / 20.0,
            "simulation_wall_ms": self.simulation_wall_ms,
            "frame": str(frame_path.resolve()),
            **extra,
        }

    def _write_trajectory(self) -> None:
        path = self.output / self.trajectory_id / "trajectory.json"
        path.write_text(json.dumps({
            "snapshot_id": self.snapshot.snapshot_id,
            "trajectory_id": self.trajectory_id,
            "max_ticks": self.max_ticks,
            "max_turns": self.max_turns,
            "tick": self.tick,
            "turn": self.turn,
            "simulation_wall_ms": self.simulation_wall_ms,
            "records": self.records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        try:
            self.environment.close()
        except (ConnectionError, OSError):
            pass


def handler(session: ClosedLoopSession):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._reply(200, {"status": "ok"})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/reset":
                    result = session.reset(payload["trajectory_id"])
                elif self.path == "/step":
                    result = session.step_text(
                        str(payload["action_text"]), dict(payload.get("model", {})),
                    )
                else:
                    self._reply(404, {"error": "not found"})
                    return
                self._reply(200, result)
            except Exception as error:
                self._reply(400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


_INVERSE_KEYMAP = {name: field for field, name in MINECRAFT_KEYMAP.items()}


def _chunk_action(keys: tuple[str, ...], mouse: tuple[int, int]) -> dict[str, bool | float]:
    overrides: dict[str, bool | float] = {
        _INVERSE_KEYMAP[key]: True for key in keys if key in _INVERSE_KEYMAP
    }
    overrides["camera_yaw"] = mouse[0] * DEGREES_PER_PIXEL
    overrides["camera_pitch"] = mouse[1] * DEGREES_PER_PIXEL
    return _action(overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 CraftGround 多轮闭环服务")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18400)
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()
    session = ClosedLoopSession(args.runtime.resolve(), args.output.resolve(), args.max_ticks, args.max_turns)
    server = ThreadingHTTPServer((args.host, args.port), handler(session))
    try:
        server.serve_forever()
    finally:
        session.close()


if __name__ == "__main__":
    main()
