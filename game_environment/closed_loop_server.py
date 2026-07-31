"""为多轮视觉策略提供常驻 CraftGround 回合服务。"""

from __future__ import annotations

import argparse
import json
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from game_environment import (
    RESET_PLAYER_COMMANDS,
    SCENE_COMMANDS,
    CraftGroundActionAdapter,
    MemorySnapshotCoordinator,
    RollingActionQueue,
    SnapshotRegion,
    build_environment,
    save_rgb,
    step_commands,
    validate_identifier,
)
from lumine.action_codec import decode_lumine_action

MAX_REQUEST_BYTES = 1_000_000


class ClosedLoopSession:
    def __init__(self, runtime: Path, output: Path, max_ticks: int, max_turns: int):
        if max_ticks < 1 or max_turns < 1:
            raise ValueError("max_ticks 和 max_turns 必须大于零")
        self.output = output
        self.max_ticks = max_ticks
        self.max_turns = max_turns
        self.environment = build_environment(runtime)
        self.action_adapter = CraftGroundActionAdapter()
        self.action_queue = RollingActionQueue()
        self.lock = threading.Lock()
        self.environment.reset(options={"fast_reset": False})
        step_commands(self.environment, SCENE_COMMANDS, ticks=20)
        initial = self.environment.step(self.action_adapter.reset())[0]
        self.output.mkdir(parents=True, exist_ok=True)
        save_rgb(initial, self.output / "initial.png")
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
            validate_identifier(trajectory_id, "trajectory_id")
            reset = self.coordinator.reset_all(self.snapshot)
            step_commands(self.environment, RESET_PLAYER_COMMANDS, ticks=8)
            observation = self.environment.step(self.action_adapter.reset())[0]
            self.trajectory_id = trajectory_id
            self.tick = 0
            self.turn = 0
            self.frame = 0
            self.simulation_wall_ms = 0.0
            self.records = []
            self.action_queue.clear()
            destination = self._frame_path()
            save_rgb(observation, destination)
            return self._status(destination, reset_wall_ms=reset.wall_ms)

    def enqueue_text(
        self,
        action_text: str,
        plan_id: str,
        start_tick: int | None,
        model_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """提交异步生成的动作计划，不阻塞环境推进。"""
        with self.lock:
            if self.trajectory_id is None:
                raise ValueError("必须先 reset")
            if self.turn >= self.max_turns:
                raise ValueError("模型指令次数预算已经耗尽")
            validate_identifier(plan_id, "plan_id")
            resolved_start_tick = self.tick if start_tick is None else start_tick
            decoded = decode_lumine_action(action_text)
            submission = self.action_queue.submit(
                plan_id,
                decoded.chunks,
                start_tick=resolved_start_tick,
                current_tick=self.tick,
            )
            self.turn += 1
            self.records.append(
                {
                    "kind": "plan",
                    "turn": self.turn,
                    "plan_id": plan_id,
                    "submitted_at_tick": self.tick,
                    "start_tick": resolved_start_tick,
                    "plan_ticks": submission.plan_ticks,
                    "accepted_ticks": submission.accepted_ticks,
                    "expired_ticks": submission.expired_ticks,
                    "replan_tick": submission.replan_tick,
                    "action_text": action_text,
                    "model": model_metadata,
                }
            )
            self._write_trajectory()
            return self._status(self._frame_path(), submission=submission.__dict__)

    def advance(self, ticks: int = 1) -> dict[str, Any]:
        """消费动作队列并推进环境；队列为空时执行释放所有控制的安全动作。"""
        if not 1 <= ticks <= 64:
            raise ValueError("ticks 必须位于 1 到 64")
        with self.lock:
            if self.trajectory_id is None:
                raise ValueError("必须先 reset")
            allowed_ticks = min(ticks, self.max_ticks - self.tick)
            if allowed_ticks < 1:
                raise ValueError("模拟 tick 预算已经耗尽")
            started = time.perf_counter()
            executed: list[dict[str, Any]] = []
            observation = None
            for _ in range(allowed_ticks):
                scheduled = self.action_queue.pop(self.tick)
                chunk = scheduled.chunk if scheduled is not None else None
                action = self.action_adapter.convert(
                    chunk.keys if chunk is not None else (),
                    chunk.mouse if chunk is not None else (0, 0),
                    chunk.scroll if chunk is not None else 0,
                )
                observation = self.environment.step(action)[0]
                executed.append(
                    {
                        "tick": self.tick,
                        "plan_id": scheduled.plan_id if scheduled is not None else None,
                        "plan_index": scheduled.plan_index if scheduled is not None else None,
                        "source": "plan" if scheduled is not None else "queue_empty",
                    }
                )
                self.tick += 1
                self.frame += 1
            simulation_ms = (time.perf_counter() - started) * 1000.0
            self.simulation_wall_ms += simulation_ms
            destination = self._frame_path()
            save_rgb(observation, destination)
            self.records.append(
                {
                    "kind": "advance",
                    "start_tick": self.tick - allowed_ticks,
                    "end_tick": self.tick,
                    "ticks": allowed_ticks,
                    "simulation_wall_ms": simulation_ms,
                    "executed": executed,
                    "frame": destination.name,
                }
            )
            self._write_trajectory()
            return self._status(destination)

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
                action = self.action_adapter.convert(chunk.keys, chunk.mouse, chunk.scroll)
                observation = self.environment.step(action)[0]
                self.tick += 1
                self.frame += 1
                destination = self._frame_path()
                save_rgb(observation, destination)
                frame_paths.append(destination.name)
                executed_chunks.append(
                    {
                        "keys": list(chunk.keys),
                        "mouse": list(chunk.mouse),
                        "scroll": chunk.scroll,
                        "selected_hotbar": self.action_adapter.selected_hotbar,
                    }
                )
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
            "selected_hotbar": self.action_adapter.selected_hotbar,
            "action_queue": self.action_queue.status(self.tick),
            "frame": str(frame_path.resolve()),
            **extra,
        }

    def _write_trajectory(self) -> None:
        path = self.output / self.trajectory_id / "trajectory.json"
        path.write_text(
            json.dumps(
                {
                    "snapshot_id": self.snapshot.snapshot_id,
                    "trajectory_id": self.trajectory_id,
                    "max_ticks": self.max_ticks,
                    "max_turns": self.max_turns,
                    "tick": self.tick,
                    "turn": self.turn,
                    "simulation_wall_ms": self.simulation_wall_ms,
                    "records": self.records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        with suppress(ConnectionError, OSError):
            self.environment.close()


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
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError(f"请求体必须在 1 到 {MAX_REQUEST_BYTES} 字节之间")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/reset":
                    result = session.reset(payload["trajectory_id"])
                elif self.path == "/step":
                    result = session.step_text(
                        str(payload["action_text"]),
                        dict(payload.get("model", {})),
                    )
                elif self.path == "/enqueue":
                    start_tick = payload.get("start_tick")
                    result = session.enqueue_text(
                        str(payload["action_text"]),
                        str(payload["plan_id"]),
                        int(start_tick) if start_tick is not None else None,
                        dict(payload.get("model", {})),
                    )
                elif self.path == "/advance":
                    result = session.advance(int(payload.get("ticks", 1)))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 CraftGround 多轮闭环服务")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18400)
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()
    session = ClosedLoopSession(
        args.runtime.resolve(), args.output.resolve(), args.max_ticks, args.max_turns
    )
    server = ThreadingHTTPServer((args.host, args.port), handler(session))
    try:
        server.serve_forever()
    finally:
        session.close()


if __name__ == "__main__":
    main()
