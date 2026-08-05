"""CraftGround 实例控制台的 FastAPI 应用。

界面层只做三件事：把内核槽位列成实例、把 `ManualActionSession` 的执行结果转成 JSON
和 MJPEG、把控件上的值写回编译器。任何涉及动作编译、下溢策略、tick 预算或环境推进的
判断都在 `online_interactive_environments` 里，这里不重复实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, Thread

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from online_interactive_environments import UnderflowPolicy
from online_interactive_environments.craftground import (
    DEFAULT_ACTION_SEQUENCE,
    EnvironmentKernel,
    ManualActionSession,
)

from .frame_stream import BOUNDARY, FrameStream
from .page import render_page


class SubmitRequest(BaseModel):
    sequence: str = Field(default=DEFAULT_ACTION_SEQUENCE)


class ControlRequest(BaseModel):
    underflow: str | None = None
    max_overrun_ticks: int | None = None
    unlimited_overrun: bool = False


class ResetRequest(BaseModel):
    world: bool = False


@dataclass
class InstanceView:
    """一个槽位的界面侧视图：会话、帧流和一把串行化执行的锁。"""

    slot: int
    session: ManualActionSession
    stream: FrameStream
    lock: Lock
    running: bool = False
    last_error: str | None = None


class VisualizationService:
    """把一个内核包装成若干可独立操控的实例视图。"""

    def __init__(self, kernel: EnvironmentKernel) -> None:
        self.kernel = kernel
        self.instances: dict[int, InstanceView] = {}
        for handle in kernel.handles():
            self.instances[handle.slot] = InstanceView(
                slot=handle.slot,
                session=ManualActionSession(handle),
                stream=FrameStream(),
                lock=Lock(),
            )
            try:
                self.instances[handle.slot].stream.publish(handle.observe()["rgb"])
            except RuntimeError:
                # 槽位尚未产生观察，等第一个 tick 执行后再有画面。
                pass

    def view(self, slot: int) -> InstanceView:
        if slot not in self.instances:
            raise HTTPException(status_code=404, detail=f"没有槽位 {slot}")
        return self.instances[slot]

    def describe(self) -> dict[str, object]:
        described = self.kernel.describe()
        slots = {slot["slot"]: slot for slot in described["slots"]}  # type: ignore[index]
        return {
            "action_backend": described["action_backend"],
            "action_space": described["action_space"],
            "runtime_version": described["runtime_version"],
            "root_snapshot": described["root_snapshot"],
            "default_sequence": DEFAULT_ACTION_SEQUENCE,
            "instances": [self.snapshot(view, slots.get(view.slot, {})) for view in self.ordered()],
        }

    def ordered(self) -> list[InstanceView]:
        return [self.instances[slot] for slot in sorted(self.instances)]

    def snapshot(self, view: InstanceView, initialization: dict | None = None) -> dict[str, object]:
        session = view.session
        budget = session.max_overrun_ticks
        return {
            "slot": view.slot,
            "instance_id": f"slot-{view.slot}",
            "initialization": initialization or {},
            "running": view.running,
            "last_error": view.last_error,
            "current_tick": session.current_tick,
            "buffered_ticks": session.buffered_ticks,
            "underflow": session.underflow.value,
            "max_overrun_ticks": budget,
            "unlimited_overrun": budget is None,
            "overrun_exhausted": session.compiler.overrun_exhausted,
            "selected_hotbar": session.handle.selected_hotbar,
            "frame_revision": view.stream.revision,
            "stats": session.stats.as_dict(),
        }

    def submit(self, slot: int, sequence: str) -> dict[str, object]:
        """提交动作序列，并在后台线程里让编译器驱动环境跑到 WAIT。"""
        view = self.view(slot)
        if view.running:
            raise HTTPException(status_code=409, detail="该实例仍在执行上一段序列")
        try:
            submission = view.session.submit(sequence)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        view.last_error = None
        view.running = True
        Thread(target=self._drive, args=(view,), daemon=True).start()
        return {
            "start_tick": submission.start_tick,
            "accepted_ticks": submission.accepted_ticks,
            "expired_ticks": submission.expired_ticks,
            "overwritten_ticks": submission.overwritten_ticks,
        }

    def control(self, slot: int, request: ControlRequest) -> dict[str, object]:
        view = self.view(slot)
        if request.underflow is not None:
            try:
                view.session.underflow = UnderflowPolicy(request.underflow)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        if request.unlimited_overrun:
            view.session.max_overrun_ticks = None
        elif request.max_overrun_ticks is not None:
            try:
                view.session.max_overrun_ticks = request.max_overrun_ticks
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        return self.snapshot(view)

    def reset(self, slot: int, *, world: bool) -> dict[str, object]:
        view = self.view(slot)
        if view.running:
            raise HTTPException(status_code=409, detail="执行中不能重置")
        with view.lock:
            try:
                view.session.reset(world=world)
            except RuntimeError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            view.last_error = None
            view.stream.clear()
            self._publish(view)
        return self.snapshot(view)

    def frames(self, slot: int) -> dict[str, object]:
        """帧历史的索引，供界面构建回放进度条。"""
        view = self.view(slot)
        revision, count, latest_tick = view.stream.snapshot()
        return {
            "slot": slot,
            "revision": revision,
            "count": count,
            "latest_tick": latest_tick,
            "ticks": list(view.stream.ticks()),
        }

    def frame_bytes(self, slot: int, *, index: int | None, tick: int | None) -> bytes:
        """取回放帧；`tick` 优先，其次 `index`，都不给则取最新一帧。"""
        view = self.view(slot)
        if tick is not None:
            record = view.stream.at_tick(tick)
        elif index is not None:
            record = view.stream.at(index)
        else:
            record = view.stream.latest()
        if record is None:
            raise HTTPException(status_code=404, detail="没有对应的帧")
        return record.jpeg

    def _drive(self, view: InstanceView) -> None:
        try:
            with view.lock:
                for session_tick in view.session.run():
                    self._publish(view, tick=session_tick.tick)
        except Exception as error:
            # 后台线程里的任何失败都要落到界面上，而不是静默丢掉。
            view.last_error = f"{type(error).__name__}: {error}"
        finally:
            view.running = False

    def _publish(self, view: InstanceView, *, tick: int | None = None) -> None:
        try:
            frame = view.session.handle.observe()["rgb"]
        except RuntimeError:
            return
        view.stream.publish(frame, tick=tick)


def create_app(kernel: EnvironmentKernel) -> FastAPI:
    service = VisualizationService(kernel)
    app = FastAPI(title="CraftGround 实例控制台")
    app.state.service = service

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_page(service.describe())

    @app.get("/api/instances")
    def instances() -> dict[str, object]:
        return service.describe()

    @app.get("/api/instances/{slot}")
    def instance(slot: int) -> dict[str, object]:
        return service.snapshot(service.view(slot))

    @app.post("/api/instances/{slot}/submit")
    def submit(slot: int, request: SubmitRequest) -> dict[str, object]:
        return service.submit(slot, request.sequence)

    @app.post("/api/instances/{slot}/control")
    def control(slot: int, request: ControlRequest) -> dict[str, object]:
        return service.control(slot, request)

    @app.post("/api/instances/{slot}/reset")
    def reset(slot: int, request: ResetRequest) -> dict[str, object]:
        return service.reset(slot, world=request.world)

    @app.get("/api/instances/{slot}/stream")
    def stream(slot: int, fps: float = 20.0) -> StreamingResponse:
        view = service.view(slot)
        return StreamingResponse(
            view.stream.multipart(fps=fps),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        )

    @app.get("/api/instances/{slot}/frames")
    def frames(slot: int) -> dict[str, object]:
        return service.frames(slot)

    @app.get("/api/instances/{slot}/frame")
    def frame(slot: int, index: int | None = None, tick: int | None = None) -> Response:
        return Response(
            service.frame_bytes(slot, index=index, tick=tick),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app
