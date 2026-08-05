"""CraftGround 控制内核：唯一持有 JVM 句柄的装配、倒档与调度入口。"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from online_interactive_environments import ActionTick

from .action_adapter import CraftGroundKeyboardMouseAdapter
from .runtime import (
    ACTION_BACKEND,
    CRAFTGROUND_ACTION_SPACE,
    CRAFTGROUND_RUNTIME_VERSION,
    create_environment,
    prepare_runtime_instance,
    prepare_runtime_template,
)
from .snapshot_pool import EnvironmentPool
from .snapshots import MemorySnapshot, MemorySnapshotCoordinator, ResetTimings, SnapshotRegion

PayloadT = TypeVar("PayloadT")
OutputT = TypeVar("OutputT")
# `reset` 返回后 Minecraft 仍在加载地形，需要继续 tick 才会推进到可玩画面。
DEFAULT_WARMUP_TICKS = 30
# 倒档后让句柄缓存的观察跟上恢复后的世界所需的 tick 数。倒档的同步 tick 走裸环境，
# 句柄那一侧的 `latest_observation` 仍是倒档之前那一帧，只推一个 tick 会把传送与区块
# 重发途中的中间画面当成快照画面交给调用方。
DEFAULT_RESET_SETTLE_TICKS = 8


@dataclass(frozen=True)
class StepOutcome:
    """一个 tick 的环境侧事实；不包含任何相位信息。"""

    slot: int
    inputs: tuple[str, ...]
    native_action: dict[str, bool | float]
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    step_elapsed_ms: float
    info: Any


class EnvironmentHandle:
    """一个槽位的对外句柄；不暴露裸 CraftGround 对象。"""

    def __init__(
        self,
        kernel: EnvironmentKernel,
        slot: int,
        environment: Any,
        adapter: CraftGroundKeyboardMouseAdapter,
    ) -> None:
        self._kernel = kernel
        self._slot = slot
        self._environment = environment
        self._adapter = adapter
        self.latest_observation: Any = None
        self.latest_info: Any = None

    @property
    def slot(self) -> int:
        return self._slot

    @property
    def selected_hotbar(self) -> int:
        return self._adapter.selected_hotbar

    def apply(self, tick: ActionTick) -> StepOutcome:
        """转译并执行一个标准输入 tick；设备边界不再让渡给调用方。"""
        native_action = self._adapter.convert(tick)
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = self._environment.step(native_action)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.latest_observation = observation
        self.latest_info = info
        return StepOutcome(
            slot=self._slot,
            inputs=tick.inputs,
            native_action=native_action,
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            step_elapsed_ms=elapsed_ms,
            info=info,
        )

    def observe(self) -> Any:
        """返回最近一次观察；CraftGround 没有独立观察通道。"""
        if self.latest_observation is None:
            raise RuntimeError("环境尚未产生观察，需要先 reset 或 apply")
        return self.latest_observation

    def preview_adapter(self) -> CraftGroundKeyboardMouseAdapter:
        """克隆当前设备状态，供调用方在不触碰环境的前提下预演转译。"""
        return CraftGroundKeyboardMouseAdapter(
            selected_hotbar=self._adapter.selected_hotbar,
            action_factory=self._adapter.action_factory,
        )

    def reset_to(self, snapshot: MemorySnapshot | str | None = None) -> float:
        """把本槽位恢复到指定内存快照，或恢复到内核的根快照。"""
        return self._kernel._reset_slot(self._slot, snapshot)

    def reset_world(
        self,
        *,
        fast_reset: bool = False,
        warmup_ticks: int = DEFAULT_WARMUP_TICKS,
    ) -> Any:
        """重新 reset 本槽位的世界；比内存倒档昂贵。

        `reset` 返回时 Minecraft 仍停在 `Loading terrain...`，地形要靠继续 tick 才会
        加载完成。因此这里默认再空跑 `warmup_ticks` 个 NoOp，使调用方拿到的第一帧是
        真实世界画面而不是加载界面。
        """
        observation, info = self._environment.reset(options={"fast_reset": fast_reset})
        self._adapter.reset()
        self.latest_observation = observation
        self.latest_info = info
        return self.warmup(warmup_ticks)

    def warmup(self, ticks: int = DEFAULT_WARMUP_TICKS) -> Any:
        """空跑若干 NoOp tick，让世界加载完成；不改变设备状态。"""
        if ticks < 0:
            raise ValueError("warmup_ticks 不能为负数")
        for _ in range(ticks):
            self.apply(ActionTick())
        return self.latest_observation


@dataclass(frozen=True)
class RolloutRequest(Generic[PayloadT, OutputT]):
    """一次 SubAgent 推演请求；`simulate` 只能看到 `EnvironmentHandle`。"""

    request_id: str
    subagent_id: str
    payload: PayloadT
    simulate: Callable[[EnvironmentHandle, PayloadT], OutputT]
    snapshot: MemorySnapshot | str | None = None


@dataclass(frozen=True)
class RolloutResult(Generic[OutputT]):
    request_id: str
    subagent_id: str
    slot: int
    waited_ms: float
    restore_ms: float
    rollout_ms: float
    output: OutputT


@dataclass
class _SlotRecord:
    slot: int
    environment: Any
    runtime_path: Path
    port: int
    baseline_world: dict[str, str] | None
    adapter: CraftGroundKeyboardMouseAdapter
    handle: EnvironmentHandle | None = field(default=None, repr=False)


class EnvironmentKernel(AbstractContextManager["EnvironmentKernel"]):
    """持有全部 CraftGround JVM 句柄；对外只暴露操控、重置与换基准三类调用。"""

    def __init__(
        self,
        slots: tuple[_SlotRecord, ...],
        *,
        template: Path,
        instance_prefix: str,
        launch_options: dict[str, Any],
    ) -> None:
        if not slots:
            raise ValueError("内核至少需要一个环境槽位")
        self._slots = slots
        self._template = template
        self._instance_prefix = instance_prefix
        self._launch_options = dict(launch_options)
        self._closed = False
        self._root_snapshot: MemorySnapshot | None = None
        self._coordinator = MemorySnapshotCoordinator(tuple(record.environment for record in slots))
        for record in slots:
            record.handle = EnvironmentHandle(self, record.slot, record.environment, record.adapter)
        self._pool: EnvironmentPool[EnvironmentHandle] = EnvironmentPool(
            tuple(record.handle for record in slots if record.handle is not None)
        )

    @classmethod
    def launch(
        cls,
        *,
        slots: int = 1,
        baseline_world: Path | str | None = None,
        baseline_world_display_name: str = "New World",
        port_base: int = 18300,
        instance_prefix: str | None = None,
        runtime_template_target: Path | None = None,
        runtime_instances_root: Path | None = None,
        template: Path | None = None,
        reset_on_launch: bool = True,
        warmup_ticks: int = DEFAULT_WARMUP_TICKS,
        environment_factory: Callable[..., Any] = create_environment,
        **environment_options: Any,
    ) -> EnvironmentKernel:
        """准备模板、逐槽位创建实例与 JVM；任一步失败时回滚已创建的环境。"""
        if slots < 1:
            raise ValueError("slots 必须大于零")
        resolved_template = (
            prepare_runtime_template(runtime_template_target) if template is None else template
        )
        prefix = instance_prefix or f"kernel-{uuid.uuid4().hex[:8]}"
        # 保留全部可重放参数，使 rebase 能用同一装配方式重建槽位。
        launch_options = {
            "slots": slots,
            "port_base": port_base,
            "baseline_world_display_name": baseline_world_display_name,
            "runtime_instances_root": runtime_instances_root,
            "reset_on_launch": reset_on_launch,
            "warmup_ticks": warmup_ticks,
            "environment_factory": environment_factory,
            **environment_options,
        }
        records: list[_SlotRecord] = []
        try:
            for slot in range(slots):
                runtime_path = prepare_runtime_instance(
                    f"{prefix}-{slot}",
                    template=resolved_template,
                    instances_root=runtime_instances_root,
                )
                environment = environment_factory(
                    runtime_path=runtime_path,
                    port=port_base + slot,
                    baseline_world_path=baseline_world,
                    baseline_world_display_name=baseline_world_display_name,
                    **environment_options,
                )
                records.append(
                    _SlotRecord(
                        slot=slot,
                        environment=environment,
                        runtime_path=Path(runtime_path),
                        port=port_base + slot,
                        baseline_world=getattr(environment, "tao_baseline_world", None),
                        adapter=CraftGroundKeyboardMouseAdapter(),
                    )
                )
        except BaseException:
            for record in records:
                _close_quietly(record.environment)
            raise
        kernel = cls(
            tuple(records),
            template=Path(resolved_template),
            instance_prefix=prefix,
            launch_options=launch_options,
        )
        if reset_on_launch:
            try:
                kernel.reset_world(warmup_ticks=warmup_ticks)
            except BaseException:
                kernel.close()
                raise
        return kernel

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def root_snapshot(self) -> MemorySnapshot | None:
        return self._root_snapshot

    def close(self) -> None:
        """关闭全部 JVM；可重复调用。"""
        if self._closed:
            return
        self._closed = True
        for record in self._slots:
            _close_quietly(record.environment)

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def lease(self, *, timeout: float | None = None) -> AbstractContextManager[EnvironmentHandle]:
        """取得一个槽位的独占句柄；无空位时等待。"""
        self._require_open()
        return self._pool.acquire(timeout)

    def raw_environment(self, slot: int = 0) -> Any:
        """仅供 IPC 与渲染验收使用的逃生口，返回裸 CraftGround 对象。

        正常控制路径应当使用 `lease()` 或 `handles()`。验收命令需要读取 `ipc`、
        `observation_converter` 和 `process` 这些内部属性，它们不属于内核对外契约。
        """
        self._require_open()
        return self._slots[slot].environment

    def handles(self) -> tuple[EnvironmentHandle, ...]:
        """按槽位顺序返回全部句柄；仅供不需要互斥的全量操作使用。"""
        self._require_open()
        return tuple(record.handle for record in self._slots if record.handle is not None)

    def capture(
        self,
        snapshot_id: str,
        *,
        region: SnapshotRegion | None = None,
        horizontal_radius: int = 24,
        as_root: bool = False,
    ) -> MemorySnapshot:
        """让全部 JVM 以同一 ID 保存各自当前状态。

        `region` 省略时按 0 号槽位的实际玩家坐标计算快照区域；可比较推演要求各槽位在
        保存同名快照前处于同一逻辑状态，因此共用一个区域而不是逐槽位各算一份。
        """
        self._require_open()
        resolved = (
            region
            if region is not None
            else self.player_region(horizontal_radius=horizontal_radius)
        )
        snapshot = self._coordinator.capture_all(snapshot_id, resolved)
        if as_root:
            self._root_snapshot = snapshot
        return snapshot

    def player_region(self, *, slot: int = 0, horizontal_radius: int = 24) -> SnapshotRegion:
        """按某个槽位的当前玩家坐标计算快照区域。"""
        self._require_open()
        handle = self._slots[slot].handle
        if handle is None:
            raise RuntimeError(f"槽位 {slot} 没有句柄")
        state = handle.observe()["full"]
        return SnapshotRegion.around_player(
            (state.x, state.y, state.z),
            horizontal_radius=horizontal_radius,
        )

    def reset(self, snapshot: MemorySnapshot | str | None = None) -> ResetTimings:
        """并行倒档全部槽位；无参表示回到根快照。"""
        self._require_open()
        target = self._resolve_snapshot(snapshot)
        timings = self._coordinator.reset_all(target)
        for record in self._slots:
            record.adapter.reset()
            # 同上：让句柄缓存的观察跟上倒档后的世界。
            if record.handle is not None:
                record.handle.warmup(DEFAULT_RESET_SETTLE_TICKS)
        return timings

    def reset_world(
        self,
        *,
        fast_reset: bool = False,
        warmup_ticks: int = DEFAULT_WARMUP_TICKS,
    ) -> None:
        """重新 reset 全部槽位的世界；用于启动与换基准后的冷启动。"""
        self._require_open()
        for record in self._slots:
            if record.handle is not None:
                record.handle.reset_world(fast_reset=fast_reset, warmup_ticks=warmup_ticks)

    def rebase(
        self,
        baseline_world: Path | str,
        *,
        display_name: str | None = None,
    ) -> EnvironmentKernel:
        """换环境基准；目录级基准无法原地修改，因此重建全部槽位。

        返回一个新的内核。替代实例复用同一批端口，所以当前内核先关闭再重建；重建失败时
        两个内核都不可用，调用方需要重新 `launch`。便宜的换基准路径是
        `capture(..., as_root=True)`，它不重启 JVM，但只覆盖玩家周边快照区域。
        """
        self._require_open()
        options = dict(self._launch_options)
        slots = options.pop("slots")
        port_base = options.pop("port_base")
        if display_name is not None:
            options["baseline_world_display_name"] = display_name
        # 替代实例复用同一批端口，因此必须先释放旧 JVM；旧实例目录仍持有上一份基准存档，
        # 所以换基准同时要换实例前缀。
        prefix = f"{self._instance_prefix}-rebased-{uuid.uuid4().hex[:6]}"
        self.close()
        return type(self).launch(
            slots=slots,
            baseline_world=baseline_world,
            port_base=port_base,
            template=self._template,
            instance_prefix=prefix,
            **options,
        )

    def rollout(
        self,
        requests: Iterable[RolloutRequest[PayloadT, OutputT]],
        *,
        wait_timeout: float | None = None,
        max_workers: int | None = None,
    ) -> tuple[RolloutResult[OutputT], ...]:
        """并行执行推演请求并按输入顺序返回；超额请求在池外等待。"""
        from concurrent.futures import ThreadPoolExecutor

        self._require_open()
        request_list = tuple(requests)
        if not request_list:
            return ()
        # 并发上限由环境池决定，不由线程池决定；默认让每个请求都进入池等待，
        # 这样 waited_ms 反映真实的槽位竞争而不是线程排队。
        workers = max_workers or len(request_list)
        if workers < 1:
            raise ValueError("max_workers 必须大于零")
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="craftground-rollout",
        ) as executor:
            return tuple(
                executor.map(
                    lambda request: self._run_one(request, wait_timeout),
                    request_list,
                )
            )

    def describe(self) -> dict[str, Any]:
        """导出内核装配事实，供轨迹导出与验收报告引用。"""
        return {
            "action_backend": ACTION_BACKEND,
            "action_space": CRAFTGROUND_ACTION_SPACE,
            "runtime_version": CRAFTGROUND_RUNTIME_VERSION,
            "template_path": str(self._template),
            "closed": self._closed,
            "root_snapshot": (
                None if self._root_snapshot is None else self._root_snapshot.snapshot_id
            ),
            "slots": [
                {
                    "slot": record.slot,
                    "port": record.port,
                    "runtime_path": str(record.runtime_path),
                    "baseline_world": record.baseline_world,
                }
                for record in self._slots
            ],
        }

    @contextmanager
    def _leased(self, timeout: float | None) -> Iterator[tuple[EnvironmentHandle, float]]:
        lease = self._pool.acquire(timeout)
        with lease as handle:
            yield handle, lease.waited_ms

    def _run_one(
        self,
        request: RolloutRequest[PayloadT, OutputT],
        wait_timeout: float | None,
    ) -> RolloutResult[OutputT]:
        with self._leased(wait_timeout) as (handle, waited_ms):
            restore_ms = 0.0
            if request.snapshot is not None or self._root_snapshot is not None:
                restore_ms = handle.reset_to(request.snapshot)
            started = time.perf_counter()
            output = request.simulate(handle, request.payload)
            rollout_ms = (time.perf_counter() - started) * 1000.0
            return RolloutResult(
                request.request_id,
                request.subagent_id,
                handle.slot,
                waited_ms,
                restore_ms,
                rollout_ms,
                output,
            )

    def _reset_slot(self, slot: int, snapshot: MemorySnapshot | str | None) -> float:
        self._require_open()
        target = self._resolve_snapshot(snapshot)
        record = self._slots[slot]
        elapsed = self._coordinator.reset_one(record.environment, target)
        record.adapter.reset()
        # 倒档的同步 tick 走的是裸环境，句柄缓存的观察还是倒档之前那一帧。
        if record.handle is not None:
            record.handle.warmup(DEFAULT_RESET_SETTLE_TICKS)
        return elapsed

    def _resolve_snapshot(self, snapshot: MemorySnapshot | str | None) -> MemorySnapshot | str:
        if snapshot is not None:
            return snapshot
        if self._root_snapshot is None:
            raise RuntimeError("内核没有根快照，请先调用 capture(..., as_root=True)")
        return self._root_snapshot

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("内核已关闭")


def _close_quietly(environment: Any) -> None:
    close = getattr(environment, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        # 回滚与批量关闭路径不能因单个 JVM 关闭失败而中断其余槽位。
        pass
