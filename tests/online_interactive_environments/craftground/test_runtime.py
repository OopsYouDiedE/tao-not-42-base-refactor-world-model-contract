from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from online_interactive_environments.craftground import runtime


class FakeDistribution:
    version = "2.7.4"

    def __init__(self, root: Path) -> None:
        self.root = root

    def locate_file(self, path: str) -> Path:
        return self.root / path


def _source_runtime(tmp_path: Path) -> Path:
    source = tmp_path / "site-packages" / "craftground_runtime_mc121"
    package = source / "src/main/java/com/kyhsgeekcode/minecraftenv"
    package.mkdir(parents=True)
    (package / "MinecraftEnv.kt").write_text(
        "class MinecraftEnv {\n"
        "    private fun handleCommand(\n"
        "        command: String,\n"
        "    ): Boolean {\n"
        "        return false\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    cpp = source / "src/main/cpp"
    cpp.mkdir(parents=True)
    (cpp / "noboost_ipc.cpp").write_text(
        "void write_observation() {\n"
        + runtime.OBSERVATION_MMAP_ORIGINAL
        + "    std::memcpy(data_start, data, observation_size);\n"
        + runtime.OBSERVATION_UNMAP_ORIGINAL
        + "}\n",
        encoding="utf-8",
    )
    (source / ("gradlew.bat" if sys.platform == "win32" else "gradlew")).write_text(
        "wrapper",
        encoding="ascii",
    )
    return source


def test_prepare_patched_runtime_builds_once(tmp_path: Path, monkeypatch) -> None:
    source = _source_runtime(tmp_path)
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(source.parent),
    )
    builds: list[object] = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: builds.append(args))
    target = tmp_path / "cache" / "runtime"

    first = runtime.prepare_patched_runtime(target)
    second = runtime.prepare_patched_runtime(target)

    package = target / "src/main/java/com/kyhsgeekcode/minecraftenv"
    content = (package / "MinecraftEnv.kt").read_text(encoding="utf-8")
    observation_write = (target / "src/main/cpp/noboost_ipc.cpp").read_text(encoding="utf-8")
    assert first == second == target.resolve()
    assert content.count(runtime.DISPATCH) == 1
    assert (package / "MemorySnapshotStore.kt").is_file()
    assert (target / ".tao-memorysnapshot-build").is_file()
    assert len(builds) == 1
    # 写观察前必须先把 j2p 扩容到整帧大小，否则 JVM memcpy 越界并 SIGSEGV。
    assert observation_write.count(runtime.OBSERVATION_MMAP_PATCHED) == 1
    assert runtime.OBSERVATION_UNMAP_ORIGINAL not in observation_write
    assert "ftruncate(j2pFd" in observation_write


def test_explicit_runtime_path_skips_automatic_install(tmp_path: Path, monkeypatch) -> None:
    explicit_runtime = tmp_path / "ready-runtime"
    explicit_runtime.mkdir()
    monkeypatch.setattr(
        runtime,
        "prepare_patched_runtime",
        lambda target=None: (_ for _ in ()).throw(AssertionError("不应自动安装")),
    )

    closed: list[str] = []

    class FakeEnvironment(dict):
        def __init__(self, config, **kwargs) -> None:
            super().__init__(kwargs)
            self.ipc = SimpleNamespace(release=lambda: closed.append("released"))

        def close(self) -> None:
            closed.append("closed")

    craftground = ModuleType("craftground")
    craftground.CraftGroundEnvironment = FakeEnvironment
    craftground.InitialEnvironmentConfig = lambda **kwargs: kwargs
    action_space = ModuleType("craftground.environment.action_space")
    action_space.ActionSpaceVersion = SimpleNamespace(V2_MINERL_HUMAN="v2")
    encoding = ModuleType("craftground.screen_encoding_modes")
    encoding.ScreenEncodingMode = SimpleNamespace(RAW="raw")
    monkeypatch.setitem(sys.modules, "craftground", craftground)
    monkeypatch.setitem(sys.modules, "craftground.environment.action_space", action_space)
    monkeypatch.setitem(sys.modules, "craftground.screen_encoding_modes", encoding)
    monkeypatch.setattr(runtime, "enable_shared_memory_reuse", lambda: None)

    environment = runtime.create_environment(explicit_runtime)
    environment.close()

    assert environment["env_path"] == str(explicit_runtime.resolve())
    assert environment["action_space_version"] == "v2"
    assert environment["find_free_port"] is False
    assert environment["use_shared_memory"] is True
    # 共享内存段必须在环境关闭之后才回收，否则会 unlink 掉仍在使用的段。
    assert closed == ["closed", "released"]


class FakeInitialEnvironment:
    def SerializeToString(self) -> bytes:
        return b"initial-environment"


def _install_fake_shared_memory(monkeypatch) -> tuple[list[tuple[int, int]], list[str]]:
    initialized: list[tuple[int, int]] = []
    destroyed: list[str] = []

    class FakeBoostIPC:
        def destroy(self) -> None:
            destroyed.append(self.p2j_shared_memory_name)

    def initialize_shared_memory(port, data, data_length, action_length, find_free_port):
        initialized.append((int(port), int(action_length)))
        return int(port)

    boost = ModuleType("craftground.environment.boost_ipc")
    boost.BoostIPC = FakeBoostIPC
    native = ModuleType("craftground.craftground_native")
    native.initialize_shared_memory = initialize_shared_memory
    monkeypatch.setitem(sys.modules, "craftground.environment.boost_ipc", boost)
    monkeypatch.setitem(sys.modules, "craftground.craftground_native", native)
    monkeypatch.setattr(runtime, "_SHARED_MEMORY_PATCHED", False)
    monkeypatch.setattr(runtime, "action_segment_capacity", lambda: 512)
    runtime.enable_shared_memory_reuse()
    return initialized, destroyed, FakeBoostIPC


def test_observation_write_patch_refuses_unexpected_source(tmp_path: Path) -> None:
    target = tmp_path / "runtime"
    cpp = target / "src/main/cpp"
    cpp.mkdir(parents=True)
    (cpp / "noboost_ipc.cpp").write_text("void write_observation() {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="j2p 映射代码与预期不符"):
        runtime._patch_observation_write(target)


def test_shared_memory_reuse_initializes_each_port_once(monkeypatch) -> None:
    initialized, destroyed, FakeBoostIPC = _install_fake_shared_memory(monkeypatch)
    environment = FakeInitialEnvironment()

    first = FakeBoostIPC(18300, False, environment, None)
    # `reset()` 内的 ensure_alive 对同一端口重复构造，必须复用而不是再次初始化。
    second = FakeBoostIPC(18300, False, environment, None)
    other_port = FakeBoostIPC(18301, False, environment, None)

    assert initialized == [(18300, 512), (18301, 512)]
    assert second.p2j_shared_memory_name == first.p2j_shared_memory_name
    prefix = "Global\\" if sys.platform == "win32" else "/"
    assert other_port.p2j_shared_memory_name == f"{prefix}craftground_18301_p2j"
    # 被丢弃的旧实例不得 unlink 仍在使用的段。
    second.destroy()
    assert destroyed == []
    second.release()
    assert destroyed == [f"{prefix}craftground_18300_p2j"]
    # 回收之后同端口可以重新初始化。
    FakeBoostIPC(18300, False, environment, None)
    assert initialized == [(18300, 512), (18301, 512), (18300, 512)]


def test_action_segment_capacity_exceeds_real_action_messages() -> None:
    pytest.importorskip("craftground.environment.action_space")
    from craftground.environment.action_space import action_v2_dict_to_message, no_op_v2

    capacity = runtime.action_segment_capacity()
    # 上游用 no_op 长度定容，而该消息序列化为 0 字节；容量必须覆盖真实动作。
    assert len(action_v2_dict_to_message(no_op_v2()).SerializeToString()) == 0
    action = no_op_v2()
    action.update(forward=True, jump=True, attack=True, camera_yaw=24.0, camera_pitch=-12.0)
    message = action_v2_dict_to_message(action)
    message.commands.extend(["memorysnapshot save teacher-log-shared-start -29 -64 -26 19 319 22"])
    assert capacity > len(message.SerializeToString())


def test_runtime_instances_have_independent_writable_directories(tmp_path: Path) -> None:
    template = tmp_path / "template"
    (template / "run").mkdir(parents=True)
    (template / "CMakeFiles").mkdir()
    (template / "CMakeCache.txt").write_text("old-path", encoding="utf-8")
    (template / "_deps" / "glm-build").mkdir(parents=True)
    (template / "_deps" / "glm-subbuild").mkdir()
    (template / ".tao-memorysnapshot-build").write_text("digest\n", encoding="ascii")
    (template / "run" / "options.txt").write_text("template", encoding="utf-8")
    (template / "run" / "saves" / "New World (20)").mkdir(parents=True)
    (template / "run" / "logs").mkdir()
    (template / "run" / "logs" / "latest.log").write_text("stale", encoding="utf-8")

    first = runtime.prepare_runtime_instance(
        "slot-0", template=template, instances_root=tmp_path / "instances"
    )
    second = runtime.prepare_runtime_instance(
        "slot-1", template=template, instances_root=tmp_path / "instances"
    )
    (first / "run" / "options.txt").write_text("first", encoding="utf-8")

    assert first != second
    assert (second / "run" / "options.txt").read_text(encoding="utf-8") == "template"
    assert (template / "run" / "options.txt").read_text(encoding="utf-8") == "template"
    assert not (first / "CMakeCache.txt").exists()
    assert not (first / "CMakeFiles").exists()
    assert not (first / "_deps" / "glm-build").exists()
    assert not (first / "_deps" / "glm-subbuild").exists()
    assert not (first / "run" / "saves").exists()
    assert not (first / "run" / "logs").exists()
    assert (template / "run" / "saves" / "New World (20)").is_dir()


def test_baseline_world_is_copied_to_independent_instance_directories(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    (baseline / "region").mkdir(parents=True)
    (baseline / "level.dat").write_bytes(b"fixed-level")
    (baseline / "region" / "r.0.0.mca").write_bytes(b"fixed-region")
    (baseline / "session.lock").write_bytes(b"stale-lock")
    runtime_zero = tmp_path / "runtime-zero"
    runtime_one = tmp_path / "runtime-one"
    runtime_zero.mkdir()
    runtime_one.mkdir()

    first = runtime.install_baseline_world(baseline, runtime_zero)
    second = runtime.install_baseline_world(baseline, runtime_one)

    first_world = Path(first["instance_world_path"])
    second_world = Path(second["instance_world_path"])
    assert first["source_sha256"] == second["source_sha256"]
    assert first_world != second_world
    assert (first_world / "level.dat").read_bytes() == b"fixed-level"
    assert (second_world / "region" / "r.0.0.mca").read_bytes() == b"fixed-region"
    assert not (first_world / "session.lock").exists()
    assert not (second_world / "session.lock").exists()
