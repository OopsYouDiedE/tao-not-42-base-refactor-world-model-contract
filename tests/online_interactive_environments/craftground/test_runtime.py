from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from online_interactive_environments.craftground import runtime

MAINTAINED_COMMIT = "ac71d4ef6fb12b994d35b36f8eec518aa3a307e7"
MAINTAINED_REPOSITORY = "https://github.com/OopsYouDiedE/CraftGround.git"


def _write_maintained_runtime(root: Path) -> None:
    java = root / "src/main/java/com/kyhsgeekcode/minecraftenv"
    java.mkdir(parents=True)
    (java / "MemorySnapshotStore.kt").write_text("object MemorySnapshotStore\n", encoding="utf-8")
    (java / "MinecraftEnv.kt").write_text(
        "MemorySnapshotStore.handle(command, client)\n", encoding="utf-8"
    )
    cpp = root / "src/main/cpp"
    (cpp / "include").mkdir(parents=True)
    (cpp / "noboost_ipc.cpp").write_text(
        "const size_t j2pRequiredSize = observation_size;\nftruncate(j2pFd, j2pRequiredSize);\n",
        encoding="utf-8",
    )
    (cpp / "CMakeLists.txt").write_text(
        'set(GL_CAPTURE_DIR "${CMAKE_CURRENT_LIST_DIR}")\n', encoding="utf-8"
    )
    (cpp / "include/framebuffer_capturer.h").write_text("#pragma once\n", encoding="utf-8")


def test_craftground_dependencies_pin_one_fork_commit() -> None:
    root = Path(__file__).resolve().parents[3]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = configuration["project"]["optional-dependencies"]["craftground"]
    pattern = re.compile(
        rf"^(craftground(?:-runtime-mc121)?) @ git\+{re.escape(MAINTAINED_REPOSITORY)}"
        rf"@([0-9a-f]{{40}})(#subdirectory=minecraft/mc121)?$"
    )
    matches = [pattern.fullmatch(dependency) for dependency in dependencies]

    assert all(match is not None for match in matches)
    parsed = [match for match in matches if match is not None]
    assert {match.group(1) for match in parsed} == {
        "craftground",
        "craftground-runtime-mc121",
    }
    assert {match.group(2) for match in parsed} == {MAINTAINED_COMMIT}
    runtime_match = next(match for match in parsed if match.group(1) == "craftground-runtime-mc121")
    assert runtime_match.group(3) == "#subdirectory=minecraft/mc121"


def test_environment_rejects_unsupported_screen_encoding_before_preparation() -> None:
    with pytest.raises(ValueError, match="screen_encoding_mode"):
        runtime.create_environment(screen_encoding_mode="png")  # type: ignore[arg-type]


def test_maintained_runtime_contract_accepts_fork_layout(tmp_path: Path) -> None:
    _write_maintained_runtime(tmp_path)

    runtime.validate_maintained_runtime(tmp_path)


def test_maintained_runtime_contract_rejects_missing_capability(tmp_path: Path) -> None:
    _write_maintained_runtime(tmp_path)
    (tmp_path / "src/main/cpp/noboost_ipc.cpp").write_text(
        "void write_observation() {}\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="观察共享内存按帧扩容"):
        runtime.validate_maintained_runtime(tmp_path)


def test_runtime_instances_have_independent_writable_directories(tmp_path: Path) -> None:
    template = tmp_path / "template"
    (template / "run").mkdir(parents=True)
    (template / "CMakeFiles").mkdir()
    (template / "CMakeCache.txt").write_text("old-path", encoding="utf-8")
    (template / "_deps" / "glm-build").mkdir(parents=True)
    (template / "_deps" / "glm-subbuild").mkdir()
    (template / ".tao-runtime-build").write_text("digest\n", encoding="ascii")
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
