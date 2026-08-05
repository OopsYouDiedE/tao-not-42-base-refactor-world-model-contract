"""CraftGround 维护版 runtime 准备与环境创建入口。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Literal

_PREPARE_LOCK = Lock()
ACTION_BACKEND = "keyboard_and_mouse_only"
CRAFTGROUND_ACTION_SPACE = "V2_MINERL_HUMAN"
CRAFTGROUND_RUNTIME_VERSION = "0.1.0+tao.2"
ScreenEncodingModeName = Literal["raw", "zerocopy_torch"]
SUPPORTED_SCREEN_ENCODING_MODES = ("raw", "zerocopy_torch")
_INSTANCE_MARKER = ".tao-runtime-instance"
_RUNTIME_BUILD_MARKER = ".tao-runtime-build"
# 实例目录只需要模板的构建产物；这些运行时目录由 JVM 重新生成。
_INSTANCE_PRUNED_PATHS = (
    "CMakeCache.txt",
    "CMakeFiles",
    "_deps/glm-build",
    "_deps/glm-subbuild",
    "run/saves",
    "run/logs",
    "run/crash-reports",
)


def directory_sha256(path: Path | str) -> str:
    """计算目录内容的稳定 SHA-256，忽略 Minecraft 运行锁。"""
    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    for file_path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = file_path.relative_to(root)
        if relative.as_posix() == "session.lock":
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def install_baseline_world(
    source: Path | str,
    runtime: Path | str,
    *,
    world_directory_name: str = "New World",
) -> dict[str, str]:
    """把基准世界复制到单个 CraftGround runtime 的独立存档目录。"""
    source_path = Path(source).expanduser().resolve()
    runtime_path = Path(runtime).expanduser().resolve()
    if not (source_path / "level.dat").is_file():
        raise FileNotFoundError(f"基准存档缺少 level.dat: {source_path}")
    if not re.fullmatch(r"[A-Za-z0-9_. -]+", world_directory_name):
        raise ValueError("world_directory_name 包含不安全字符")
    destination = runtime_path / "run" / "saves" / world_directory_name
    if destination.exists():
        raise FileExistsError(f"实例存档目录已存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_path,
        destination,
        ignore=shutil.ignore_patterns("session.lock"),
    )
    return {
        "source_path": str(source_path),
        "source_sha256": directory_sha256(source_path),
        "instance_world_path": str(destination.resolve()),
        "world_directory_name": world_directory_name,
    }


def validate_maintained_runtime(runtime_path: Path | str) -> None:
    """校验 runtime 包含自维护分支承诺的源码能力。

    Args:
        runtime_path: `craftground_runtime_mc121` 包目录或它的副本。

    Raises:
        FileNotFoundError: runtime 缺少维护版要求的文件。
        RuntimeError: 文件存在，但关键实现并非维护版合同。
    """
    root = Path(runtime_path).expanduser().resolve()
    required_files = (
        "src/main/java/com/kyhsgeekcode/minecraftenv/MemorySnapshotStore.kt",
        "src/main/java/com/kyhsgeekcode/minecraftenv/MinecraftEnv.kt",
        "src/main/cpp/noboost_ipc.cpp",
        "src/main/cpp/CMakeLists.txt",
        "src/main/cpp/include/framebuffer_capturer.h",
    )
    missing = [relative for relative in required_files if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"CraftGround 维护版 runtime 缺少文件：{missing}")

    minecraft_env = (
        root / "src/main/java/com/kyhsgeekcode/minecraftenv/MinecraftEnv.kt"
    ).read_text(encoding="utf-8")
    native_ipc = (root / "src/main/cpp/noboost_ipc.cpp").read_text(encoding="utf-8")
    cmake = (root / "src/main/cpp/CMakeLists.txt").read_text(encoding="utf-8")
    contracts = {
        "内存快照命令分发": "MemorySnapshotStore.handle(command, client)" in minecraft_env,
        "观察共享内存按帧扩容": (
            "j2pRequiredSize" in native_ipc and "ftruncate(j2pFd" in native_ipc
        ),
        "runtime native 源码自包含": "CMAKE_CURRENT_LIST_DIR" in cmake,
    }
    failed = [name for name, satisfied in contracts.items() if not satisfied]
    if failed:
        raise RuntimeError(f"CraftGround runtime 不满足维护版合同：{failed}")


def prepare_runtime_template(target: Path | None = None, *, build: bool = True) -> Path:
    """复制并构建已安装的 CraftGround 维护版 runtime，不修改其中源码。"""
    distribution = importlib.metadata.distribution("craftground-runtime-mc121")
    if distribution.version != CRAFTGROUND_RUNTIME_VERSION:
        raise RuntimeError(
            "craftground-runtime-mc121 版本不符："
            f"期望 {CRAFTGROUND_RUNTIME_VERSION}，实际 {distribution.version}"
        )
    source = Path(distribution.locate_file("craftground_runtime_mc121")).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"找不到 CraftGround runtime 源目录：{source}")
    validate_maintained_runtime(source)
    source_digest = directory_sha256(source)

    if target is None:
        safe_version = re.sub(r"[^A-Za-z0-9_.-]", "-", distribution.version)
        target = (
            Path.home()
            / ".cache"
            / "tao"
            / f"craftground-runtime-mc121-{safe_version}-{source_digest[:12]}"
        )
    target = target.expanduser().resolve()
    build_identity = f"{distribution.version}\n{source_digest}\n"

    with _PREPARE_LOCK:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        validate_maintained_runtime(target)
        build_marker = target / _RUNTIME_BUILD_MARKER
        already_built = (
            build_marker.is_file() and build_marker.read_text(encoding="ascii") == build_identity
        )
        if build and not already_built:
            gradle = target / ("gradlew.bat" if sys.platform == "win32" else "gradlew")
            if not gradle.is_file():
                raise FileNotFoundError(f"runtime 缺少 Gradle wrapper：{gradle}")
            subprocess.run(
                [str(gradle), "build", "--no-daemon"],
                cwd=target,
                check=True,
            )
            build_marker.write_text(build_identity, encoding="ascii")
    return target


def prepare_runtime_instance(
    instance_id: str,
    *,
    template: Path | None = None,
    instances_root: Path | None = None,
) -> Path:
    """从只读构建模板创建一个可独立运行的 CraftGround 工作目录。"""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "-", instance_id).strip(".-")
    if not safe_id:
        raise ValueError("instance_id 必须包含字母、数字或安全分隔符")
    resolved_template = (
        prepare_runtime_template() if template is None else template.expanduser().resolve()
    )
    if not resolved_template.is_dir():
        raise FileNotFoundError(f"CraftGround runtime 模板不存在：{resolved_template}")
    root = (
        Path.home() / ".cache" / "tao" / "craftground-runtime-instances"
        if instances_root is None
        else instances_root.expanduser().resolve()
    )
    target = root / safe_id
    template_marker = resolved_template / _RUNTIME_BUILD_MARKER
    template_identity = hashlib.sha256(
        (str(resolved_template) + "\n").encode("utf-8")
        + (template_marker.read_bytes() if template_marker.is_file() else b"unbuilt")
    ).hexdigest()

    with _PREPARE_LOCK:
        marker = target / _INSTANCE_MARKER
        current_identity = marker.read_text(encoding="ascii").strip() if marker.is_file() else None
        if current_identity != template_identity:
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copytree(resolved_template, temporary)
                for relative in _INSTANCE_PRUNED_PATHS:
                    cache_path = temporary / relative
                    if cache_path.is_dir():
                        shutil.rmtree(cache_path)
                    elif cache_path.exists():
                        cache_path.unlink()
                (temporary / _INSTANCE_MARKER).write_text(
                    template_identity + "\n", encoding="ascii"
                )
                temporary.replace(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    return target.resolve()


def create_environment(
    runtime_path: Path | str | None = None,
    *,
    image_width: int = 640,
    image_height: int = 360,
    seed: str = "424242",
    render_distance: int = 3,
    simulation_distance: int = 5,
    port: int = 18300,
    find_free_port: bool = False,
    use_shared_memory: bool = True,
    auto_prepare_runtime: bool = True,
    runtime_template_target: Path | None = None,
    instance_id: str | None = None,
    runtime_instances_root: Path | None = None,
    baseline_world_path: Path | str | None = None,
    baseline_world_display_name: str = "New World",
    cleanup_world: bool = True,
    verbose: bool = False,
    verbose_gradle: bool = False,
    verbose_jvm: bool = False,
    screen_encoding_mode: ScreenEncodingModeName = "raw",
) -> Any:
    """创建 CraftGround 键鼠后端环境；默认准备维护版 runtime 的独立副本。

    默认使用共享内存 IPC，观察不经 socket 序列化。维护版 CraftGround 负责共享内存
    的容量、重建和幂等销毁；该路径也不触发 SocketIPC 的全局 java 进程扫描。
    """
    if screen_encoding_mode not in SUPPORTED_SCREEN_ENCODING_MODES:
        raise ValueError(
            f"screen_encoding_mode 必须是以下值之一：{SUPPORTED_SCREEN_ENCODING_MODES}"
        )
    if runtime_path is not None and runtime_template_target is not None:
        raise ValueError("runtime_path 与 runtime_template_target 不能同时提供")
    if runtime_path is None:
        if not auto_prepare_runtime:
            raise ValueError("关闭自动准备 runtime 时必须提供 runtime_path")
        template = prepare_runtime_template(runtime_template_target)
        resolved_runtime = prepare_runtime_instance(
            instance_id or f"environment-{uuid.uuid4().hex}",
            template=template,
            instances_root=runtime_instances_root,
        )
    else:
        resolved_runtime = Path(runtime_path).expanduser().resolve()
        if not resolved_runtime.is_dir():
            raise FileNotFoundError(f"CraftGround runtime 不存在：{resolved_runtime}")

    baseline_world = None
    if baseline_world_path is not None:
        baseline_world = install_baseline_world(
            baseline_world_path,
            resolved_runtime,
            world_directory_name=baseline_world_display_name,
        )

    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.screen_encoding_modes import ScreenEncodingMode

    resolved_screen_encoding_mode = {
        "raw": ScreenEncodingMode.RAW,
        "zerocopy_torch": ScreenEncodingMode.ZEROCOPY_TORCH,
    }[screen_encoding_mode]

    config = InitialEnvironmentConfig(
        image_width=image_width,
        image_height=image_height,
        seed=seed,
        render_distance=render_distance,
        simulation_distance=simulation_distance,
        screen_encoding_mode=resolved_screen_encoding_mode,
        level_display_name_to_play=(
            baseline_world_display_name if baseline_world is not None else ""
        ),
    )
    environment = CraftGroundEnvironment(
        config,
        action_space_version=getattr(ActionSpaceVersion, CRAFTGROUND_ACTION_SPACE),
        env_path=str(resolved_runtime),
        port=port,
        find_free_port=find_free_port,
        use_shared_memory=use_shared_memory,
        cleanup_world=cleanup_world,
        verbose=verbose,
        verbose_gradle=verbose_gradle,
        verbose_jvm=verbose_jvm,
    )
    environment.tao_baseline_world = baseline_world
    environment.tao_runtime_path = str(resolved_runtime)
    return environment
