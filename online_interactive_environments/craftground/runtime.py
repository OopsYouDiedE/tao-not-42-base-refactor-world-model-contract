"""CraftGround runtime 补丁安装与环境创建入口。"""

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
from typing import Any

DISPATCH = "        if (MemorySnapshotStore.handle(command, client)) return true\n"
SIGNATURE = "    ): Boolean {\n"
# runtime 编译的 noboost_ipc.cpp 只映射 24 字节的 j2p 头部，随后把整帧观察 memcpy
# 到该映射之后，导致 JVM 在 writeObservationImpl 内 SIGSEGV。同仓库的 boost_ipc.cpp
# 已有正确做法（先按需扩容再映射），这里把同样的扩容补进被编译的那份实现。
OBSERVATION_MMAP_ORIGINAL = """    void *j2pPtr = mmap(
        0,
        sizeof(J2PSharedMemoryLayout),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        j2pFd,
        0
    );
    if (j2pPtr == MAP_FAILED) {
        perror("mmap j2p failed while writing to shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        close(j2pFd);
        return;
    }
#endif
    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pPtr);
    j2pLayout->data_offset = sizeof(J2PSharedMemoryLayout);
"""
OBSERVATION_MMAP_PATCHED = """    const size_t j2pRequiredSize =
        sizeof(J2PSharedMemoryLayout) + observation_size;
    struct stat j2pStat;
    if (fstat(j2pFd, &j2pStat) == -1) {
        perror("fstat j2p failed while writing to shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        close(j2pFd);
        return;
    }
    if (static_cast<size_t>(j2pStat.st_size) < j2pRequiredSize &&
        ftruncate(j2pFd, static_cast<off_t>(j2pRequiredSize)) == -1) {
        perror("ftruncate j2p failed while writing to shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        close(j2pFd);
        return;
    }
    void *j2pPtr = mmap(
        0,
        j2pRequiredSize,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        j2pFd,
        0
    );
    if (j2pPtr == MAP_FAILED) {
        perror("mmap j2p failed while writing to shared memory");
        munmap(p2jPtr, sizeof(SharedMemoryLayout));
        close(p2jFd);
        close(j2pFd);
        return;
    }
#endif
    J2PSharedMemoryLayout *j2pLayout =
        static_cast<J2PSharedMemoryLayout *>(j2pPtr);
    j2pLayout->layout_size = sizeof(J2PSharedMemoryLayout);
    j2pLayout->data_offset = sizeof(J2PSharedMemoryLayout);
"""
OBSERVATION_UNMAP_ORIGINAL = "    munmap(j2pPtr, sizeof(J2PSharedMemoryLayout));\n"
OBSERVATION_UNMAP_PATCHED = "    munmap(j2pPtr, j2pRequiredSize);\n"
_PREPARE_LOCK = Lock()
ACTION_BACKEND = "keyboard_and_mouse_only"
CRAFTGROUND_ACTION_SPACE = "V2_MINERL_HUMAN"
_INSTANCE_MARKER = ".tao-runtime-instance"
_SHARED_MEMORY_PATCH_LOCK = Lock()
_SHARED_MEMORY_PATCHED = False
# 动作段容量按最长命令估算；内存快照 save 命令是当前最长的一条。
_LONGEST_COMMAND = "memorysnapshot save " + "s" * 64 + " -30000 -64 -30000 30000 320 30000"
_COMMAND_SLOTS = 8
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


def action_segment_capacity() -> int:
    """按最大真实动作消息给出动作段容量。

    上游用 `no_op_v2()` 的序列化长度作为动作段容量，而全默认值的 protobuf 序列化为
    0 字节。任何真实动作都超过该容量，JVM 写入时越界并以 SIGABRT 终止。这里改用
    “全部按键按下 + 非零视角 + 一条最长命令” 的消息长度，并留出余量。
    """
    from craftground.environment.action_space import action_v2_dict_to_message, no_op_v2

    action = no_op_v2()
    for key in action:
        action[key] = True if isinstance(action[key], bool) else 24.0
    message = action_v2_dict_to_message(action)
    message.commands.extend([_LONGEST_COMMAND] * _COMMAND_SLOTS)
    return len(message.SerializeToString()) * 2


def enable_shared_memory_reuse() -> None:
    """修正共享内存段容量，并让同一端口只初始化一次。

    上游共享内存路径有两处缺陷。

    其一是重复初始化：`CraftGroundEnvironment.__init__` 已经通过 `BoostIPC` 创建
    `/craftground_<port>_p2j` 与 `_j2p` 并写入初始环境消息，随后 `reset()` 内的
    `ensure_alive()` 会对同一端口再构造一个 `BoostIPC`，命中 native 层的
    “already exists” 检查而失败。先 `destroy()` 再让它重建同样不可行：destroy 之后
    native 模块对该段名的映射失效，读侧拿到坏 fd 并返回空字节，Python 侧表现为
    `cannot reshape array of size 0`。

    其二是动作段容量按 `no_op_v2()` 计算，而该消息序列化为 0 字节，JVM 写入真实
    观察时越界并以退出码 134 终止。

    这里按端口缓存首次初始化结果：重复构造复用已有段名，不再调用 native 初始化，
    也不销毁正在使用的段；容量改由 `action_segment_capacity()` 给出。同时把隐式析构
    改为显式 `release()`，避免被丢弃的旧实例 unlink 掉新实例仍在使用的段。
    """
    global _SHARED_MEMORY_PATCHED
    with _SHARED_MEMORY_PATCH_LOCK:
        if _SHARED_MEMORY_PATCHED:
            return
        from craftground.craftground_native import initialize_shared_memory
        from craftground.environment.boost_ipc import BoostIPC

        registry: dict[int, int] = {}
        registry_lock = Lock()
        capacity = action_segment_capacity()

        def patched_init(self, port, find_free_port, initial_environment, logger):
            self.logger = logger
            self.find_free_port = find_free_port
            self.SHMEM_PREFIX = "Global\\" if sys.platform == "win32" else "/"
            with registry_lock:
                established = registry.get(int(port))
                if established is None:
                    initial_bytes = initial_environment.SerializeToString()
                    established = initialize_shared_memory(
                        int(port),
                        initial_bytes,
                        len(initial_bytes),
                        capacity,
                        find_free_port,
                    )
                    registry[int(port)] = established
            self.port = established
            self.p2j_shared_memory_name = f"{self.SHMEM_PREFIX}craftground_{established}_p2j"
            self.j2p_shared_memory_name = f"{self.SHMEM_PREFIX}craftground_{established}_j2p"

        def release(self) -> None:
            """真正回收本端口的共享内存段；只应在环境关闭时调用一次。"""
            port = getattr(self, "port", None)
            with registry_lock:
                if port is not None:
                    registry.pop(int(port), None)
            original_destroy(self)

        original_destroy = BoostIPC.destroy
        BoostIPC.__init__ = patched_init
        # `terminate()` 与 `__del__` 都会调用 destroy；改为空操作后由 release 显式回收。
        BoostIPC.destroy = lambda self: None
        BoostIPC.release = release
        _SHARED_MEMORY_PATCHED = True


def prepare_patched_runtime(target: Path | None = None, *, build: bool = True) -> Path:
    """把已安装的 CraftGround runtime 复制到缓存、注入补丁并按需构建。"""
    distribution = importlib.metadata.distribution("craftground-runtime-mc121")
    source = Path(distribution.locate_file("craftground_runtime_mc121")).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"找不到 CraftGround runtime 源目录：{source}")

    if target is None:
        safe_version = re.sub(r"[^A-Za-z0-9_.-]", "-", distribution.version)
        target = (
            Path.home()
            / ".cache"
            / "tao"
            / f"craftground-runtime-mc121-{safe_version}-memorysnapshot"
        )
    target = target.expanduser().resolve()
    patch = Path(__file__).with_name("runtime_patch") / "MemorySnapshotStore.kt"
    # 构建标记覆盖全部补丁内容：Kotlin 快照扩展和 j2p 观察写入扩容。
    patch_digest = hashlib.sha256(
        patch.read_bytes() + OBSERVATION_MMAP_PATCHED.encode("utf-8")
    ).hexdigest()

    with _PREPARE_LOCK:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)

        package = target / "src/main/java/com/kyhsgeekcode/minecraftenv"
        minecraft_env = package / "MinecraftEnv.kt"
        if not minecraft_env.is_file():
            raise FileNotFoundError(f"runtime 缺少 MinecraftEnv.kt：{minecraft_env}")

        installed_patch = package / patch.name
        content = minecraft_env.read_text(encoding="utf-8")
        changed = (
            not installed_patch.is_file() or installed_patch.read_bytes() != patch.read_bytes()
        )
        if changed:
            shutil.copy2(patch, installed_patch)
        if DISPATCH not in content:
            handle_start = content.index("    private fun handleCommand(")
            insertion = content.index(SIGNATURE, handle_start) + len(SIGNATURE)
            minecraft_env.write_text(
                content[:insertion] + DISPATCH + content[insertion:],
                encoding="utf-8",
            )
            changed = True
        changed = _patch_observation_write(target) or changed

        build_marker = target / ".tao-memorysnapshot-build"
        already_built = (
            build_marker.is_file()
            and build_marker.read_text(encoding="ascii").strip() == patch_digest
        )
        if build and (changed or not already_built):
            gradle = target / ("gradlew.bat" if sys.platform == "win32" else "gradlew")
            if not gradle.is_file():
                raise FileNotFoundError(f"runtime 缺少 Gradle wrapper：{gradle}")
            subprocess.run(
                [str(gradle), "build", "--no-daemon"],
                cwd=target,
                check=True,
            )
            build_marker.write_text(patch_digest + "\n", encoding="ascii")
    return target


def _patch_observation_write(target: Path) -> bool:
    """让被编译的 IPC 实现在写观察前扩容并映射完整 j2p 段。

    runtime 的 `CMakeLists.txt` 编译 `noboost_ipc.cpp`。该文件的 `write_observation`
    只映射 `sizeof(J2PSharedMemoryLayout)`（24 字节），随后把整帧观察 memcpy 到该映射
    之后；640x360x3 的观察约 691 KB，于是 JVM 在
    `Java_..._writeObservationImpl` 内 SIGSEGV，Gradle 报退出码 134。同仓库的
    `boost_ipc.cpp` 已有正确做法：先 `truncate` 到
    `observation_size + sizeof(J2PSharedMemoryLayout)` 再映射。这里把等价的扩容补进
    实际参与编译的那一份。
    """
    source = target / "src/main/cpp/noboost_ipc.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"runtime 缺少 noboost_ipc.cpp：{source}")
    content = source.read_text(encoding="utf-8")
    if OBSERVATION_MMAP_PATCHED in content:
        return False
    if OBSERVATION_MMAP_ORIGINAL not in content:
        raise RuntimeError(f"noboost_ipc.cpp 的 j2p 映射代码与预期不符，无法安全打补丁：{source}")
    patched = content.replace(OBSERVATION_MMAP_ORIGINAL, OBSERVATION_MMAP_PATCHED, 1)
    write_start = patched.index(OBSERVATION_MMAP_PATCHED)
    unmap_at = patched.index(OBSERVATION_UNMAP_ORIGINAL, write_start)
    source.write_text(
        patched[:unmap_at]
        + OBSERVATION_UNMAP_PATCHED
        + patched[unmap_at + len(OBSERVATION_UNMAP_ORIGINAL) :],
        encoding="utf-8",
    )
    return True


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
        prepare_patched_runtime() if template is None else template.expanduser().resolve()
    )
    if not resolved_template.is_dir():
        raise FileNotFoundError(f"CraftGround runtime 模板不存在：{resolved_template}")
    root = (
        Path.home() / ".cache" / "tao" / "craftground-runtime-instances"
        if instances_root is None
        else instances_root.expanduser().resolve()
    )
    target = root / safe_id
    template_marker = resolved_template / ".tao-memorysnapshot-build"
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
    auto_install_patch: bool = True,
    patched_runtime_target: Path | None = None,
    instance_id: str | None = None,
    runtime_instances_root: Path | None = None,
    baseline_world_path: Path | str | None = None,
    baseline_world_display_name: str = "New World",
    cleanup_world: bool = True,
    verbose: bool = False,
) -> Any:
    """创建 CraftGround 键鼠后端环境；默认自动准备内存快照 runtime。

    默认使用共享内存 IPC，观察不经 socket 序列化。共享内存段按端口命名，配合
    `enable_shared_memory_reuse()` 消除上游同端口重复初始化，可支持多实例并行；
    该路径也不触发 SocketIPC 那个不分端口的全局 java 进程扫描。
    """
    if runtime_path is not None and patched_runtime_target is not None:
        raise ValueError("runtime_path 与 patched_runtime_target 不能同时提供")
    if runtime_path is None:
        if not auto_install_patch:
            raise ValueError("关闭自动补丁时必须提供 runtime_path")
        template = prepare_patched_runtime(patched_runtime_target)
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

    if use_shared_memory:
        enable_shared_memory_reuse()
    config = InitialEnvironmentConfig(
        image_width=image_width,
        image_height=image_height,
        seed=seed,
        render_distance=render_distance,
        simulation_distance=simulation_distance,
        screen_encoding_mode=ScreenEncodingMode.RAW,
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
    )
    environment.tao_baseline_world = baseline_world
    environment.tao_runtime_path = str(resolved_runtime)
    if use_shared_memory:
        _release_shared_memory_on_close(environment)
    return environment


def _release_shared_memory_on_close(environment: Any) -> None:
    """把共享内存段的回收绑定到环境关闭，替代被停用的隐式析构。"""
    original_close = environment.close

    def close() -> None:
        try:
            original_close()
        finally:
            release = getattr(environment.ipc, "release", None)
            if release is not None:
                release()

    environment.close = close
