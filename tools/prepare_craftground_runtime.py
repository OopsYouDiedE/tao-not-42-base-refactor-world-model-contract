"""幂等准备带 TAO 内存快照扩展的 CraftGround runtime。"""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
from pathlib import Path

DISPATCH = "        if (MemorySnapshotStore.handle(command, client)) return true\n"
SIGNATURE = "    ): Boolean {\n"


def prepare(source_file: Path, target: Path, *, build: bool = True) -> Path:
    distribution = importlib.metadata.distribution("craftground-runtime-mc121")
    source = Path(distribution.locate_file("craftground_runtime_mc121")).resolve()
    if not target.exists():
        shutil.copytree(source, target)

    package = target / "src/main/java/com/kyhsgeekcode/minecraftenv"
    shutil.copy2(source_file, package / source_file.name)
    minecraft_env = package / "MinecraftEnv.kt"
    content = minecraft_env.read_text(encoding="utf-8")
    if DISPATCH not in content:
        handle_start = content.index("    private fun handleCommand(")
        insertion = content.index(SIGNATURE, handle_start) + len(SIGNATURE)
        content = content[:insertion] + DISPATCH + content[insertion:]
        minecraft_env.write_text(content, encoding="utf-8")

    if build:
        subprocess.run(
            [str(target / "gradlew"), "build", "--no-daemon"],
            cwd=target,
            check=True,
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="准备带 memorysnapshot 的 CraftGround runtime")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path.home() / ".cache/tao/craftground-runtime-patched",
    )
    parser.add_argument("--no-build", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = prepare(
        root / "game_environment/craftground_mod/MemorySnapshotStore.kt",
        arguments.target.expanduser().resolve(),
        build=not arguments.no_build,
    )
    print(runtime)


if __name__ == "__main__":
    main()
