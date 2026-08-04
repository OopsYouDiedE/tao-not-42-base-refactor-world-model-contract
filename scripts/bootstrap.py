"""使用锁定或最新兼容依赖准备项目环境。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_tools.environment_checks import detect_accelerator  # noqa: E402


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="安装项目依赖")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--locked", action="store_true", help="使用锁定版本（默认）")
    mode.add_argument("--latest", action="store_true", help="解析兼容范围内的最新版本")
    parser.add_argument("--accelerator", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-github-auth", action="store_true")
    parser.add_argument("--skip-huggingface-auth", action="store_true")
    parser.add_argument("--skip-optional-auth", action="store_true")
    arguments = parser.parse_args()
    accelerator = detect_accelerator() if arguments.accelerator == "auto" else arguments.accelerator
    if arguments.latest:
        target = ".[cuda]" if accelerator == "cuda" else "."
        _run([sys.executable, "-m", "pip", "install", "-e", target])
    else:
        lock = ROOT / "requirements" / f"locked-{accelerator}.txt"
        _run([sys.executable, "-m", "pip", "install", "-r", str(lock)])
        _run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."])
    if accelerator == "cpu":
        torch_specification = "torch" if arguments.latest else "torch==2.12.0"
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                torch_specification,
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
    check_command = [
        sys.executable,
        str(ROOT / "scripts" / "check_environment.py"),
        "--accelerator",
        accelerator,
    ]
    for enabled, flag in (
        (arguments.skip_github_auth, "--skip-github-auth"),
        (arguments.skip_huggingface_auth, "--skip-huggingface-auth"),
        (arguments.skip_optional_auth, "--skip-optional-auth"),
    ):
        if enabled:
            check_command.append(flag)
    _run(check_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
