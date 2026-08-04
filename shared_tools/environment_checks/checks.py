"""操作系统、Python 和计算后端检查。"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

CheckState = Literal["passed", "failed", "unavailable", "not_applicable"]
Accelerator = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: CheckState
    summary: str
    details: dict[str, object]
    remediation: str | None = None


def detect_accelerator() -> Accelerator:
    """根据真实 PyTorch CUDA 状态解析计算后端。"""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _command_version(executable: str, arguments: tuple[str, ...]) -> CheckResult:
    resolved = shutil.which(executable)
    if resolved is None:
        return CheckResult(executable, "unavailable", f"未找到 {executable}", {})
    try:
        completed = subprocess.run(
            [resolved, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return CheckResult(executable, "failed", f"{executable} 执行失败", {"error": str(error)})
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return CheckResult(
        executable,
        "passed" if completed.returncode == 0 else "failed",
        f"{executable} 可执行" if completed.returncode == 0 else f"{executable} 返回失败",
        {"version": output[0] if output else None, "exit_code": completed.returncode},
    )


def check_environment(accelerator: Accelerator | None = None) -> tuple[CheckResult, ...]:
    """检查当前 Python、核心依赖和目标计算后端。"""
    selected = accelerator or detect_accelerator()
    results = [
        CheckResult(
            "python",
            "passed" if sys.version_info >= (3, 11) else "failed",
            f"Python {platform.python_version()}",
            {"executable": sys.executable, "platform": platform.platform()},
        )
    ]
    for package in ("httpx", "rich", "huggingface_hub", "numpy", "PIL"):
        available = importlib.util.find_spec(package) is not None
        results.append(
            CheckResult(
                f"package:{package}",
                "passed" if available else "failed",
                f"{package} {'已安装' if available else '未安装'}",
                {},
            )
        )
    try:
        import torch

        cpu_value = torch.ones(2, dtype=torch.float32).sum().item()
        results.append(
            CheckResult("torch:cpu", "passed", "PyTorch CPU 计算通过", {"value": cpu_value})
        )
        if selected == "cuda":
            if torch.cuda.is_available():
                cuda_value = torch.ones(2, device="cuda").sum().item()
                results.append(
                    CheckResult(
                        "torch:cuda",
                        "passed",
                        "PyTorch CUDA 计算通过",
                        {"value": cuda_value, "device": torch.cuda.get_device_name(0)},
                    )
                )
            else:
                results.append(CheckResult("torch:cuda", "failed", "PyTorch 无法使用 CUDA", {}))
        else:
            results.append(CheckResult("torch:cuda", "not_applicable", "CPU 安装不验证 CUDA", {}))
    except ImportError:
        results.append(CheckResult("torch:cpu", "failed", "未安装 PyTorch", {}))
    results.extend((_command_version("gh", ("--version",)), _command_version("hf", ("--version",))))
    return tuple(results)
