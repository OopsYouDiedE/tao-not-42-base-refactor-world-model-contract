"""通过官方命令行工具检查本地鉴权状态。"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

AuthenticationState = Literal["authenticated", "unauthenticated", "unavailable", "skipped"]


@dataclass(frozen=True)
class AuthenticationStatus:
    provider: str
    method: str
    status: AuthenticationState
    remediation: str | None = None


def _check_cli(
    provider: str,
    executable: str,
    arguments: tuple[str, ...],
    login_command: str,
    *,
    skip: bool,
) -> AuthenticationStatus:
    if skip:
        return AuthenticationStatus(provider, f"{executable}-cli", "skipped")
    resolved = shutil.which(executable)
    if resolved is None:
        return AuthenticationStatus(
            provider,
            f"{executable}-cli",
            "unavailable",
            f"安装 {executable} 后执行 `{login_command}`",
        )
    try:
        completed = subprocess.run(
            [resolved, *arguments],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return AuthenticationStatus(
            provider,
            f"{executable}-cli",
            "unavailable",
            f"检查失败；确认 {executable} 可运行后重试",
        )
    if completed.returncode == 0:
        return AuthenticationStatus(provider, f"{executable}-cli", "authenticated")
    return AuthenticationStatus(
        provider,
        f"{executable}-cli",
        "unauthenticated",
        f"执行 `{login_command}`",
    )


def check_github_authentication(*, skip: bool = False) -> AuthenticationStatus:
    """通过 ``gh auth status`` 检查 GitHub 本地鉴权。"""
    return _check_cli("github", "gh", ("auth", "status"), "gh auth login", skip=skip)


def check_huggingface_authentication(*, skip: bool = False) -> AuthenticationStatus:
    """通过 ``hf auth whoami`` 检查 Hugging Face 本地鉴权。"""
    return _check_cli("huggingface", "hf", ("auth", "whoami"), "hf auth login", skip=skip)
