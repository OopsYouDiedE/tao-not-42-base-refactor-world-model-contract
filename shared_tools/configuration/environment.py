"""类型明确的环境变量读取函数。"""

from __future__ import annotations

import os
from collections.abc import Mapping

from .env_files import EnvironmentConfigurationError


def require_env(name: str, *, environ: Mapping[str, str] | None = None) -> str:
    """读取非空的必填环境变量。"""
    source = os.environ if environ is None else environ
    value = source.get(name, "").strip()
    if not value:
        raise EnvironmentConfigurationError(f"缺少环境变量：{name}")
    return value
