"""严格读取简单的 KEY=VALUE 环境变量文件。"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvironmentConfigurationError(ValueError):
    """环境配置文件不符合项目合同。"""


def load_env_file(
    path: Path,
    *,
    override: bool = False,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """读取环境变量文件。

    Args:
        path: UTF-8 或带 BOM 的 UTF-8 环境变量文件。
        override: 是否覆盖已经存在的环境变量。
        environ: 可选的目标映射；默认使用当前进程环境。

    Returns:
        本次实际写入的环境变量名称。

    Raises:
        EnvironmentConfigurationError: 文件包含无效行或变量名。
    """
    target = os.environ if environ is None else environ
    loaded: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise EnvironmentConfigurationError(f"{path}:{line_number} 缺少等号")
        name, value = (part.strip() for part in line.split("=", 1))
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise EnvironmentConfigurationError(
                f"{path}:{line_number} 包含无效环境变量名：{name!r}"
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if name in target and not override:
            continue
        target[name] = value
        loaded.append(name)
    return tuple(loaded)
