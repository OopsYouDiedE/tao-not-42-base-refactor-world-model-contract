"""环境变量与 `.env` 文件读取。

配置读取只在 CLI 装配时发生，业务模块不在导入时调用这里的函数。`.env` 只补齐尚未
设置的变量，因此运行环境里已有的值（含 Secret 注入）永远优先于文件内容。
"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvironmentConfigurationError(RuntimeError):
    """环境变量缺失或 `.env` 内容不合法。"""


def require_env(name: str, *, environ: MutableMapping[str, str] | None = None) -> str:
    """读取必须存在且非空的环境变量。"""
    source = os.environ if environ is None else environ
    value = source.get(name, "").strip()
    if not value:
        raise EnvironmentConfigurationError(f"缺少环境变量 {name}")
    return value


def load_env_file(
    path: Path | str,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """把 `.env` 中尚未设置的变量补进环境，返回本次真正写入的变量名。

    忽略空行与 `#` 注释；值两侧的单引号或双引号成对出现时去掉。已经存在的变量不被
    覆盖，使显式设置的运行环境优先于文件。

    Raises:
        EnvironmentConfigurationError: 出现无效变量名或缺少 `=` 的行。
    """
    target = os.environ if environ is None else environ
    content = Path(path).expanduser().read_text(encoding="utf-8")
    loaded: list[str] = []
    for number, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator:
            raise EnvironmentConfigurationError(f"第 {number} 行缺少 `=`：{raw!r}")
        name = name.strip()
        if not _NAME.fullmatch(name):
            raise EnvironmentConfigurationError(f"第 {number} 行出现无效环境变量名：{name!r}")
        if name in target:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        target[name] = value
        loaded.append(name)
    return tuple(loaded)
