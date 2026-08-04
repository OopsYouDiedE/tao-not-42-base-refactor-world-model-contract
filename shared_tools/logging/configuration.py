"""仅由 CLI 入口调用的日志初始化。"""

from __future__ import annotations

import logging
from pathlib import Path

from .jsonl import JsonlHandler


def configure_logging(
    *, level: int = logging.INFO, events_path: Path | None = None
) -> logging.Logger:
    """配置根 logger，并可选写入结构化事件文件。"""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    try:
        from rich.logging import RichHandler

        console: logging.Handler = RichHandler(show_path=False, markup=False)
    except ImportError:
        console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    if events_path is not None:
        root.addHandler(JsonlHandler(events_path))
    return root
