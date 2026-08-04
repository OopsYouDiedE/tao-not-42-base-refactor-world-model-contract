"""项目静态图输出规范。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_figure(figure: Any, path: Path, *, dpi: int = 200, close: bool = True) -> None:
    """以稳定尺寸边界保存 Matplotlib figure。"""
    if dpi < 150:
        raise ValueError("静态图分辨率不得低于 150 DPI")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError("关闭静态图需要 matplotlib") from error
        plt.close(figure)
