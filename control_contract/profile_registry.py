# -*- coding: utf-8 -*-
"""随包发布的 BindingProfile 数据文件查找。

对外接口：
    PROFILE_DIRECTORY — 内置 profile JSON 目录。
    available_profile_names — 列出内置 profile 名。
    load_named_profile — 按名字加载内置 profile。

新增游戏或设备只需往 ``profiles/`` 放一个 JSON，无需改动任何 Python 代码。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from control_contract.binding_profile import BindingProfile, load_binding_profile

PROFILE_DIRECTORY = Path(__file__).parent / "profiles"


def available_profile_names() -> Tuple[str, ...]:
    """列出内置 profile 名（按字母序）。"""
    return tuple(sorted(path.stem for path in PROFILE_DIRECTORY.glob("*.json")))


def load_named_profile(name: str) -> BindingProfile:
    """按名字加载内置 profile。

    Parameters
    ----------
    name : str
        文件名（不含 ``.json``），如 ``minecraft_mouse_keyboard``。

    Returns
    -------
    BindingProfile

    Raises
    ------
    FileNotFoundError
        没有这个内置 profile。
    """
    path = PROFILE_DIRECTORY / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"内置 profile {name!r} 不存在；可用：{available_profile_names()}")
    return load_binding_profile(path)
