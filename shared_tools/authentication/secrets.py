"""防止意外显示明文的秘密值容器。"""

from __future__ import annotations


class SecretValue:
    """仅允许在外部请求边界显式读取的秘密值。"""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not value.strip():
            raise ValueError("秘密值不能为空")
        self._value = value

    def reveal(self) -> str:
        """返回明文；调用方不得记录返回值。"""
        return self._value

    def __repr__(self) -> str:
        return "SecretValue('***')"

    def __str__(self) -> str:
        return "***"
