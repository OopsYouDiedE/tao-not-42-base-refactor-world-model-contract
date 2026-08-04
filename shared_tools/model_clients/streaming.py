"""Server-Sent Events 流的通用 JSON 数据解析。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from .contracts import ModelTransportError


def iter_sse_json(lines: Iterable[bytes | str]) -> Iterator[dict[str, Any]]:
    """解析 SSE 中的 JSON data 事件，忽略注释和结束标记。"""
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise ModelTransportError("SSE data 事件不是有效 JSON") from error
        if not isinstance(event, dict):
            raise ModelTransportError("SSE data 事件必须是 JSON 对象")
        yield event
