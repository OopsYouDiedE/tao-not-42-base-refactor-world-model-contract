"""在线交互 Agent 使用的模型响应合同。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResponse:
    """一次模型调用的文本、用量和延迟。"""

    text: str
    provider: str
    model: str
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: float
