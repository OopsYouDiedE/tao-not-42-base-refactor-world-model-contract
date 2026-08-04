"""不包含 Agent 领域语义的大模型传输类型。"""

from __future__ import annotations

from dataclasses import dataclass


class ModelTransportError(RuntimeError):
    """模型请求在传输、鉴权或厂商响应解析阶段失败。"""


@dataclass(frozen=True)
class ModelResponse:
    """一次大模型调用的文本、用量和延迟。"""

    text: str
    provider: str
    model: str
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: float
