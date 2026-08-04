"""厂商无关的大模型传输合同和流解析。"""

from .contracts import ModelResponse, ModelTransportError
from .streaming import iter_sse_json

__all__ = ["ModelResponse", "ModelTransportError", "iter_sse_json"]
