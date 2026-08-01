"""教师大模型 API 的显式环境配置契约。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class TeacherAPIConfig:
    api_key: str
    model: str
    api_url: str

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("API_KEY 不能为空")
        if not self.model.strip():
            raise ValueError("API_MODEL 不能为空")
        parsed = urlparse(self.api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API_URL 必须是完整的 HTTP(S) URL")

    @classmethod
    def from_environment(cls) -> TeacherAPIConfig:
        missing = [name for name in ("API_KEY", "API_MODEL", "API_URL") if not os.getenv(name)]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(
                f"教师模型启动前必须 export {names}；"
                "可先 source <(python3 -m tools.export_codex_api_env)"
            )
        return cls(
            api_key=os.environ["API_KEY"],
            model=os.environ["API_MODEL"],
            api_url=os.environ["API_URL"],
        )

    def audit_dict(self) -> dict[str, str]:
        return {
            "model": self.model,
            "api_url": self.api_url,
            "api_key": "<redacted>",
        }
