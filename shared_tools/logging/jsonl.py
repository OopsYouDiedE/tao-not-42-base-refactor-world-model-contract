"""标准库 logging 的 JSONL handler。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonlHandler(logging.Handler):
    """将日志记录写为 UTF-8 JSONL。"""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", newline="")

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.msg if isinstance(record.msg, str) else "log"),
            "message": record.getMessage(),
            "component": record.name,
        }
        for name, value in record.__dict__.items():
            if name not in _STANDARD_RECORD_FIELDS and name not in payload:
                payload[name] = value
        try:
            self._stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._stream.close()
        super().close()
