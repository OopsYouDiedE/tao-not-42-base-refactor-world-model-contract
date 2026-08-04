"""在线交互 Agent 使用的教师模型合同。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from online_interactive_environments import extract_action_sequence_text


class TeacherModelError(RuntimeError):
    """教师模型调用或输出校验失败。"""


@dataclass(frozen=True)
class TeacherDecisionEnvelope:
    control: str
    non_control_text: str


def parse_teacher_decision(text: str) -> TeacherDecisionEnvelope:
    """提取唯一动作控制块，并保留未执行的响应外壳用于审计。"""
    normalized = text.strip()
    if not normalized:
        raise ValueError("教师动作不能为空")
    control = extract_action_sequence_text(normalized)
    start = normalized.index(control)
    non_control = (normalized[:start] + normalized[start + len(control) :]).strip()
    return TeacherDecisionEnvelope(control, non_control)


@dataclass(frozen=True)
class TeacherRequest:
    system_prompt: str
    task_context: str
    step_context: str
    observation_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TeacherResponse:
    """一次模型调用的文本、用量和延迟。"""

    text: str
    provider: str
    model: str
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: float


class TeacherBackend(Protocol):
    provider: str
    model: str

    def generate(self, request: TeacherRequest) -> TeacherResponse: ...
