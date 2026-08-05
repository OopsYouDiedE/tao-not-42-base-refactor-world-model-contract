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


class ScriptedBackend:
    """按预先写定的动作序列作答的后端；人工提交动作时代替模型。

    可视化控制台和确定性回归都需要在不调用任何模型的前提下把一段动作送进执行器，
    因此这是正式实现而不是测试替身。每次 `generate` 取出下一段脚本；脚本用尽后
    重复最后一段，使调用方可以持续推进而不必自己补齐轮数。
    """

    provider = "scripted"

    def __init__(self, *sequences: str, model: str = "scripted-action-sequence") -> None:
        if not sequences:
            raise ValueError("ScriptedBackend 至少需要一段动作序列")
        for sequence in sequences:
            extract_action_sequence_text(sequence)
        self.model = model
        self.sequences = tuple(sequences)
        self.calls = 0

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        index = min(self.calls, len(self.sequences) - 1)
        self.calls += 1
        text = self.sequences[index]
        return TeacherResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            request_id=f"scripted-{index}",
            input_tokens=None,
            output_tokens=None,
            elapsed_ms=0.0,
        )
