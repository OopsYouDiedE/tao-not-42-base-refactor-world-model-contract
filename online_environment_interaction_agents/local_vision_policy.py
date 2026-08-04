"""使用本地视觉模型生成可审计的 CraftGround 闭环动作。"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from online_interactive_environments import (
    extract_action_sequence_text,
    parse_action_sequence_strict,
)
from shared_tools.model_clients import ModelResponse

from .teacher_trajectory import TeacherModelError, TeacherRequest


@dataclass(frozen=True)
class LocalPolicyGeneration:
    """一次本地策略生成的文本、概率和输入来源。"""

    trajectory_id: str
    response_text: str
    response_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    observation_paths: tuple[str, ...]
    sampling_parameters: dict[str, Any]


class LocalVisionPolicyBackend:
    """加载一次视觉策略，并为多个真实环境分支串行执行 GPU 推理。"""

    provider = "local-transformers"

    def __init__(
        self,
        model_name: str,
        *,
        adapter: str,
        load_in_4bit: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
    ) -> None:
        from behavior_cloning_training import load_vision_model

        if temperature <= 0 or not 0 < top_p <= 1 or max_new_tokens < 1:
            raise ValueError("本地策略采样参数无效")
        self.model = model_name
        self.adapter = str(Path(adapter).resolve())
        self._model, self._processor = load_vision_model(
            model_name,
            adapter=adapter,
            load_in_4bit=load_in_4bit,
            max_sequence_length=4096,
        )
        self._parameters = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }
        self._lock = threading.Lock()
        self._records: list[LocalPolicyGeneration] = []

    @property
    def policy_version(self) -> str:
        return f"{self.model}@{self.adapter}"

    def records(self) -> tuple[LocalPolicyGeneration, ...]:
        with self._lock:
            return tuple(self._records)

    def generate(self, request: TeacherRequest) -> ModelResponse:
        import torch
        from PIL import Image

        paths = tuple(path.resolve() for path in request.observation_paths)
        if not paths or any(not path.is_file() for path in paths):
            raise TeacherModelError("本地视觉策略必须读取真实且存在的观察图片")
        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB").copy())
        prompt = "\n\n".join((request.system_prompt, request.task_context, request.step_context))
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            rendered = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(images=images, text=[rendered], return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            output = self._model.generate(
                **inputs, **self._parameters, return_dict_in_generate=True, output_scores=True
            )
            generated = output.sequences[0, inputs["input_ids"].shape[1] :]
            token_ids, response_text = _complete_protocol_prefix(
                tuple(map(int, generated.tolist())), self._processor.tokenizer
            )
            scores = torch.stack(tuple(score[0] for score in output.scores[: len(token_ids)]))
            selected = torch.tensor(token_ids, device=scores.device)[:, None]
            old_logprobs = scores.log_softmax(-1).gather(1, selected).squeeze(1)
            trajectory_id = _trajectory_id(request.task_context)
            self._records.append(
                LocalPolicyGeneration(
                    trajectory_id,
                    response_text,
                    token_ids,
                    tuple(map(float, old_logprobs.tolist())),
                    tuple(map(str, paths)),
                    dict(self._parameters),
                )
            )
        return ModelResponse(
            text=response_text,
            provider=self.provider,
            model=self.model,
            request_id=f"local-{uuid.uuid4().hex}",
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=len(token_ids),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def _complete_protocol_prefix(
    generated_ids: tuple[int, ...], tokenizer: Any
) -> tuple[tuple[int, ...], str]:
    for end in range(1, len(generated_ids) + 1):
        text = tokenizer.decode(generated_ids[:end], skip_special_tokens=False)
        try:
            control = extract_action_sequence_text(text)
            parse_action_sequence_strict(control)
        except ValueError:
            continue
        if text.strip() != control:
            raise TeacherModelError("本地策略动作协议之外包含非协议文字")
        return generated_ids[:end], control
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    raise TeacherModelError(f"本地策略未生成完整动作协议：{raw_text[:500]!r}")


def _trajectory_id(task_context: str) -> str:
    match = re.search(r"^trajectory_id:\s*(\S+)$", task_context, flags=re.MULTILINE)
    if match is None:
        raise TeacherModelError("策略请求缺少 trajectory_id")
    return match.group(1)
