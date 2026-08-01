"""从 BC 视觉 LoRA 生成带逐 token 行为概率的 TAP policy rollout。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tao.protocols.action import decode_action_sequence


@dataclass(frozen=True)
class PolicyGeneration:
    action_text: str
    response_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    policy_version: str
    sampling_parameters: dict[str, Any]


def _prompt() -> str:
    return (
        "The image is the current Minecraft observation and the intent is supplied as text. "
        "Infer one reasonable action sequence for the supplied future horizon that advances "
        "this intent. Return one valid action block with exactly the supplied number of 50 ms "
        "ticks. Preserve the required duration of mining, movement, bow drawing, eating, or "
        "continuous use; omit unsupported 1-2 pixel camera jitter and preserve GUI click order. "
        "Return only a JSON array.\n"
        "Required action-block tick counts: [40]\n"
        "Action format example for a 3-tick block: "
        '"<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>". '
        "Each JSON array item must be one string action block; do not return nested tick arrays.\n"
        "Output the complete executable JSON action array first. Then start a new line with "
        '"Reason:" and briefly explain the visual evidence, intent, and duration choice. The '
        "action array must remain independently parseable because generation may stop after it.\n"
        "Intent: Safely approach the visible tree trunk while remaining alive and on the ground."
    )


def _truncate_at_action_end(token_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    """保留第一个完整 action 结束标记，丢弃其后的自由文本。"""
    markers = [
        tokenizer.encode(prefix + "<|action_end|>", add_special_tokens=False)
        for prefix in ("", " ", "\n")
    ]
    markers = [marker for marker in markers if marker]
    values = token_ids.tolist()
    if not markers:
        raise RuntimeError("tokenizer 无法编码 <|action_end|>")
    matches = [
        start + len(marker)
        for marker in markers
        for start in range(len(values) - len(marker) + 1)
        if values[start : start + len(marker)] == marker
    ]
    if matches:
        return token_ids[: min(matches)]
    preview = tokenizer.decode(token_ids[:256], skip_special_tokens=False)
    raise RuntimeError(f"模型输出未包含 <|action_end|>；输出前缀={preview!r}")


@torch.inference_mode()
def generate_policy_rollouts(
    model: Any,
    processor: Any,
    image_path: Path,
    *,
    policy_version: str,
    count: int = 6,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
) -> list[PolicyGeneration]:
    """对同一观察独立采样，并保存生成策略下的逐 token logprob。"""
    if count != 6:
        raise ValueError("2+6 合同要求正好生成 6 条 policy rollout")
    if temperature <= 0 or not 0 < top_p <= 1 or max_new_tokens < 8:
        raise ValueError("采样参数不合法")
    with Image.open(image_path) as source:
        image = source.convert("RGB").copy()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _prompt()},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=[image], text=[rendered], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    parameters = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    generations: list[PolicyGeneration] = []
    for _ in range(count):
        output = model.generate(
            **inputs,
            **parameters,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0, prompt_length:]
        token_ids = _truncate_at_action_end(generated_ids, processor.tokenizer)
        if not output.scores or len(output.scores) != generated_ids.numel():
            raise RuntimeError("模型未返回与生成 token 对齐的 scores")
        score_tensor = torch.stack(tuple(score[0] for score in output.scores[: token_ids.numel()]))
        logprobs = score_tensor.log_softmax(dim=-1).gather(1, token_ids[:, None]).squeeze(1)
        text = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
        action = decode_action_sequence(text, expected_ticks=None)
        if len(action.ticks) < 8:
            raise RuntimeError(f"模型输出只有 {len(action.ticks)} tick，低于合同要求的 8 tick")
        generations.append(
            PolicyGeneration(
                action_text=text,
                response_token_ids=tuple(int(value) for value in token_ids.tolist()),
                old_logprobs=tuple(float(value) for value in logprobs.tolist()),
                policy_version=policy_version,
                sampling_parameters=dict(parameters),
            )
        )
    return generations
