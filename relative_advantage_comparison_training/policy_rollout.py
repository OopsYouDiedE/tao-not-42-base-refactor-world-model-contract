"""生成带逐 token 行为概率的标准动作策略样本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from online_interactive_environments import (
    extract_action_sequence_text,
    parse_action_sequence_strict,
)


@dataclass(frozen=True)
class PolicyGeneration:
    action_text: str
    response_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    policy_version: str
    sampling_parameters: dict[str, Any]


def policy_prompt(intent: str) -> str:
    return (
        "Use the observations and intent to produce one executable standard-input-action/v1 "
        "sequence. Return Device KeyboardMouse, Tick lines, and <action> blocks only. "
        f"Intent: {intent}"
    )


def _complete_protocol_prefix(token_ids: Any, tokenizer: Any) -> tuple[Any, str]:
    values = token_ids.tolist()
    for end in range(1, len(values) + 1):
        text = tokenizer.decode(values[:end], skip_special_tokens=False)
        try:
            protocol = extract_action_sequence_text(text)
            parse_action_sequence_strict(protocol)
        except ValueError:
            continue
        return token_ids[:end], protocol
    raise RuntimeError(
        "policy generation did not contain a complete standard-input-action/v1 sequence"
    )


def generate_policy_rollouts(
    model: Any,
    processor: Any,
    images: list[Path],
    *,
    intent: str,
    policy_version: str,
    count: int = 6,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
) -> list[PolicyGeneration]:
    import torch
    from PIL import Image

    if count != 6:
        raise ValueError("the 2+6 contract requires exactly six policy samples")
    loaded_images = []
    for path in images:
        with Image.open(path) as source:
            loaded_images.append(source.convert("RGB").copy())
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in loaded_images),
                {"type": "text", "text": policy_prompt(intent)},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=loaded_images, text=[rendered], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()
    }
    parameters = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    results = []
    with torch.inference_mode():
        for _ in range(count):
            output = model.generate(
                **inputs, **parameters, return_dict_in_generate=True, output_scores=True
            )
            generated = output.sequences[0, inputs["input_ids"].shape[1] :]
            token_ids, action_text = _complete_protocol_prefix(generated, processor.tokenizer)
            scores = torch.stack(tuple(score[0] for score in output.scores[: token_ids.numel()]))
            logprobs = scores.log_softmax(-1).gather(1, token_ids[:, None]).squeeze(1)
            results.append(
                PolicyGeneration(
                    action_text,
                    tuple(map(int, token_ids.tolist())),
                    tuple(map(float, logprobs.tolist())),
                    policy_version,
                    dict(parameters),
                )
            )
    return results
