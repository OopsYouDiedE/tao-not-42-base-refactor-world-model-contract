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
        "Use the observations and intent to produce exactly one executable "
        "standard-input-action/v1 sequence. The first line must be `Device KeyboardMouse`. "
        "Each tick header must be outside its action block. Return protocol text only, with "
        "no Markdown fence or explanation. Required structure:\n"
        "Device KeyboardMouse\n"
        "Tick 0\n"
        "<action>NoOp</action>\n"
        "Replace NoOp with the action required by the intent. "
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


def _allowed_next_tokens(
    generated_ids: tuple[int, ...], candidates: tuple[tuple[int, ...], ...], eos_token_id: int
) -> list[int]:
    allowed: set[int] = set()
    for candidate in candidates:
        if candidate[: len(generated_ids)] != generated_ids:
            continue
        if len(generated_ids) == len(candidate):
            allowed.add(eos_token_id)
        else:
            allowed.add(candidate[len(generated_ids)])
    if not allowed:
        raise RuntimeError("generated tokens left the configured protocol candidate set")
    return sorted(allowed)


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
    allowed_action_texts: tuple[str, ...] | None = None,
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
    generation_parameters = dict(parameters)
    if allowed_action_texts is not None:
        if not allowed_action_texts:
            raise ValueError("allowed_action_texts must not be empty")
        for action_text in allowed_action_texts:
            parse_action_sequence_strict(action_text)
        candidate_ids = tuple(
            tuple(processor.tokenizer.encode(text, add_special_tokens=False))
            for text in allowed_action_texts
        )
        prompt_length = inputs["input_ids"].shape[1]
        eos_token_id = processor.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("the processor tokenizer must define eos_token_id")

        def prefix_allowed_tokens_fn(_batch_id: int, token_ids: Any) -> list[int]:
            generated_ids = tuple(map(int, token_ids[prompt_length:].tolist()))
            return _allowed_next_tokens(generated_ids, candidate_ids, eos_token_id)

        generation_parameters["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        parameters["allowed_action_texts"] = list(allowed_action_texts)
    results = []
    with torch.inference_mode():
        for _ in range(count):
            output = model.generate(
                **inputs,
                **generation_parameters,
                return_dict_in_generate=True,
                output_scores=True,
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
