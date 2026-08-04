"""2+6 clipped 相对优势视觉策略训练。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from behavior_cloning_training import LoRASettings, load_vision_model
from relative_advantage_comparison_training.objectives import clipped_token_joint_objective
from relative_advantage_comparison_training.policy_rollout import policy_prompt
from relative_advantage_comparison_training.rollouts import (
    RolloutSample,
    load_execution_group,
    require_on_policy_logprobs,
)


def top_half_training_mask(samples: list[RolloutSample]) -> Any:
    import torch

    if len(samples) != 8:
        raise ValueError("top-half selection requires one complete 2+6 group")
    ranked = sorted(
        range(8), key=lambda index: (-samples[index].reward, samples[index].candidate_id)
    )
    selected = torch.zeros(8, dtype=torch.bool)
    selected[ranked[:4]] = True
    return selected


def token_logprobs(logits: Any, labels: Any) -> tuple[Any, Any]:
    import torch

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have aligned batch dimensions")
    visual_tokens = logits.shape[1] - labels.shape[1]
    if visual_tokens < 0:
        raise ValueError("logits sequence is shorter than labels")
    if visual_tokens:
        labels = torch.nn.functional.pad(labels, (visual_tokens, 0), value=-100)
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    safe = targets.masked_fill(~mask, 0)
    values = logits[:, :-1].log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return values, mask


def align_rollout_metadata(
    samples: list[RolloutSample], input_ids: Any, labels: Any
) -> tuple[Any, Any, Any, Any]:
    import torch

    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have equal shapes")
    response_mask = labels[:, 1:].ne(-100)
    old = torch.zeros(response_mask.shape, dtype=torch.float32)
    for row, sample in enumerate(samples):
        if not sample.policy_eligible:
            continue
        positions = response_mask[row].nonzero(as_tuple=False).flatten()
        count = len(sample.response_token_ids)
        if len(positions) < count:
            raise ValueError(f"insufficient response positions for {sample.candidate_id}")
        labels[row, positions + 1] = -100
        rollout_positions = positions[:count]
        tokens = torch.tensor(
            sample.response_token_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        input_ids[row, rollout_positions + 1] = tokens
        labels[row, rollout_positions + 1] = tokens
        old[row, rollout_positions] = torch.tensor(sample.old_logprobs, dtype=torch.float32)
    advantages = torch.tensor(
        [sample.relative_advantage for sample in samples], dtype=torch.float32
    )
    selected = top_half_training_mask(samples)
    policy = torch.tensor([sample.policy_eligible for sample in samples]) & selected
    reference = torch.tensor([sample.behavior_cloning_eligible for sample in samples]) & selected
    return old, advantages, policy, reference


def _conversation(sample: RolloutSample, intent: str) -> dict[str, Any]:
    from PIL import Image

    images = []
    for path in sample.image_paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": policy_prompt(intent)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample.action_text}]},
        ]
    }


def run_policy_training(
    *,
    model: str,
    adapter: str,
    execution: Path,
    output: Path,
    intent: str,
    learning_rate: float = 1e-6,
    epochs: int = 1,
    clip_epsilon: float = 0.2,
) -> dict[str, Any]:
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    try:
        import torch
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as error:
        raise RuntimeError("RLHF training requires torch and unsloth") from error
    samples = load_execution_group(execution)
    require_on_policy_logprobs(samples)
    loaded, processor = load_vision_model(
        model, adapter=adapter, lora=LoRASettings(), max_sequence_length=4096
    )
    batch = UnslothVisionDataCollator(loaded, processor)(
        [_conversation(sample, intent) for sample in samples]
    )
    old, advantages, policy, reference = align_rollout_metadata(
        samples, batch["input_ids"], batch["labels"]
    )
    device = next(loaded.parameters()).device
    optimizer = torch.optim.AdamW(
        (parameter for parameter in loaded.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    output.mkdir(parents=True, exist_ok=True)
    history = []
    loaded.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        current, response_mask = token_logprobs(loaded(**inputs).logits, inputs["labels"])
        aligned_old = torch.nn.functional.pad(old, (current.shape[1] - old.shape[1], 0))
        result = clipped_token_joint_objective(
            current,
            aligned_old.to(device),
            response_mask,
            advantages.to(device),
            policy.to(device),
            reference.to(device),
            clip_epsilon=clip_epsilon,
        )
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in loaded.parameters() if parameter.requires_grad), 0.3
        )
        optimizer.step()
        metrics = {
            "epoch": epoch + 1,
            "loss": float(result.total.detach()),
            "policy_loss": float(result.relative_advantage.detach()),
            "bc_loss": float(result.behavior_cloning.detach()),
            "approximate_kl": float(result.approximate_kl.detach()),
            "clip_fraction": float(result.clip_fraction.detach()),
        }
        history.append(metrics)
        checkpoint = output / f"epoch-{epoch + 1:03d}"
        loaded.save_pretrained(str(checkpoint))
        processor.save_pretrained(str(checkpoint))
    adapter_path = output / "lora_adapter"
    loaded.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    payload = {
        "model": model,
        "source_adapter": adapter,
        "execution": str(execution),
        "adapter": str(adapter_path),
        "completed_epochs": epochs,
        "history": history,
        "protocol": "standard-input-action/v1",
    }
    (output / "training_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 2+6 标准动作视觉策略")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    arguments = parser.parse_args()
    result = run_policy_training(
        model=arguments.model,
        adapter=arguments.adapter,
        execution=arguments.execution,
        output=arguments.output,
        intent=arguments.intent,
        learning_rate=arguments.learning_rate,
        epochs=arguments.epochs,
        clip_epsilon=arguments.clip_epsilon,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
