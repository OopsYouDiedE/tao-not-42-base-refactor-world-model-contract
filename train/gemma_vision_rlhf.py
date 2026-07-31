"""Gemma 4 视觉 2+6 clipped RLHF 训练入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import unsloth  # noqa: F401
from huggingface_hub import HfApi
from PIL import Image
from unsloth.trainer import UnslothVisionDataCollator

from train.objectives import clipped_token_joint_objective
from train.policy_rollout import _prompt
from train.rollout_contract import RolloutSample, load_execution_group, require_on_policy_logprobs
from train.unsloth_vision_sft import LoRASettings, load_vision_model


def top_half_training_mask(samples: list[RolloutSample]) -> torch.Tensor:
    """在同一 2+6 组内按真实奖励选择前四名。"""
    if len(samples) != 8:
        raise ValueError("top-half 筛选要求正好 8 条轨迹")
    ranked = sorted(
        range(len(samples)),
        key=lambda index: (-samples[index].reward, samples[index].candidate_id),
    )
    selected = torch.zeros(8, dtype=torch.bool)
    selected[ranked[:4]] = True
    return selected


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回与 causal labels 对齐的逐 token logprob 和响应 mask。"""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits/labels 形状不匹配：logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    visual_tokens = logits.shape[1] - labels.shape[1]
    if visual_tokens < 0:
        raise ValueError("logits 序列短于 labels")
    if visual_tokens:
        labels = torch.nn.functional.pad(labels, (visual_tokens, 0), value=-100)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    values = logits[:, :-1].log_softmax(dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return values, mask


def _conversation(sample: RolloutSample) -> dict[str, Any]:
    with Image.open(sample.image_paths[0]) as image:
        policy_anchor = image.convert("RGB").copy()
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": policy_anchor},
                    {"type": "text", "text": _prompt()},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample.action_text}]},
        ]
    }


def align_rollout_metadata(
    samples: list[RolloutSample],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 rollout 的旧概率和来源元数据对齐到 causal label 位置。"""
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids/labels 形状不匹配")
    targets = labels[:, 1:]
    response_mask = targets.ne(-100)
    old = torch.zeros(targets.shape, dtype=torch.float32)
    for row, sample in enumerate(samples):
        positions = response_mask[row].nonzero(as_tuple=False).flatten()
        if sample.policy_eligible:
            token_count = len(sample.response_token_ids)
            if len(positions) < token_count:
                raise ValueError(
                    f"{sample.candidate_id} 的训练响应位置少于 rollout token"
                )
            labels[row, positions + 1] = -100
            rollout_positions = positions[:token_count]
            rollout_ids = torch.tensor(
                sample.response_token_ids, dtype=input_ids.dtype, device=input_ids.device
            )
            input_ids[row, rollout_positions + 1] = rollout_ids
            labels[row, rollout_positions + 1] = rollout_ids
            old[row, rollout_positions] = torch.tensor(
                sample.old_logprobs, dtype=torch.float32
            )
    advantages = torch.tensor(
        [sample.relative_advantage for sample in samples], dtype=torch.float32
    )
    selected = top_half_training_mask(samples)
    policy = torch.tensor([sample.policy_eligible for sample in samples], dtype=torch.bool)
    reference = torch.tensor(
        [sample.behavior_cloning_eligible for sample in samples], dtype=torch.bool
    )
    return old, advantages, policy & selected, reference & selected


def run_rlhf(
    *,
    adapter: str,
    execution: Path,
    output: Path,
    learning_rate: float,
    epochs: int,
    clip_epsilon: float,
    hf_repo: str | None = None,
) -> dict[str, Any]:
    samples = load_execution_group(execution)
    require_on_policy_logprobs(samples)
    model, processor = load_vision_model(
        adapter,
        LoRASettings(),
        adapter=adapter,
        max_sequence_length=2048,
    )
    collator = UnslothVisionDataCollator(model, processor)
    batch = collator([_conversation(sample) for sample in samples])
    labels = batch["labels"]
    old, advantages, policy, reference = align_rollout_metadata(
        samples, batch["input_ids"], labels
    )
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    history: list[dict[str, float]] = []
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi() if hf_repo else None
    if api is not None:
        api.create_repo(hf_repo, repo_type="model", private=False, exist_ok=True)
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = model(**inputs)
        current, response_mask = token_logprobs(outputs.logits, inputs["labels"])
        visual_tokens = current.shape[1] - old.shape[1]
        aligned_old = torch.nn.functional.pad(old, (visual_tokens, 0))
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
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 0.3)
        optimizer.step()
        metrics = {
            "epoch": float(epoch + 1),
            "loss": float(result.total.detach()),
            "policy_loss": float(result.relative_advantage.detach()),
            "bc_loss": float(result.behavior_cloning.detach()),
            "approximate_kl": float(result.approximate_kl.detach()),
            "clip_fraction": float(result.clip_fraction.detach()),
            "selected_count": float((policy | reference).sum()),
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        epoch_directory = output / f"epoch-{epoch + 1:03d}"
        model.save_pretrained(str(epoch_directory))
        processor.save_pretrained(str(epoch_directory))
        result_payload = {
            "adapter": adapter,
            "execution": str(execution),
            "hf_repo": hf_repo,
            "checkpoint": str(epoch_directory),
            "completed_epochs": epoch + 1,
            "history": history,
        }
        (output / "training_result.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if api is not None:
            api.upload_folder(
                repo_id=hf_repo,
                repo_type="model",
                folder_path=epoch_directory,
                path_in_repo=f"epoch-{epoch + 1:03d}",
                commit_message=f"Upload RLHF epoch {epoch + 1}/{epochs}",
            )
            api.upload_file(
                repo_id=hf_repo,
                repo_type="model",
                path_or_fileobj=output / "training_result.json",
                path_in_repo="training_result.json",
                commit_message=f"Update metrics through RLHF epoch {epoch + 1}/{epochs}",
            )
    model.save_pretrained(str(output / "lora_adapter"))
    processor.save_pretrained(str(output / "lora_adapter"))
    result_payload = {
        "adapter": adapter,
        "execution": str(execution),
        "hf_repo": hf_repo,
        "completed_epochs": epochs,
        "history": history,
    }
    (output / "training_result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="在严格 2+6 真实 rollout 上训练 Gemma 4 LoRA")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--hf-repo",
        help="公开 HF 模型仓库；每个 epoch 保存后立即上传独立 LoRA 检查点",
    )
    arguments = parser.parse_args()
    run_rlhf(
        adapter=arguments.adapter,
        execution=arguments.execution,
        output=arguments.output_dir,
        learning_rate=arguments.learning_rate,
        epochs=arguments.epochs,
        clip_epsilon=arguments.clip_epsilon,
        hf_repo=arguments.hf_repo,
    )


if __name__ == "__main__":
    main()
