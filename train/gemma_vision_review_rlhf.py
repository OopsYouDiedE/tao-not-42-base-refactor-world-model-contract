"""四题型轨迹审核 2+6 rollout 与 clipped RLHF 入口。"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

import h5py
import torch

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import unsloth  # noqa: F401
from huggingface_hub import HfApi
from PIL import Image
from unsloth.trainer import UnslothVisionDataCollator

from train.gemma_vision_rlhf import align_rollout_metadata, token_logprobs
from train.objectives import clipped_token_joint_objective
from train.review_rlhf_contract import (
    ReviewCandidate,
    make_review_candidate,
    reference_review,
    relative_advantages,
    score_review,
)
from train.rollout_contract import RolloutSample, require_on_policy_logprobs
from train.unsloth_vision_sft import LoRASettings, load_vision_model

TASK_TYPES = (
    "demonstration_optimization",
    "image_sequence_to_action",
    "history_to_future_action",
    "single_frame_intent_to_action",
)


def review_prompt(question: dict[str, Any], candidate: ReviewCandidate) -> str:
    """构造不泄漏期望判定的严格审核提示词。"""
    schema = {
        "decision": "approve | revise | reject",
        "scores": {
            "visual_answerability": 1,
            "action_validity": 1,
            "duration_consistency": 1,
            "causal_consistency": 1,
            "gui_order": 1,
        },
        "reasons": ["concise verifiable reason"],
    }
    visible_candidate = {
        "task_type": candidate.answer.get("task_type", question.get("task_type")),
        "reference_action_sequence": candidate.answer["reference_action_sequence"],
    }
    return (
        "Review the proposed Minecraft action answer against the chronological images and the "
        "task. Check visual answerability, action syntax, exact duration requirements, causal "
        "consistency, unsupported keys or camera movement, and GUI click order. Scores are binary: "
        "1 means the criterion passes and 0 means it fails. Approve only when every material "
        "criterion passes. Return exactly one JSON object and no markdown.\n"
        f"Output schema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Question: {json.dumps(question, ensure_ascii=False)}\n"
        f"Proposed answer: {json.dumps(visible_candidate, ensure_ascii=False)}"
    )


def _load_four_examples(
    archive_path: Path,
) -> list[tuple[str, dict[str, Any], dict[str, Any], list[bytes]]]:
    found: dict[str, tuple[str, dict[str, Any], dict[str, Any], list[bytes]]] = {}
    with h5py.File(archive_path, "r") as archive:
        if archive.attrs.get("format") != "minestudio_trajectory_sft_v1":
            raise ValueError("不是受支持的 MineStudio 轨迹 SFT HDF5")
        for key, group in archive["samples"].items():
            question = json.loads(group.attrs["question_json"])
            task_type = question.get("task_type")
            if task_type not in found and task_type in TASK_TYPES:
                answer = json.loads(group.attrs["answer_json"])
                images = [dataset[()].tobytes() for dataset in group["images"].values()]
                found[task_type] = (key, question, answer, images)
    missing = set(TASK_TYPES) - set(found)
    if missing:
        raise ValueError(f"HDF5 缺少题型：{sorted(missing)}")
    return [found[task] for task in TASK_TYPES]


def _decode_images(image_paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())
    return images


@torch.inference_mode()
def _sample_reviews(
    model: Any,
    processor: Any,
    image_paths: list[Path],
    prompt: str,
    *,
    policy_version: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    images = _decode_images(image_paths)
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=[rendered], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()
    }
    parameters = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
    }
    results: list[dict[str, Any]] = []
    for _ in range(6):
        output = model.generate(
            **inputs, **parameters, return_dict_in_generate=True, output_scores=True
        )
        prompt_length = inputs["input_ids"].shape[1]
        token_ids = output.sequences[0, prompt_length:]
        if not output.scores or len(output.scores) != token_ids.numel():
            raise RuntimeError("生成 scores 与响应 token 未对齐")
        scores = torch.stack(tuple(score[0] for score in output.scores))
        old = scores.log_softmax(dim=-1).gather(1, token_ids[:, None]).squeeze(1)
        results.append(
            {
                "text": processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                "response_token_ids": [int(value) for value in token_ids.tolist()],
                "old_logprobs": [float(value) for value in old.tolist()],
                "policy_version": policy_version,
                "sampling_parameters": parameters,
            }
        )
    return results


def generate_rollouts(
    *,
    adapter: str,
    archive: Path,
    output: Path,
    max_new_tokens: int = 192,
) -> dict[str, Any]:
    """为四题型各生成一个 2 reference + 6 policy 审核组。"""
    output.mkdir(parents=True, exist_ok=True)
    model, processor = load_vision_model(
        adapter, LoRASettings(), adapter=adapter, max_sequence_length=4096
    )
    model.eval()
    groups: list[dict[str, Any]] = []
    for task_index, (sample_key, question, answer, image_bytes) in enumerate(
        _load_four_examples(archive)
    ):
        group_id = f"review-{question['task_type']}-{sample_key}"
        group_dir = output / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        image_paths: list[Path] = []
        for index, data in enumerate(image_bytes):
            path = group_dir / f"frame-{index:02d}.jpg"
            with Image.open(io.BytesIO(data)) as image:
                image.convert("RGB").save(path, quality=95)
            image_paths.append(path)
        candidate = make_review_candidate(answer, mutate=bool(task_index % 2))
        prompt = review_prompt(question, candidate)
        generated = _sample_reviews(
            model,
            processor,
            image_paths,
            prompt,
            policy_version=adapter,
            max_new_tokens=max_new_tokens,
        )
        trajectories: list[dict[str, Any]] = []
        for index in range(2):
            text = reference_review(candidate, wording=index)
            reward, audit = score_review(text, candidate)
            trajectories.append(
                {
                    "candidate_id": f"R{index + 1:02d}",
                    "source_role": "reference_expert",
                    "review_text": text,
                    "score": reward,
                    "audit": audit,
                }
            )
        for index, generation in enumerate(generated):
            reward, audit = score_review(generation["text"], candidate)
            trajectories.append(
                {
                    "candidate_id": f"P{index + 1:02d}",
                    "source_role": "policy_sample",
                    "review_text": generation.pop("text"),
                    "score": reward,
                    "audit": audit,
                    **generation,
                }
            )
        advantages = relative_advantages([item["score"] for item in trajectories])
        frames = [{"path": str(path.relative_to(output))} for path in image_paths]
        for item, advantage in zip(trajectories, advantages, strict=True):
            item["relative_advantage"] = advantage
            item["frames"] = frames
        groups.append(
            {
                "group_id": group_id,
                "task_type": question["task_type"],
                "question": question,
                "candidate_answer": candidate.answer,
                "candidate_origin": candidate.candidate_origin,
                "mutation_type": candidate.mutation_type,
                "expected_decision": candidate.expected_decision,
                "review_prompt": prompt,
                "trajectories": trajectories,
            }
        )
        print(json.dumps({"group_id": group_id, "generated": 6}, ensure_ascii=False), flush=True)
    payload = {"format": "minestudio_review_rlhf_v1", "adapter": adapter, "groups": groups}
    (output / "execution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _load_training_groups(execution: Path) -> list[tuple[dict[str, Any], list[RolloutSample]]]:
    payload = json.loads(execution.read_text(encoding="utf-8"))
    if payload.get("format") != "minestudio_review_rlhf_v1":
        raise ValueError("不是受支持的审核 rollout")
    result = []
    for group in payload["groups"]:
        samples = []
        for item in group["trajectories"]:
            samples.append(
                RolloutSample(
                    group_id=group["group_id"],
                    candidate_id=item["candidate_id"],
                    source_role=item["source_role"],
                    action_text=item["review_text"],
                    reward=float(item["score"]),
                    relative_advantage=float(item["relative_advantage"]),
                    image_paths=tuple(execution.parent / frame["path"] for frame in item["frames"]),
                    original_width=0,
                    original_height=0,
                    response_token_ids=tuple(item.get("response_token_ids", ())),
                    old_logprobs=tuple(item.get("old_logprobs", ())),
                    policy_version=item.get("policy_version"),
                    sampling_parameters=tuple(
                        sorted(
                            (str(key), json.dumps(value))
                            for key, value in item.get("sampling_parameters", {}).items()
                        )
                    ),
                )
            )
        require_on_policy_logprobs(samples)
        result.append((group, samples))
    return result


def _conversation(sample: RolloutSample, prompt: str) -> dict[str, Any]:
    images = _decode_images(list(sample.image_paths))
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": sample.action_text}]},
        ]
    }


def run_review_rlhf(
    *,
    adapter: str,
    execution: Path,
    output: Path,
    learning_rate: float,
    epochs: int,
    clip_epsilon: float,
    hf_repo: str | None,
) -> dict[str, Any]:
    """逐组训练四题型审核能力，并保存 LoRA 与审计指标。"""
    groups = _load_training_groups(execution)
    model, processor = load_vision_model(
        adapter, LoRASettings(), adapter=adapter, max_sequence_length=4096
    )
    collator = UnslothVisionDataCollator(model, processor)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    device = next(model.parameters()).device
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(epochs):
        for group, samples in groups:
            batch = collator([_conversation(sample, group["review_prompt"]) for sample in samples])
            old, advantages, policy, reference = align_rollout_metadata(
                samples, batch["input_ids"], batch["labels"]
            )
            inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            current, response_mask = token_logprobs(outputs.logits, inputs["labels"])
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
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 0.3)
            optimizer.step()
            audits = [
                item["audit"]
                for item in group["trajectories"]
                if item["source_role"] == "policy_sample"
            ]
            metrics = {
                "epoch": epoch + 1,
                "group_id": group["group_id"],
                "task_type": group["task_type"],
                "loss": float(result.total.detach()),
                "policy_loss": float(result.relative_advantage.detach()),
                "bc_loss": float(result.behavior_cloning.detach()),
                "approximate_kl": float(result.approximate_kl.detach()),
                "clip_fraction": float(result.clip_fraction.detach()),
                "decision_accuracy": sum(a["decision_correct"] for a in audits) / 6,
                "json_validity": sum(a["json_valid"] for a in audits) / 6,
                "false_approve": sum(a.get("false_approve", False) for a in audits),
                "false_reject": sum(a.get("false_reject", False) for a in audits),
            }
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    adapter_dir = output / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    result_payload = {
        "adapter": adapter,
        "execution": str(execution),
        "completed_epochs": epochs,
        "history": history,
        "hf_repo": hf_repo,
    }
    result_path = output / "training_result.json"
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if hf_repo:
        api = HfApi()
        api.create_repo(hf_repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=hf_repo,
            repo_type="model",
            folder_path=adapter_dir,
            commit_message="Upload reviewer RLHF LoRA",
        )
        api.upload_file(
            repo_id=hf_repo,
            repo_type="model",
            path_or_fileobj=result_path,
            path_in_repo="training_result.json",
            commit_message="Upload reviewer RLHF metrics",
        )
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="生成并训练四题型 MineStudio 审核 RLHF")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--adapter", required=True)
    rollout.add_argument("--archive", required=True, type=Path)
    rollout.add_argument("--output-dir", required=True, type=Path)
    rollout.add_argument("--max-new-tokens", type=int, default=192)
    train = subparsers.add_parser("train")
    train.add_argument("--adapter", required=True)
    train.add_argument("--execution", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--learning-rate", type=float, default=1e-6)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--clip-epsilon", type=float, default=0.2)
    train.add_argument("--hf-repo")
    arguments = parser.parse_args()
    if arguments.command == "rollout":
        generate_rollouts(
            adapter=arguments.adapter,
            archive=arguments.archive,
            output=arguments.output_dir,
            max_new_tokens=arguments.max_new_tokens,
        )
    else:
        run_review_rlhf(
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
