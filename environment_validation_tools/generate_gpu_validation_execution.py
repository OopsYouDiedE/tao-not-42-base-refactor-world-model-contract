"""使用真实本地视觉策略生成 GPU 训练验收用的 2+6 execution group。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from behavior_cloning_training import load_vision_model
from online_interactive_environments import parse_action_sequence_strict
from relative_advantage_comparison_training import PolicyGeneration, generate_policy_rollouts
from shared_tools import atomic_write_json

DEFAULT_ACTIONS = ("W", "NoOp", "A", "D")


def build_execution_payload(
    generations: list[PolicyGeneration], *, frame_name: str, reference_action_text: str
) -> dict[str, Any]:
    """构造一个用于验证训练链路的完整 2+6 execution group。"""
    if len(generations) != 6:
        raise ValueError("GPU validation requires exactly six policy generations")
    parse_action_sequence_strict(reference_action_text)
    trajectories: list[dict[str, Any]] = []
    for index in range(2):
        trajectories.append(
            {
                "candidate_id": f"reference-{index + 1}",
                "source_role": "reference_expert",
                "action_text": reference_action_text,
                "score": 2.0,
                "relative_advantage": 0.0,
                "frames": [{"path": frame_name}],
            }
        )
    for index, generation in enumerate(generations):
        trajectories.append(
            {
                "candidate_id": f"policy-{index + 1}",
                "source_role": "policy_sample",
                "action_text": generation.action_text,
                "score": 1.0,
                "relative_advantage": 0.5,
                "frames": [{"path": frame_name}],
                "response_token_ids": list(generation.response_token_ids),
                "old_logprobs": list(generation.old_logprobs),
                "policy_version": generation.policy_version,
                "sampling_parameters": generation.sampling_parameters,
            }
        )
    return {"snapshot_id": "gpu-validation-snapshot", "trajectories": trajectories}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 GPU 验收用的 2+6 策略 execution group")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-version", default="gpu-validation-bc")
    parser.add_argument("--action", action="append", dest="actions")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--load-in-4bit", action="store_true")
    arguments = parser.parse_args()
    actions = tuple(arguments.actions or DEFAULT_ACTIONS)
    allowed = tuple(
        f"Device KeyboardMouse\nTick 0\n<action>{action}</action>" for action in actions
    )
    model, processor = load_vision_model(
        arguments.model,
        adapter=arguments.adapter,
        load_in_4bit=arguments.load_in_4bit,
        max_sequence_length=512,
    )
    generations = generate_policy_rollouts(
        model,
        processor,
        [arguments.image],
        intent=arguments.intent,
        policy_version=arguments.policy_version,
        max_new_tokens=arguments.max_new_tokens,
        allowed_action_texts=allowed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame = arguments.output.parent / "observation.png"
    shutil.copy2(arguments.image, frame)
    payload = build_execution_payload(
        generations, frame_name=frame.name, reference_action_text=allowed[0]
    )
    atomic_write_json(arguments.output, payload)
    print(arguments.output)


if __name__ == "__main__":
    main()
