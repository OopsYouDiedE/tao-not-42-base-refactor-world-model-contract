"""在固定 seed 的 CraftGround 环境中闭环评估 Qwen3VL 动作策略 checkpoint。"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from craftground.screen_encoding_modes import ScreenEncodingMode
from PIL import Image

from datasets.minestudio.groups import get_dataset_group
from net.action_token_codec import ActionTokenFormat
from net.qwen3vl_policy import (
    HistoryContext,
    Qwen3VLPolicyConfiguration,
    build_qwen3vl_action_policy,
)
from rl_training_environments.craftground.action_contract import V2_KEYS
from rl_training_environments.craftground.environment import (
    MinecraftCraftGroundEnvironment,
)
from train.minecraft.evaluation import structured_to_v2_action, wilson_interval
from train.minecraft.world_model_training import CHECKPOINT_VERSION, _load_adapter
from train.minecraft.world_model_training import _CheckpointConfiguration


def _image(image: np.ndarray | torch.Tensor, size: tuple[int, int]) -> Image.Image:
    """把 CraftGround HWC 观测转为缩放后的 PIL 图像。"""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError("CraftGround RGB 观测必须为 [H,W,3]")
    if array.dtype != np.uint8:
        array = np.clip(array if array.max() > 1.0 else array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).resize((size[1], size[0]))


@torch.inference_mode()
def _evaluate_seed(
    seed: int,
    maximum_steps: int,
    port: int,
    success_inventory_substring: str,
    history_frames: int,
    image_size: tuple[int, int],
    task_text: str,
    policy,
    device: torch.device,
    noop_policy: bool = False,
    policy_name: str = "checkpoint",
) -> dict[str, object]:
    """运行一个不自动重置的固定 seed episode。策略一次生成动作块，逐步执行。"""
    environment = MinecraftCraftGroundEnvironment(
        seed=seed, max_steps=maximum_steps, port=port,
        screen_encoding_mode=ScreenEncodingMode.RAW,
    )
    try:
        initial_image = environment.reset()
        frame_history: deque[Image.Image] = deque(
            [_image(initial_image, image_size)] * history_frames, maxlen=history_frames,
        )
        planned: list = []
        total_reward = 0.0
        unlocked: set[int] = set()
        success = False
        completed_steps = 0
        for step in range(1, maximum_steps + 1):
            if noop_policy:
                action = {key: False for key in V2_KEYS}
                action.update({"camera_yaw": 0.0, "camera_pitch": 0.0})
            else:
                if not planned:
                    context = HistoryContext(
                        frames=list(frame_history), task_text=task_text, past_actions=[],
                    )
                    actions, _text = policy.generate_actions(
                        context, include_past_actions=False, temperature=0.0,
                    )
                    planned = list(actions)
                action = structured_to_v2_action(planned.pop(0))
            next_image, reward, done, information = environment.step_v2(action)
            total_reward += reward
            completed_steps = step
            unlocked.update(int(value) for value in information["successes"])
            success = any(
                success_inventory_substring in key for key in information["inventory_keys"]
            )
            if success or done:
                break
            if not noop_policy:
                frame_history.append(_image(next_image, image_size))
        return {
            "policy": "noop" if noop_policy else policy_name,
            "seed": seed,
            "success": success,
            "steps": completed_steps,
            "reward": total_reward,
            "achievements": len(unlocked),
        }
    finally:
        environment.close()


def main() -> None:
    """加载 checkpoint 并报告固定 seed 闭环成功率与置信区间。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-group", default="10xx", choices=("7xx", "9xx", "10xx"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--maximum-steps", type=int, default=12000)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--success-inventory-substring", default="diamond_pickaxe")
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--image-height", type=int, default=252)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument(
        "--compare-noop", action=argparse.BooleanOptionalAction, default=True,
    )
    arguments = parser.parse_args()
    if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("seeds 必须非空且不能重复")
    if arguments.maximum_steps < 1:
        raise ValueError("maximum-steps 必须大于零")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("闭环评估需要支持 BF16 的 CUDA GPU")

    device = torch.device("cuda")
    metadata = json.loads(
        Path(arguments.checkpoint).with_suffix(".json").read_text(encoding="utf-8"),
    )
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError("checkpoint 版本不兼容")
    if metadata.get("dataset_group") != arguments.dataset_group:
        raise RuntimeError("checkpoint 的 dataset_group 与评估参数不一致")
    configuration_dict = metadata["configuration"]
    checkpoint_configuration = _CheckpointConfiguration(**configuration_dict)
    policy_configuration = Qwen3VLPolicyConfiguration(
        model_name=checkpoint_configuration.model_name,
        action_format=ActionTokenFormat(checkpoint_configuration.action_format),
        action_horizon=checkpoint_configuration.action_horizon,
        max_history_frames=checkpoint_configuration.history_frames,
        max_new_tokens=max(64, 24 * checkpoint_configuration.action_horizon),
    )
    policy = build_qwen3vl_action_policy(
        policy_configuration, device, arguments.cache_directory,
        lora_rank=checkpoint_configuration.lora_rank,
    )
    _load_adapter(Path(arguments.checkpoint), policy, checkpoint_configuration)
    policy.model.eval()
    task_text = get_dataset_group(arguments.dataset_group).task_text
    image_size = (arguments.image_height, arguments.image_width)
    history_frames = checkpoint_configuration.history_frames

    results = []
    noop_results = []
    for index, seed in enumerate(arguments.seeds):
        result = _evaluate_seed(
            seed, arguments.maximum_steps, arguments.base_port + index * 10,
            arguments.success_inventory_substring, history_frames,
            image_size, task_text, policy, device,
        )
        results.append(result)
        print(json.dumps({"event": "closed_loop_episode", **result}), flush=True)
        if arguments.compare_noop:
            noop_result = _evaluate_seed(
                seed, arguments.maximum_steps,
                arguments.base_port + (len(arguments.seeds) + index) * 10,
                arguments.success_inventory_substring, history_frames,
                image_size, task_text, policy, device, noop_policy=True,
            )
            noop_results.append(noop_result)
            print(json.dumps({"event": "closed_loop_episode", **noop_result}), flush=True)
    successes = sum(int(result["success"]) for result in results)
    lower, upper = wilson_interval(successes, len(results))
    summary = {
        "event": "closed_loop_summary",
        "checkpoint_step": int(metadata["step"]),
        "dataset_group": arguments.dataset_group,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
        "wilson_95_low": lower,
        "wilson_95_high": upper,
        "mean_reward": sum(float(r["reward"]) for r in results) / len(results),
        "mean_achievements": sum(int(r["achievements"]) for r in results) / len(results),
    }
    if noop_results:
        noop_successes = sum(int(r["success"]) for r in noop_results)
        noop_lower, noop_upper = wilson_interval(noop_successes, len(noop_results))
        summary.update({
            "noop_success_rate": noop_successes / len(noop_results),
            "noop_wilson_95_low": noop_lower,
            "noop_wilson_95_high": noop_upper,
            "effective_over_noop_95": lower > noop_upper,
        })
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
