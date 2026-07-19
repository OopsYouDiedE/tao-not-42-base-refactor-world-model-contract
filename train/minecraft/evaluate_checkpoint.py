"""在固定 seed 的 CraftGround 环境中闭环评估 Minecraft 快塔 checkpoint。"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from craftground.screen_encoding_modes import ScreenEncodingMode

from datasets.minestudio.groups import get_dataset_group
from net.spatiotemporal_fast_tower import (
    SpatiotemporalFastTowerConfiguration,
    build_spatiotemporal_fast_tower,
)
from rl_training_environments.craftground.action_contract import (
    CAM_MAX_DEG,
    V2_KEYS,
)
from rl_training_environments.craftground.environment import (
    MinecraftCraftGroundEnvironment,
)
from train.minecraft.evaluation import deterministic_v2_action, wilson_interval
from train.minecraft.world_model_training import (
    CHECKPOINT_VERSION,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    FrozenFeatureEncoders,
)


def _image_tensor(image: np.ndarray | torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """把 CraftGround HWC 观测变为一个经过缩放的 ``[3,H,W]`` uint8 帧。"""
    frame = torch.as_tensor(image)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError("CraftGround RGB 观测必须为 [H,W,3]")
    frame = frame.permute(2, 0, 1).float()
    if float(frame.max()) <= 1.0:
        frame = frame * 255.0
    frame = F.interpolate(
        frame.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False,
    )[0]
    return frame.round().clamp(0.0, 255.0).byte().cpu()


def _canonical_action(action: dict[str, object]) -> torch.Tensor:
    """把完整 V2 动作转换为快塔历史动作 ``[22]``。"""
    return torch.tensor(
        [
            float(action["camera_yaw"]) / CAM_MAX_DEG,
            float(action["camera_pitch"]) / CAM_MAX_DEG,
            *[float(bool(action[key])) for key in V2_KEYS],
        ],
        dtype=torch.float32,
    )


@torch.inference_mode()
def _evaluate_seed(
    seed: int,
    maximum_steps: int,
    port: int,
    success_inventory_substring: str,
    history: int,
    image_size: tuple[int, int],
    task_text: str,
    encoders: FrozenFeatureEncoders,
    tower: torch.nn.Module,
    configuration: SpatiotemporalFastTowerConfiguration,
    device: torch.device,
    noop_policy: bool = False,
    policy_name: str = "checkpoint",
) -> dict[str, object]:
    """运行一个不自动重置的固定 seed episode。"""
    environment = MinecraftCraftGroundEnvironment(
        seed=seed,
        max_steps=maximum_steps,
        port=port,
        screen_encoding_mode=ScreenEncodingMode.RAW,
    )
    try:
        initial_image = environment.reset()
        actions = deque(
            [torch.zeros(configuration.action_dim) for _ in range(history)],
            maxlen=history,
        )
        time_deltas = deque([0.0 for _ in range(history)], maxlen=history)
        if not noop_policy:
            initial_frame = _image_tensor(initial_image, image_size)
            current_patches = encoders.encode_images(
                initial_frame[None, None], configuration.grid_hw,
            )[0, 0]
            patch_history = deque(
                [current_patches.clone() for _ in range(history)], maxlen=history,
            )
            text_tokens, text_mask = encoders.encode_text(
                [task_text], configuration.max_text_tokens,
            )
        total_reward = 0.0
        unlocked_achievements: set[int] = set()
        success = False
        completed_steps = 0
        for step in range(1, maximum_steps + 1):
            if noop_policy:
                action = {key: False for key in V2_KEYS}
                action.update({"camera_yaw": 0.0, "camera_pitch": 0.0})
            else:
                past_actions = torch.stack([
                    torch.zeros(configuration.action_dim), *actions,
                ]).unsqueeze(0).to(device)
                dt = torch.tensor(
                    [[0.0, *time_deltas]], dtype=torch.float32, device=device,
                ).unsqueeze(-1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = tower(
                        current_patches=current_patches[None],
                        history_patches=torch.stack([*patch_history])[None],
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                        past_actions=past_actions,
                        dt=dt,
                    )
                action = deterministic_v2_action(output)
            next_image, reward, done, information = environment.step_v2(action)
            total_reward += reward
            completed_steps = step
            unlocked_achievements.update(int(value) for value in information["successes"])
            inventory_keys = information["inventory_keys"]
            success = any(
                success_inventory_substring in key for key in inventory_keys
            )
            if success or done:
                break
            actions.append(_canonical_action(action))
            time_deltas.append(1.0 / 20.0)
            if not noop_policy:
                patch_history.append(current_patches)
                next_frame = _image_tensor(next_image, image_size)
                current_patches = encoders.encode_images(
                    next_frame[None, None], configuration.grid_hw,
                )[0, 0]
        return {
            "policy": "noop" if noop_policy else policy_name,
            "seed": seed,
            "success": success,
            "steps": completed_steps,
            "reward": total_reward,
            "achievements": len(unlocked_achievements),
        }
    finally:
        environment.close()


def main() -> None:
    """加载 checkpoint 并报告固定 seed 闭环成功率与置信区间。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--baseline-checkpoint", default=None,
        help="可选旧版或消融 checkpoint；将在相同 seed 上给出相对基线的区间判定",
    )
    parser.add_argument("--dataset-group", default="10xx", choices=("7xx", "9xx", "10xx"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--maximum-steps", type=int, default=12000)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--success-inventory-substring", default="diamond_pickaxe")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument(
        "--compare-noop", action=argparse.BooleanOptionalAction, default=True,
        help="在相同 seed 上额外运行 no-op 基线并给出保守有效性判定",
    )
    arguments = parser.parse_args()
    if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("seeds 必须非空且不能重复")
    if arguments.maximum_steps < 1 or arguments.history < 1:
        raise ValueError("maximum-steps 和 history 必须大于零")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("闭环评估需要支持 BF16 的 CUDA GPU")

    device = torch.device("cuda")
    checkpoint = torch.load(
        Path(arguments.checkpoint), map_location="cpu", weights_only=True,
    )
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError("checkpoint 版本不兼容")
    if checkpoint.get("dataset_group") != arguments.dataset_group:
        raise RuntimeError("checkpoint 的 dataset_group 与评估参数不一致")
    configuration = SpatiotemporalFastTowerConfiguration(
        **checkpoint["tower_configuration"],
    )
    if arguments.history > configuration.max_history - 1:
        raise ValueError("history 超过 checkpoint 快塔的历史容量")
    encoders = FrozenFeatureEncoders(
        arguments.vision_model, arguments.text_model, device,
        arguments.cache_directory,
    )
    expected_visual = configuration.visual_dim
    expected_text = configuration.text_dim
    if (encoders.visual_dim, encoders.text_dim) != (expected_visual, expected_text):
        raise RuntimeError("冻结编码器维度与 checkpoint 不一致")
    tower = build_spatiotemporal_fast_tower(configuration).to(device).eval()
    tower.load_state_dict(checkpoint["tower"], strict=True)
    baseline_tower = None
    if arguments.baseline_checkpoint:
        baseline_checkpoint = torch.load(
            Path(arguments.baseline_checkpoint), map_location="cpu", weights_only=True,
        )
        if baseline_checkpoint.get("version") != CHECKPOINT_VERSION:
            raise RuntimeError("baseline checkpoint 版本不兼容")
        if baseline_checkpoint.get("dataset_group") != arguments.dataset_group:
            raise RuntimeError("baseline checkpoint 的 dataset_group 不一致")
        if baseline_checkpoint.get("tower_configuration") != checkpoint.get(
            "tower_configuration",
        ):
            raise RuntimeError("baseline checkpoint 的快塔配置不一致")
        baseline_tower = build_spatiotemporal_fast_tower(configuration).to(device).eval()
        baseline_tower.load_state_dict(baseline_checkpoint["tower"], strict=True)
    patch_size = int(encoders.vision.config.patch_size)
    image_size = (
        configuration.grid_hw[0] * patch_size,
        configuration.grid_hw[1] * patch_size,
    )
    task_text = get_dataset_group(arguments.dataset_group).task_text
    results = []
    noop_results = []
    baseline_results = []
    for index, seed in enumerate(arguments.seeds):
        result = _evaluate_seed(
            seed, arguments.maximum_steps, arguments.base_port + index * 10,
            arguments.success_inventory_substring, arguments.history,
            image_size, task_text, encoders, tower, configuration, device,
        )
        results.append(result)
        print(json.dumps({"event": "closed_loop_episode", **result}), flush=True)
        if arguments.compare_noop:
            noop_result = _evaluate_seed(
                seed, arguments.maximum_steps,
                arguments.base_port + (len(arguments.seeds) + index) * 10,
                arguments.success_inventory_substring, arguments.history,
                image_size, task_text, encoders, tower, configuration, device,
                noop_policy=True,
            )
            noop_results.append(noop_result)
            print(json.dumps({
                "event": "closed_loop_episode", **noop_result,
            }), flush=True)
        if baseline_tower is not None:
            baseline_result = _evaluate_seed(
                seed, arguments.maximum_steps,
                arguments.base_port + (2 * len(arguments.seeds) + index) * 10,
                arguments.success_inventory_substring, arguments.history,
                image_size, task_text, encoders, baseline_tower,
                configuration, device, policy_name="baseline_checkpoint",
            )
            baseline_results.append(baseline_result)
            print(json.dumps({
                "event": "closed_loop_episode", **baseline_result,
            }), flush=True)
    successes = sum(int(result["success"]) for result in results)
    lower, upper = wilson_interval(successes, len(results))
    summary = {
        "event": "closed_loop_summary",
        "checkpoint_step": int(checkpoint["step"]),
        "dataset_group": arguments.dataset_group,
        "success_inventory_substring": arguments.success_inventory_substring,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
        "wilson_95_low": lower,
        "wilson_95_high": upper,
        "mean_reward": sum(float(result["reward"]) for result in results) / len(results),
        "mean_achievements": (
            sum(int(result["achievements"]) for result in results) / len(results)
        ),
    }
    if noop_results:
        noop_successes = sum(int(result["success"]) for result in noop_results)
        noop_lower, noop_upper = wilson_interval(noop_successes, len(noop_results))
        summary.update({
            "noop_successes": noop_successes,
            "noop_success_rate": noop_successes / len(noop_results),
            "noop_wilson_95_low": noop_lower,
            "noop_wilson_95_high": noop_upper,
            "effective_over_noop_95": lower > noop_upper,
        })
    if baseline_results:
        baseline_successes = sum(
            int(result["success"]) for result in baseline_results
        )
        baseline_lower, baseline_upper = wilson_interval(
            baseline_successes, len(baseline_results),
        )
        summary.update({
            "baseline_successes": baseline_successes,
            "baseline_success_rate": baseline_successes / len(baseline_results),
            "baseline_wilson_95_low": baseline_lower,
            "baseline_wilson_95_high": baseline_upper,
            "effective_over_baseline_95": lower > baseline_upper,
        })
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
