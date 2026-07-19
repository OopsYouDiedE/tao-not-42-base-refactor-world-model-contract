"""计算快塔动作指标、世界模型反事实指标与闭环置信区间。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from net.spatiotemporal_fast_tower import StructuredActionOutput
from rl_training_environments.craftground.action_contract import (
    CAM_MAX_DEG,
    V2_KEYS,
)
from train.minecraft.action_supervision import bin_centers


@dataclass
class ActionMetricAccumulator:
    """跨验证批次累计无类别频率偏置的结构化动作指标。"""

    camera_absolute_error: float = 0.0
    camera_values: int = 0
    exact_actions: int = 0
    noop_exact_actions: int = 0
    actions: int = 0
    true_positive: torch.Tensor = field(
        default_factory=lambda: torch.zeros(len(V2_KEYS), dtype=torch.float64),
    )
    false_positive: torch.Tensor = field(
        default_factory=lambda: torch.zeros(len(V2_KEYS), dtype=torch.float64),
    )
    false_negative: torch.Tensor = field(
        default_factory=lambda: torch.zeros(len(V2_KEYS), dtype=torch.float64),
    )

    def update(
        self,
        output: StructuredActionOutput,
        camera_bins: torch.Tensor,
        keys: torch.Tensor,
    ) -> None:
        """累计一个批次的相机误差、macro-F1、稀有动作 recall 与精确动作率。"""
        predicted_camera, predicted_keys = output.sample_legacy(deterministic=True)
        target_keys = keys.bool()
        predicted_keys = predicted_keys.bool()
        predicted_degrees = bin_centers(predicted_camera).float() * CAM_MAX_DEG
        target_degrees = bin_centers(camera_bins).float() * CAM_MAX_DEG
        valid = ~(
            (target_keys[..., 0] & target_keys[..., 1])
            | (target_keys[..., 2] & target_keys[..., 3])
            | (target_keys[..., 5] & target_keys[..., 6])
            | (target_keys[..., 11:20].sum(dim=-1) > 1)
        )
        self.camera_absolute_error += float(
            (predicted_degrees - target_degrees).abs()[valid].sum(),
        )
        self.camera_values += int(valid.sum()) * camera_bins.shape[-1]
        flattened_prediction = predicted_keys[valid].cpu()
        flattened_target = target_keys[valid].cpu()
        self.true_positive += (flattened_prediction & flattened_target).sum(dim=0)
        self.false_positive += (flattened_prediction & ~flattened_target).sum(dim=0)
        self.false_negative += (~flattened_prediction & flattened_target).sum(dim=0)
        camera_exact = (predicted_camera == camera_bins).all(dim=-1)
        key_exact = (predicted_keys == target_keys).all(dim=-1)
        self.exact_actions += int((camera_exact & key_exact & valid).sum())
        center = output.camera_logits.shape[-1] // 2
        self.noop_exact_actions += int(
            (
                ((camera_bins == center).all(dim=-1))
                & ~target_keys.any(dim=-1)
                & valid
            ).sum(),
        )
        self.actions += int(valid.sum())

    def compute(self) -> dict[str, float]:
        """返回可直接写入 JSON 日志的最终指标。"""
        f1_denominator = (
            2.0 * self.true_positive + self.false_positive + self.false_negative
        ).clamp(min=1e-4)
        macro_f1 = (2.0 * self.true_positive / f1_denominator).mean()
        rare_indices = torch.tensor([4, 7, 8, 9, 10], dtype=torch.long)
        rare_positive = self.true_positive[rare_indices].sum()
        rare_denominator = (
            rare_positive + self.false_negative[rare_indices].sum()
        ).clamp(min=1e-4)
        return {
            "camera_mae_degrees": self.camera_absolute_error / max(self.camera_values, 1),
            "key_macro_f1": float(macro_f1),
            "rare_button_recall": float(rare_positive / rare_denominator),
            "exact_action_accuracy": self.exact_actions / max(self.actions, 1),
            "noop_exact_action_accuracy": self.noop_exact_actions / max(self.actions, 1),
        }


@torch.no_grad()
def open_loop_latent_errors(
    world_model: torch.nn.Module,
    observation: torch.Tensor,
    action: torch.Tensor,
    dt_seconds: torch.Tensor,
) -> torch.Tensor:
    """从首帧后验开始纯想象，返回各未来步的 fp32 Smooth-L1 误差。"""
    state, _ = world_model.initialize(observation[:, 0])
    errors = []
    for horizon in range(action.shape[1]):
        prediction = world_model.imagine(
            state, action[:, horizon], dt_seconds[:, horizon],
        )
        errors.append(F.smooth_l1_loss(
            prediction.observation.float(),
            observation[:, horizon + 1].float(),
        ))
        state = prediction.next_state
    return torch.stack(errors)


def shuffled_actions(action: torch.Tensor) -> torch.Tensor:
    """确定性打乱批次动作；单样本时反转时间，避免反事实仍等于真实动作。"""
    if action.shape[0] > 1:
        return torch.roll(action, shifts=1, dims=0)
    return torch.flip(action, dims=(1,))


def wilson_interval(successes: int, episodes: int, z_score: float = 1.96) -> tuple[float, float]:
    """计算二项成功率 Wilson 95% 置信区间。"""
    if episodes < 1:
        raise ValueError("episodes 必须大于零")
    probability = successes / episodes
    denominator = 1.0 + z_score**2 / episodes
    center = (probability + z_score**2 / (2.0 * episodes)) / denominator
    radius = (
        z_score
        * math.sqrt(
            probability * (1.0 - probability) / episodes
            + z_score**2 / (4.0 * episodes**2),
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def deterministic_v2_action(output: StructuredActionOutput) -> dict[str, object]:
    """把批量为一的第一个动作头解码为完整 CraftGround V2 动作。"""
    camera, keys = output.sample_legacy(deterministic=True)
    camera_degrees = bin_centers(camera[0, 0]).float() * CAM_MAX_DEG
    key_values = keys[0, 0].bool().tolist()
    action = {key: bool(value) for key, value in zip(V2_KEYS, key_values)}
    action["camera_yaw"] = float(camera_degrees[0])
    action["camera_pitch"] = float(camera_degrees[1])
    return action
