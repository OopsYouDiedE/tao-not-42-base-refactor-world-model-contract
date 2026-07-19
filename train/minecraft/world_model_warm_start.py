"""以流式 VPT 数据联合暖启动时空快塔和 Dreamer-lite 世界模型。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from datasets.vpt.rolling_download import RollingVPTDownloadManager
from datasets.vpt.video_dataset import VPTStreamDataset
from net.latent_world_model import (
    LatentWorldModelConfiguration,
    balanced_categorical_kl_loss,
    build_latent_world_model,
)
from net.spatiotemporal_fast_tower import (
    SpatiotemporalFastTowerConfiguration,
    build_spatiotemporal_fast_tower,
)
from train.minecraft.action_supervision import (
    CAMERA_SCALE,
    encode_targets,
    structured_action_loss,
)

CHECKPOINT_VERSION = "minecraft_dreamer_lite_v1"
DEFAULT_VISION_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FrozenFeatureEncoders:
    """运行冻结的 DINOv3-S 和文本编码器，不进入 checkpoint 优化参数。"""

    def __init__(self, vision_model_name: str, text_model_name: str, device: torch.device):
        self.device = device
        self.vision = AutoModel.from_pretrained(
            vision_model_name, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text = AutoModel.from_pretrained(
            text_model_name, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        for model in (self.vision, self.text):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.visual_dim = int(self.vision.config.hidden_size)
        self.text_dim = int(self.text.config.hidden_size)
        mean = getattr(self.vision.config, "image_mean", [0.485, 0.456, 0.406])
        standard_deviation = getattr(
            self.vision.config, "image_std", [0.229, 0.224, 0.225],
        )
        self.image_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.image_standard_deviation = torch.tensor(
            standard_deviation, device=device,
        ).view(1, 3, 1, 1).clamp(min=1e-4)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """把 ``[B,T,3,H,W]`` uint8 图像编码为 ``[B,T,N,Dv]`` patch。"""
        batch_size, time_steps = images.shape[:2]
        pixels = images.flatten(0, 1).to(self.device, non_blocking=True).float() / 255.0
        pixels = (pixels - self.image_mean) / self.image_standard_deviation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.vision(pixel_values=pixels).last_hidden_state
        register_tokens = int(getattr(self.vision.config, "num_register_tokens", 0))
        patches = output[:, 1 + register_tokens:]
        expected = grid_hw[0] * grid_hw[1]
        if patches.shape[1] != expected:
            raise RuntimeError(
                f"DINO patch 数为 {patches.shape[1]}，但配置要求 {expected}; "
                "检查输入高宽与 patch_size",
            )
        return patches.reshape(batch_size, time_steps, expected, self.visual_dim)

    @torch.no_grad()
    def encode_text(self, texts: list[str], maximum_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """把任务文本编码为完整 token 和布尔有效位。"""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=maximum_tokens,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.text(**encoded).last_hidden_state
        return tokens, encoded["attention_mask"].bool()


def _configurations(
    encoders: FrozenFeatureEncoders,
    image_height: int,
    image_width: int,
    action_horizon: int,
    small: bool,
) -> tuple[SpatiotemporalFastTowerConfiguration, LatentWorldModelConfiguration]:
    patch_size = int(getattr(encoders.vision.config, "patch_size", 16))
    if image_height % patch_size or image_width % patch_size:
        raise ValueError("DINO 输入高宽必须能被 patch_size 整除")
    grid_hw = (image_height // patch_size, image_width // patch_size)
    if small:
        tower = SpatiotemporalFastTowerConfiguration(
            visual_dim=encoders.visual_dim, text_dim=encoders.text_dim,
            d=64, heads=4, spatial_layers=1, temporal_layers=1,
            grid_hw=grid_hw, action_horizon=action_horizon,
        )
        world = LatentWorldModelConfiguration(
            observation_dim=encoders.visual_dim, d=64,
            stochastic_variables=4, stochastic_classes=4, dynamics_layers=1,
        )
        return tower, world
    tower = SpatiotemporalFastTowerConfiguration(
        visual_dim=encoders.visual_dim,
        text_dim=encoders.text_dim,
        grid_hw=grid_hw,
        action_horizon=action_horizon,
    )
    world = LatentWorldModelConfiguration(observation_dim=encoders.visual_dim)
    return tower, world


def _prepare_context(
    action: torch.Tensor,
    dt_frames: torch.Tensor,
    history: int,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造无当前动作泄漏的历史输入和未来动作块监督。"""
    camera_bins, keys, canonical = encode_targets(action.to(device, non_blocking=True))
    zero_action = torch.zeros(canonical.shape[0], 1, canonical.shape[-1], device=device)
    zero_dt = torch.zeros(canonical.shape[0], 1, 1, device=device)
    past_actions = torch.cat([zero_action, canonical[:, :history]], dim=1)
    past_dt = torch.cat([zero_dt, dt_frames[:, :history, None].to(device) / 20.0], dim=1)
    target_slice = slice(history, history + horizon)
    return (
        past_actions,
        past_dt,
        camera_bins[:, target_slice],
        keys[:, target_slice],
        canonical[:, target_slice],
    )


def _world_model_loss(
    world_model: torch.nn.Module,
    observation: torch.Tensor,
    action: torch.Tensor,
    dt_seconds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在真实后验之间展开多步先验，返回 latent 和 KL 损失。"""
    state, _ = world_model.initialize(observation[:, 0].detach())
    latent_losses = []
    kl_losses = []
    for index in range(action.shape[1]):
        prediction = world_model.imagine(state, action[:, index], dt_seconds[:, index])
        target = observation[:, index + 1].detach()
        latent_losses.append(F.smooth_l1_loss(
            prediction.observation.float(), target.float(),
        ))
        state, posterior_logits = world_model.observe(prediction.next_state, target)
        kl_losses.append(balanced_categorical_kl_loss(
            posterior_logits, prediction.prior_logits,
        ))
    return torch.stack(latent_losses).mean(), torch.stack(kl_losses).mean()


def _save_checkpoint(
    path: Path,
    tower: torch.nn.Module,
    world_model: torch.nn.Module,
    tower_configuration: SpatiotemporalFastTowerConfiguration,
    world_configuration: LatentWorldModelConfiguration,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    """原子保存显式版本化的训练状态。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "version": CHECKPOINT_VERSION,
        "tower": tower.state_dict(),
        "world_model": world_model.state_dict(),
        "tower_configuration": asdict(tower_configuration),
        "world_model_configuration": asdict(world_configuration),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, temporary)
    temporary.replace(path)


def main() -> None:
    """运行联合行为克隆和潜动力学暖启动。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="runs/data/vpt_stream")
    parser.add_argument("--data-manifest", default="",
                        help="可选 JSON/JSONL 路径或 URL；启用有界滚动下载")
    parser.add_argument("--stream-cache-gib", type=float, default=80.0)
    parser.add_argument("--stream-ready-pairs", type=int, default=4)
    parser.add_argument("--output", default="runs/checkpoints/minecraft_dreamer_lite")
    parser.add_argument("--resume", default="", help="从同版本 last.pt 精确续训")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--world-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small", action="store_true",
                        help="结构与损失冒烟用小模型；checkpoint 与正式模型不兼容")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("该训练入口需要 CUDA；CPU 单测不代表 AutoDL 训练可运行")
    if arguments.history < 1 or arguments.action_horizon < 1:
        raise ValueError("history 和 action_horizon 必须大于零")

    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    data_directory = Path(arguments.data)
    data_directory.mkdir(parents=True, exist_ok=True)
    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    downloader = None
    if arguments.data_manifest:
        downloader = RollingVPTDownloadManager(
            arguments.data_manifest,
            data_directory,
            maximum_cache_bytes=int(arguments.stream_cache_gib * 1024 ** 3),
            minimum_ready_pairs=arguments.stream_ready_pairs,
            seed=arguments.seed,
        ).start()

    try:
        encoders = FrozenFeatureEncoders(
            arguments.vision_model, arguments.text_model, device,
        )
        tower_configuration, world_configuration = _configurations(
            encoders, arguments.image_height, arguments.image_width,
            arguments.action_horizon, arguments.small,
        )
        tower = build_spatiotemporal_fast_tower(tower_configuration).to(device)
        world_model = build_latent_world_model(world_configuration).to(device)
        trainable = list(tower.parameters()) + list(world_model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=arguments.learning_rate)
        first_step = 1
        if arguments.resume:
            checkpoint = torch.load(arguments.resume, map_location=device, weights_only=True)
            if checkpoint.get("version") != CHECKPOINT_VERSION:
                raise RuntimeError("checkpoint 版本不兼容，拒绝静默部分加载")
            if checkpoint.get("tower_configuration") != asdict(tower_configuration):
                raise RuntimeError("checkpoint 快塔配置与本次配置不一致")
            if checkpoint.get("world_model_configuration") != asdict(world_configuration):
                raise RuntimeError("checkpoint 世界模型配置与本次配置不一致")
            tower.load_state_dict(checkpoint["tower"], strict=True)
            world_model.load_state_dict(checkpoint["world_model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            first_step = int(checkpoint["step"]) + 1
        sequence_length = arguments.history + arguments.action_horizon + 1
        dataset = VPTStreamDataset(
            str(data_directory), seq_len=sequence_length,
            img_size=(arguments.image_height, arguments.image_width),
            camera_scale=CAMERA_SCALE, frame_skip=1, split=None,
            clip_cache=2, clip_refresh=128, seed=arguments.seed,
        )
        loader = DataLoader(
            dataset, batch_size=arguments.batch, num_workers=arguments.workers,
            pin_memory=True, persistent_workers=arguments.workers > 0,
            prefetch_factor=2 if arguments.workers else None,
        )
        iterator = iter(loader)
        tower_parameters = sum(parameter.numel() for parameter in tower.parameters())
        world_parameters = sum(parameter.numel() for parameter in world_model.parameters())
        print(
            f"tower={tower_parameters / 1e6:.1f}M "
            f"world_model={world_parameters / 1e6:.1f}M",
            flush=True,
        )
        started = time.time()
        for step in range(first_step, arguments.steps + 1):
            if downloader is not None:
                downloader.check()
            batch = next(iterator)
            patches = encoders.encode_images(
                batch["img"], tower_configuration.grid_hw,
            )
            text_tokens, text_mask = encoders.encode_text(
                list(batch["task_text"]), tower_configuration.max_text_tokens,
            )
            past_actions, past_dt, camera_bins, keys, target_actions = _prepare_context(
                batch["act_agg"], batch["dt"], arguments.history,
                arguments.action_horizon, device,
            )
            aim_xy = torch.zeros(arguments.batch, 2, device=device)
            aim_valid = torch.zeros(arguments.batch, dtype=torch.bool, device=device)
            tower.train()
            world_model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action_output, _ = tower.forward_with_state(
                    current_patches=patches[:, arguments.history],
                    history_patches=patches[:, :arguments.history],
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    past_actions=past_actions,
                    dt=past_dt,
                    aim_xy=aim_xy,
                    aim_valid=aim_valid,
                )
                action_loss = structured_action_loss(action_output, camera_bins, keys)
                observation = patches[
                    :, arguments.history:arguments.history + arguments.action_horizon + 1
                ].float().mean(dim=2)
                future_dt = (
                    batch["dt"][:, arguments.history:arguments.history + arguments.action_horizon]
                    .to(device)[:, :, None] / 20.0
                )
                latent_loss, kl_loss = _world_model_loss(
                    world_model, observation, target_actions, future_dt,
                )
                loss = (
                    action_loss
                    + arguments.world_weight * latent_loss
                    + arguments.kl_weight * kl_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if step % 20 == 0:
                elapsed = max(time.time() - started, 1e-4)
                print(json.dumps({
                    "step": step,
                    "loss": round(float(loss), 5),
                    "action": round(float(action_loss), 5),
                    "latent": round(float(latent_loss), 5),
                    "kl": round(float(kl_loss), 5),
                    "steps_per_second": round((step - first_step + 1) / elapsed, 3),
                }), flush=True)
            if step % arguments.save_every == 0 or step == arguments.steps:
                _save_checkpoint(
                    output_directory / "last.pt", tower, world_model,
                    tower_configuration, world_configuration, optimizer, step,
                )
    finally:
        if downloader is not None:
            downloader.stop()


if __name__ == "__main__":
    main()
