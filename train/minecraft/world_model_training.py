"""完整下载 MineStudio 数据并无限训练时空快塔和 Dreamer-lite 世界模型。"""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from dataclasses import asdict, dataclass
import gc
import itertools
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterator

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, default_collate
from transformers import AutoModel, AutoTokenizer

from datasets.minestudio.groups import (
    MINESTUDIO_DATASET_GROUPS,
    get_dataset_group,
)
from datasets.minestudio.dataset import MineStudioLMDBDataset
from datasets.minestudio.download import prepare_dataset_group
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
    DEGREES_PER_MOUSE_PIXEL,
    encode_targets,
    structured_action_loss,
)

CHECKPOINT_VERSION = "minecraft_dreamer_lite_v6"
DEFAULT_VISION_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class TrainingLosses:
    """一次批次计算得到的联合训练损失。"""

    total: torch.Tensor
    action: torch.Tensor
    latent: torch.Tensor
    kl: torch.Tensor


class FrozenFeatureEncoders:
    """运行冻结的 DINOv3-S 和文本编码器，不进入 checkpoint 优化参数。"""

    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        device: torch.device,
        cache_directory: str | Path | None,
    ):
        self.device = device
        self.vision = AutoModel.from_pretrained(
            vision_model_name, torch_dtype=torch.bfloat16,
            cache_dir=cache_directory, token=True,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_name, cache_dir=cache_directory, token=True,
        )
        self.text = AutoModel.from_pretrained(
            text_model_name, torch_dtype=torch.bfloat16,
            cache_dir=cache_directory, token=True,
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
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

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
        if texts and len(set(texts)) == 1:
            task_text = texts[0]
            cached = self._text_cache.get(task_text)
            if cached is None:
                cached = self._encode_text_batch([task_text], maximum_tokens)
                self._text_cache[task_text] = cached
            tokens, mask = cached
            return tokens.expand(len(texts), -1, -1), mask.expand(len(texts), -1)
        return self._encode_text_batch(texts, maximum_tokens)

    def _encode_text_batch(
        self,
        texts: list[str],
        maximum_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """编码一个不可变文本批次，供同任务批次缓存复用。"""
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


def _batch_losses(
    batch: dict[str, object],
    encoders: FrozenFeatureEncoders,
    tower: torch.nn.Module,
    world_model: torch.nn.Module,
    tower_configuration: SpatiotemporalFastTowerConfiguration,
    history: int,
    action_horizon: int,
    world_weight: float,
    kl_weight: float,
    device: torch.device,
) -> TrainingLosses:
    """计算一个 MineStudio 批次的结构化 BC 与潜动力学损失。"""
    images = batch["img"]
    actions = batch["act_agg"]
    dt_frames = batch["dt"]
    task_text = batch["task_text"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("batch['img'] 必须为 Tensor")
    if not isinstance(actions, torch.Tensor) or not isinstance(dt_frames, torch.Tensor):
        raise TypeError("batch 动作与 dt 必须为 Tensor")
    if not isinstance(task_text, (list, tuple)):
        raise TypeError("batch['task_text'] 必须为文本序列")
    patches = encoders.encode_images(images, tower_configuration.grid_hw)
    text_tokens, text_mask = encoders.encode_text(
        list(task_text), tower_configuration.max_text_tokens,
    )
    past_actions, past_dt, camera_bins, keys, target_actions = _prepare_context(
        actions, dt_frames, history, action_horizon, device,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        action_output, _ = tower.forward_with_state(
            current_patches=patches[:, history],
            history_patches=patches[:, :history],
            text_tokens=text_tokens,
            text_mask=text_mask,
            past_actions=past_actions,
            dt=past_dt,
        )
        action_loss = structured_action_loss(action_output, camera_bins, keys)
        observation = patches[
            :, history:history + action_horizon + 1
        ].float().mean(dim=2)
        future_dt = (
            dt_frames[:, history:history + action_horizon]
            .to(device, non_blocking=True)[:, :, None] / 20.0
        )
        latent_loss, kl_loss = _world_model_loss(
            world_model, observation, target_actions, future_dt,
        )
        total = action_loss + world_weight * latent_loss + kl_weight * kl_loss
    return TrainingLosses(total, action_loss, latent_loss, kl_loss)


def _data_loader(
    dataset: MineStudioLMDBDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    prefetch_factor: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """构造适合本地 LMDB 和 CUDA pin-memory 的 DataLoader。"""
    arguments: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "generator": generator,
    }
    if workers > 0:
        arguments["prefetch_factor"] = prefetch_factor
    return DataLoader(**arguments)


def _cycle_batches(loader: DataLoader) -> Iterator[dict[str, object]]:
    """无限轮转 DataLoader，让 CUDA 预取跨 epoch 保持连续。"""
    while True:
        yield from loader


class _CUDABatchPrefetcher:
    """用独立 CUDA stream 把下一批张量提前搬入显存。"""

    def __init__(
        self,
        batches: Iterator[dict[str, object]],
        device: torch.device,
    ):
        self.batches = batches
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch: dict[str, object] | None = None
        self._preload()

    def _preload(self) -> None:
        batch = next(self.batches)
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                name: (
                    value.to(self.device, non_blocking=True)
                    if isinstance(value, torch.Tensor) else value
                )
                for name, value in batch.items()
            }

    def __iter__(self) -> "_CUDABatchPrefetcher":
        return self

    def __next__(self) -> dict[str, object]:
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch


@torch.no_grad()
def _evaluate(
    loader: DataLoader,
    maximum_batches: int,
    encoders: FrozenFeatureEncoders,
    tower: torch.nn.Module,
    world_model: torch.nn.Module,
    tower_configuration: SpatiotemporalFastTowerConfiguration,
    history: int,
    action_horizon: int,
    world_weight: float,
    kl_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """在当前图像分片的 episode 级留出集上计算平均损失。"""
    tower.eval()
    world_model.eval()
    totals = {"total": 0.0, "action": 0.0, "latent": 0.0, "kl": 0.0}
    batches = 0
    for batch in loader:
        losses = _batch_losses(
            batch, encoders, tower, world_model, tower_configuration,
            history, action_horizon, world_weight, kl_weight, device,
        )
        for name in totals:
            totals[name] += float(getattr(losses, name))
        batches += 1
        if batches >= maximum_batches:
            break
    if batches == 0:
        raise RuntimeError("验证 DataLoader 没有可评估批次")
    return {name: value / batches for name, value in totals.items()}


def _save_checkpoint(
    path: Path,
    tower: torch.nn.Module,
    world_model: torch.nn.Module,
    tower_configuration: SpatiotemporalFastTowerConfiguration,
    world_configuration: LatentWorldModelConfiguration,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset_group: str,
    image_shards: tuple[str, ...],
) -> None:
    """原子保存显式版本化的训练状态。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "tower": tower.state_dict(),
        "world_model": world_model.state_dict(),
        "tower_configuration": asdict(tower_configuration),
        "world_model_configuration": asdict(world_configuration),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "dataset_group": dataset_group,
        "image_shards": image_shards,
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    checkpoint_status = path.stat()
    metadata_path = path.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(json.dumps({
        "version": CHECKPOINT_VERSION,
        "step": step,
        "dataset_group": dataset_group,
        "image_shards": list(image_shards),
        "checkpoint_size": checkpoint_status.st_size,
        "checkpoint_modified_ns": checkpoint_status.st_mtime_ns,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_temporary.replace(metadata_path)


class PublicHubCheckpointUploader:
    """把本地原子 checkpoint 快照异步提交到公开 Hugging Face 模型仓库。"""

    def __init__(self, repository_id: str, output_directory: Path):
        self.repository_id = repository_id
        self.output_directory = output_directory
        self.snapshot_root = output_directory / ".hub_uploads"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.api = HfApi(token=True)
        self.api.create_repo(
            repo_id=repository_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        repository = self.api.repo_info(
            repo_id=repository_id, repo_type="model", token=True,
        )
        if repository.private:
            raise RuntimeError(
                f"Hugging Face 仓库 {repository_id} 已存在但仍是私有仓库；"
                "请改用新的公开仓库或先手动公开它",
            )
        self.futures: list[Future[object]] = []

    def publish(self, checkpoint_path: Path, step: int) -> None:
        """创建不随后续保存变化的硬链接，并在后台上传为远端 last checkpoint。"""
        self._raise_finished_failures()
        snapshot_directory = self.snapshot_root / f"step-{step}-{time.time_ns()}"
        snapshot_directory.mkdir()
        metadata_path = checkpoint_path.with_suffix(".json")
        os.link(checkpoint_path, snapshot_directory / "last.pt")
        os.link(metadata_path, snapshot_directory / "last.json")
        future = self.api.upload_folder(
            repo_id=self.repository_id,
            repo_type="model",
            folder_path=snapshot_directory,
            path_in_repo="",
            commit_message=f"上传训练 checkpoint step {step}",
            token=True,
            run_as_future=True,
        )
        future.add_done_callback(
            lambda _completed, directory=snapshot_directory: shutil.rmtree(
                directory, ignore_errors=True,
            ),
        )
        self.futures.append(future)

    def close(self) -> None:
        """等待已排队上传结束，并传播后台上传错误。"""
        for future in self.futures:
            future.result()
        shutil.rmtree(self.snapshot_root, ignore_errors=True)
        self.futures.clear()

    def _raise_finished_failures(self) -> None:
        unfinished: list[Future[object]] = []
        for future in self.futures:
            if future.done():
                future.result()
            else:
                unfinished.append(future)
        self.futures = unfinished


def _batch_tensor_bytes(batch: dict[str, object]) -> int:
    """计算一个 CPU 批次中张量的总字节数。"""
    return sum(
        value.numel() * value.element_size()
        for value in batch.values()
        if isinstance(value, torch.Tensor)
    )


def _measure_training_batch(
    batch: dict[str, object],
    encoders: FrozenFeatureEncoders,
    tower: torch.nn.Module,
    world_model: torch.nn.Module,
    tower_configuration: SpatiotemporalFastTowerConfiguration,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    arguments: argparse.Namespace,
    device: torch.device,
    initialize_optimizer: bool,
) -> int:
    """执行真实前反向并返回含下一批预取余量的峰值显存字节数。"""
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    losses = _batch_losses(
        batch, encoders, tower, world_model, tower_configuration,
        arguments.history, arguments.action_horizon,
        arguments.world_weight, arguments.kl_weight, device,
    )
    losses.total.backward()
    if initialize_optimizer:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_reserved(device)
    optimizer.zero_grad(set_to_none=True)
    return peak_bytes + _batch_tensor_bytes(batch)


def _automatic_batch_size(
    dataset: MineStudioLMDBDataset,
    encoders: FrozenFeatureEncoders,
    tower: torch.nn.Module,
    world_model: torch.nn.Module,
    tower_configuration: SpatiotemporalFastTowerConfiguration,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    arguments: argparse.Namespace,
    device: torch.device,
    optimizer_is_initialized: bool,
) -> tuple[int, int, int]:
    """实测并返回不超过目标显存比例的最大 batch、估计峰值和校准 step 数。"""
    target_bytes = int(
        torch.cuda.get_device_properties(device).total_memory
        * arguments.target_memory_fraction
    )
    maximum_batch_size = min(arguments.maximum_auto_batch, len(dataset))
    sample = dataset[0]
    measurements: dict[int, int | None] = {}
    calibration_steps = 0

    def measure(batch_size: int, initialize_optimizer: bool = False) -> int | None:
        if batch_size in measurements:
            return measurements[batch_size]
        try:
            batch = default_collate([sample] * batch_size)
            peak_bytes = _measure_training_batch(
                batch, encoders, tower, world_model, tower_configuration,
                optimizer, trainable, arguments, device, initialize_optimizer,
            )
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            peak_bytes = None
        measurements[batch_size] = peak_bytes
        return peak_bytes

    first_peak = measure(1, initialize_optimizer=not optimizer_is_initialized)
    if not optimizer_is_initialized:
        calibration_steps = 1
    if first_peak is None:
        raise RuntimeError("batch=1 仍然 CUDA OOM，当前模型无法在该 GPU 上训练")
    if first_peak > target_bytes:
        return 1, first_peak, calibration_steps

    lower = 1
    upper = 2
    while upper <= maximum_batch_size:
        peak = measure(upper)
        if peak is None or peak > target_bytes:
            break
        lower = upper
        upper *= 2
    upper = min(upper, maximum_batch_size)
    if lower == maximum_batch_size:
        return lower, int(measurements[lower]), calibration_steps
    while lower + 1 <= upper:
        candidate = (lower + upper + 1) // 2
        peak = measure(candidate)
        if peak is not None and peak <= target_bytes:
            lower = candidate
        else:
            upper = candidate - 1
    return lower, int(measurements[lower]), calibration_steps


def main() -> None:
    """完整下载指定 MineStudio 范围后无限联合训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="runs/data/minestudio")
    parser.add_argument(
        "--dataset-group",
        choices=[configuration.dataset_group for configuration in MINESTUDIO_DATASET_GROUPS],
        default="10xx",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=["image", "action"],
        choices=("action", "meta_info", "image", "event", "motion", "segmentation"),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--cache-directory", default=None,
        help="DINOv3、文本模型与 Hugging Face 下载缓存目录",
    )
    parser.add_argument("--output", default="runs/checkpoints/minecraft_dreamer_lite")
    parser.add_argument(
        "--hub-repo-id", required=True,
        help="持续上传 last.pt 的公开 Hugging Face 模型仓库，例如 user/model",
    )
    parser.add_argument(
        "--resume", default="auto",
        help="auto 自动恢复 output/last.pt；空串从头训练；也可给定 checkpoint",
    )
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--target-memory-fraction", type=float, default=0.75,
        help="自动 batch 的目标峰值显存比例",
    )
    parser.add_argument("--maximum-auto-batch", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=0,
                        help="0 表示使用不重叠窗口")
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--world-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--fused-optimizer", action=argparse.BooleanOptionalAction, default=True,
        help="CUDA 上使用 fused AdamW；排查兼容性时可传 --no-fused-optimizer",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small", action="store_true",
                        help="结构与损失冒烟用小模型；checkpoint 与正式模型不兼容")
    arguments = parser.parse_args()
    required_modalities = {"image", "action"}
    if not required_modalities.issubset(arguments.modalities):
        raise ValueError("无限训练必须同时下载 image 和 action")
    if len(set(arguments.modalities)) != len(arguments.modalities):
        raise ValueError("modalities 不能重复")
    if not torch.cuda.is_available():
        raise RuntimeError("该训练入口需要 CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16")
    if arguments.history < 1 or arguments.action_horizon < 1:
        raise ValueError("history 和 action_horizon 必须大于零")
    if (
        arguments.workers < 0 or arguments.prefetch_factor < 1
        or arguments.download_workers < 1 or arguments.maximum_auto_batch < 1
    ):
        raise ValueError("worker、预取和自动 batch 上限参数非法")
    if not 0.0 < arguments.target_memory_fraction < 1.0:
        raise ValueError("target-memory-fraction 必须位于 (0,1)")
    if arguments.learning_rate <= 0.0:
        raise ValueError("learning-rate 必须大于零")
    if arguments.world_weight < 0.0 or arguments.kl_weight < 0.0:
        raise ValueError("损失权重不能为负")
    if arguments.save_every < 1 or arguments.log_every < 1:
        raise ValueError("save-every 和 log-every 必须大于零")
    if arguments.validate_every < 1 or arguments.validation_batches < 1:
        raise ValueError("验证间隔和批次数必须大于零")

    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dataset_configuration = get_dataset_group(arguments.dataset_group)
    data_directory, download_selection = prepare_dataset_group(
        dataset_group=dataset_configuration.dataset_group,
        data_root=arguments.data_root,
        modalities=tuple(arguments.modalities),
        maximum_workers=arguments.download_workers,
        revision=arguments.revision,
        cache_directory=arguments.cache_directory,
    )
    print(json.dumps({
        "event": "dataset_ready",
        "dataset_group": dataset_configuration.dataset_group,
        "modalities": list(download_selection.modalities),
        "image_shards": list(download_selection.image_shards),
        "data_directory": str(data_directory),
    }, ensure_ascii=False), flush=True)
    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_uploader = PublicHubCheckpointUploader(
        arguments.hub_repo_id, output_directory,
    )
    encoders = FrozenFeatureEncoders(
        arguments.vision_model, arguments.text_model, device,
        arguments.cache_directory,
    )
    tower_configuration, world_configuration = _configurations(
        encoders, arguments.image_height, arguments.image_width,
        arguments.action_horizon, arguments.small,
    )
    tower = build_spatiotemporal_fast_tower(tower_configuration).to(device)
    world_model = build_latent_world_model(world_configuration).to(device)
    trainable = list(tower.parameters()) + list(world_model.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=arguments.learning_rate, fused=arguments.fused_optimizer,
    )
    resume_path = (
        output_directory / "last.pt"
        if arguments.resume == "auto" else Path(arguments.resume)
    )
    if arguments.resume not in {"", "auto"} and not resume_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {resume_path}")
    resume = bool(arguments.resume) and resume_path.is_file()
    first_step = 1
    if resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise RuntimeError("checkpoint 版本不兼容，拒绝静默部分加载")
        if checkpoint.get("tower_configuration") != asdict(tower_configuration):
            raise RuntimeError("checkpoint 快塔配置与本次配置不一致")
        if checkpoint.get("world_model_configuration") != asdict(world_configuration):
            raise RuntimeError("checkpoint 世界模型配置与本次配置不一致")
        if checkpoint.get("dataset_group") != dataset_configuration.dataset_group:
            raise RuntimeError("checkpoint 数据范围与本次 dataset-group 不一致")
        tower.load_state_dict(checkpoint["tower"], strict=True)
        world_model.load_state_dict(checkpoint["world_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = arguments.learning_rate
        first_step = int(checkpoint["step"]) + 1
    sequence_length = arguments.history + arguments.action_horizon + 1
    include_metadata_targets = "meta_info" in arguments.modalities
    dataset = MineStudioLMDBDataset(
        data_directory=data_directory,
        sequence_length=sequence_length,
        image_size=(arguments.image_height, arguments.image_width),
        task_text=dataset_configuration.task_text,
        camera_max_degrees=CAMERA_SCALE * DEGREES_PER_MOUSE_PIXEL,
        stride=arguments.window_stride or sequence_length,
        split="train",
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        include_metadata_targets=include_metadata_targets,
    )
    batch_size, peak_bytes, calibration_steps = _automatic_batch_size(
        dataset, encoders, tower, world_model, tower_configuration,
        optimizer, trainable, arguments, device, optimizer_is_initialized=resume,
    )
    first_step += calibration_steps
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(json.dumps({
        "event": "automatic_batch",
        "batch_size": batch_size,
        "estimated_peak_gib": round(peak_bytes / 1024**3, 2),
        "total_memory_gib": round(total_memory / 1024**3, 2),
        "estimated_fraction": round(peak_bytes / total_memory, 4),
        "target_fraction": arguments.target_memory_fraction,
    }), flush=True)
    data_generator = torch.Generator().manual_seed(arguments.seed)
    loader = _data_loader(
        dataset, batch_size, arguments.workers,
        shuffle=True, drop_last=True, prefetch_factor=arguments.prefetch_factor,
        generator=data_generator,
    )
    validation_loader = None
    if arguments.validation_fraction > 0.0:
        try:
            validation_dataset = MineStudioLMDBDataset(
                data_directory=data_directory,
                sequence_length=sequence_length,
                image_size=(arguments.image_height, arguments.image_width),
                task_text=dataset_configuration.task_text,
                camera_max_degrees=CAMERA_SCALE * DEGREES_PER_MOUSE_PIXEL,
                stride=arguments.window_stride or sequence_length,
                split="validation",
                validation_fraction=arguments.validation_fraction,
                seed=arguments.seed,
                include_metadata_targets=include_metadata_targets,
            )
            validation_loader = _data_loader(
                validation_dataset, batch_size, arguments.workers,
                shuffle=False, drop_last=False,
                prefetch_factor=arguments.prefetch_factor,
            )
        except RuntimeError as error:
            if "没有可用于该 split 的共同 episode" not in str(error):
                raise
            print(json.dumps({
                "event": "validation_disabled",
                "reason": str(error),
            }, ensure_ascii=False), flush=True)
    iterator = _CUDABatchPrefetcher(_cycle_batches(loader), device)
    tower_parameters = sum(parameter.numel() for parameter in tower.parameters())
    world_parameters = sum(parameter.numel() for parameter in world_model.parameters())
    print(
        f"dataset_group={dataset_configuration.dataset_group} "
        f"image_shards={','.join(dataset.image_shards)} "
        f"local_frames={dataset.total_frames} windows={len(dataset)} "
        f"tower={tower_parameters / 1e6:.1f}M "
        f"world_model={world_parameters / 1e6:.1f}M "
        f"batch={batch_size}",
        flush=True,
    )
    started = time.time()
    last_step = first_step - 1
    try:
        for step in itertools.count(first_step):
            tower.train()
            world_model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = next(iterator)
            losses = _batch_losses(
                batch, encoders, tower, world_model, tower_configuration,
                arguments.history, arguments.action_horizon,
                arguments.world_weight, arguments.kl_weight, device,
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            last_step = step
            if step % arguments.log_every == 0:
                elapsed = max(time.time() - started, 1e-4)
                print(json.dumps({
                    "split": "train",
                    "step": step,
                    "loss": round(float(losses.total.detach()), 5),
                    "action": round(float(losses.action.detach()), 5),
                    "latent": round(float(losses.latent.detach()), 5),
                    "kl": round(float(losses.kl.detach()), 5),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "steps_per_second": round((step - first_step + 1) / elapsed, 3),
                }), flush=True)
            if validation_loader is not None and step % arguments.validate_every == 0:
                validation = _evaluate(
                    validation_loader, arguments.validation_batches,
                    encoders, tower, world_model, tower_configuration,
                    arguments.history, arguments.action_horizon,
                    arguments.world_weight, arguments.kl_weight, device,
                )
                print(json.dumps({
                    "split": "validation",
                    "step": step,
                    **{name: round(value, 5) for name, value in validation.items()},
                }), flush=True)
            if step % arguments.save_every == 0:
                checkpoint_path = output_directory / "last.pt"
                _save_checkpoint(
                    checkpoint_path, tower, world_model,
                    tower_configuration, world_configuration, optimizer, step,
                    dataset_configuration.dataset_group, dataset.image_shards,
                )
                checkpoint_uploader.publish(checkpoint_path, step)
    except KeyboardInterrupt:
        if last_step >= first_step:
            checkpoint_path = output_directory / "last.pt"
            _save_checkpoint(
                checkpoint_path, tower, world_model,
                tower_configuration, world_configuration, optimizer, last_step,
                dataset_configuration.dataset_group,
                dataset.image_shards,
            )
            checkpoint_uploader.publish(checkpoint_path, last_step)
        print(json.dumps({"event": "training_interrupted", "step": last_step}), flush=True)
    finally:
        checkpoint_uploader.close()


if __name__ == "__main__":
    main()
