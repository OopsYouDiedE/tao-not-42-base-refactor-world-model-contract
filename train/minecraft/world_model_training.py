"""下载 MineStudio 数据并对 Qwen3VL 动作策略做无限行为克隆(SFT)训练。

以 Qwen3VL 为视觉大模型，用 LoRA 适配器把冻结主干微调成直接输出动作 token 的策略；
监督信号是数据集真实动作经 codec 编码的目标文本。保留自动 batch 探测、resume 与公开
Hugging Face 仓库异步上传的骨架。旧的快塔 + Dreamer-lite 世界模型路径已删除。
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterator

import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader

from datasets.minestudio.groups import (
    MINESTUDIO_DATASET_GROUPS,
    get_dataset_group,
)
from datasets.minestudio.dataset import MineStudioLMDBDataset
from datasets.minestudio.download import prepare_dataset_group
from net.action_token_codec import ActionTokenFormat
from net.qwen3vl_policy import (
    HistoryContext,
    Qwen3VLPolicyConfiguration,
    build_qwen3vl_action_policy,
)
from train.minecraft.action_supervision import (
    CAMERA_SCALE,
    DEGREES_PER_MOUSE_PIXEL,
    vpt_actions_to_structured,
)
from train.minecraft.evaluation import key_action_agreement_rate

CHECKPOINT_VERSION = "minecraft_qwen3vl_vla_v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def _frames_to_images(frames: torch.Tensor):
    """把 ``[T,3,H,W]`` uint8 帧转为 PIL 图像列表。"""
    from PIL import Image

    images = []
    for index in range(frames.shape[0]):
        array = frames[index].permute(1, 2, 0).contiguous().to(torch.uint8).cpu().numpy()
        images.append(Image.fromarray(array))
    return images


def _sample_to_context_and_target(
    sample: dict[str, object],
    history_frames: int,
    action_horizon: int,
    task_text: str,
):
    """把一个数据窗口拆成(历史上下文, 未来目标动作序列)。

    历史帧含当前帧共 ``history_frames`` 帧；过去动作对应除当前帧外的历史帧；目标是
    从当前帧起的未来 ``action_horizon`` 帧动作。
    """
    frames = sample["img"]
    actions = sample["act_agg"]
    history_images = _frames_to_images(frames[:history_frames])
    past = (
        vpt_actions_to_structured(actions[:history_frames - 1])
        if history_frames > 1 else []
    )
    future = vpt_actions_to_structured(
        actions[history_frames - 1:history_frames - 1 + action_horizon],
    )
    context = HistoryContext(frames=history_images, task_text=task_text, past_actions=past)
    return context, future


def _sft_loss(policy, sample, history_frames, action_horizon, task_text) -> torch.Tensor:
    """对一个数据窗口计算 SFT 交叉熵损失。"""
    context, future = _sample_to_context_and_target(
        sample, history_frames, action_horizon, task_text,
    )
    if len(future) != action_horizon:
        raise RuntimeError("窗口未来动作不足 action_horizon，检查 sequence_length")
    return policy.supervised_loss(context, future)


def _single_sample_collate(batch: list[dict]) -> dict:
    """DataLoader 只取单样本(逐窗口做 SFT，图像数随窗口变，无法张量化 batch)。"""
    if len(batch) != 1:
        raise RuntimeError("Qwen3VL SFT 训练按单窗口前向，batch_size 必须为 1")
    return batch[0]


def _data_loader(
    dataset: MineStudioLMDBDataset,
    workers: int,
    shuffle: bool,
    prefetch_factor: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """构造单窗口 DataLoader。"""
    arguments: dict[str, object] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": workers,
        "collate_fn": _single_sample_collate,
        "persistent_workers": workers > 0,
        "generator": generator,
    }
    if workers > 0:
        arguments["prefetch_factor"] = prefetch_factor
    return DataLoader(**arguments)


def _cycle_batches(loader: DataLoader) -> Iterator[dict]:
    """无限轮转 DataLoader。"""
    while True:
        yield from loader


@dataclass
class _CheckpointConfiguration:
    """写入 checkpoint 的策略与训练配置摘要，用于严格恢复校验。"""

    model_name: str
    action_format: str
    action_horizon: int
    history_frames: int
    lora_rank: int


def _save_checkpoint(
    path: Path,
    policy,
    checkpoint_configuration: _CheckpointConfiguration,
    step: int,
    dataset_group: str,
    image_shards: tuple[str, ...],
) -> None:
    """原子保存 LoRA 适配器权重与显式版本化配置。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 只保存需要梯度的适配器参数，冻结主干不进 checkpoint。
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in policy.model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "version": CHECKPOINT_VERSION,
        "adapter_state": trainable_state,
        "configuration": vars(checkpoint_configuration),
        "step": step,
        "dataset_group": dataset_group,
        "image_shards": list(image_shards),
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    status = path.stat()
    metadata_path = path.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(json.dumps({
        "version": CHECKPOINT_VERSION,
        "step": step,
        "dataset_group": dataset_group,
        "image_shards": list(image_shards),
        "configuration": vars(checkpoint_configuration),
        "checkpoint_size": status.st_size,
        "checkpoint_modified_ns": status.st_mtime_ns,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_temporary.replace(metadata_path)


def _load_adapter(path: Path, policy, checkpoint_configuration: _CheckpointConfiguration) -> int:
    """严格校验配置后加载 LoRA 适配器，返回已完成的 step。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError("checkpoint 版本不兼容，拒绝静默部分加载")
    if checkpoint.get("configuration") != vars(checkpoint_configuration):
        raise RuntimeError("checkpoint 策略配置与本次配置不一致")
    trainable = {
        name: parameter
        for name, parameter in policy.model.named_parameters()
        if parameter.requires_grad
    }
    adapter_state = checkpoint["adapter_state"]
    if set(adapter_state) != set(trainable):
        raise RuntimeError("checkpoint 适配器参数集合与当前模型不一致")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(adapter_state[name].to(parameter.device, parameter.dtype))
    return int(checkpoint["step"])


class PublicHubCheckpointUploader:
    """把本地原子 checkpoint 快照异步提交到公开 Hugging Face 模型仓库。"""

    def __init__(self, repository_id: str, output_directory: Path):
        self.repository_id = repository_id
        self.output_directory = output_directory
        self.snapshot_root = output_directory / ".hub_uploads"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.api = HfApi(token=True)
        self.api.create_repo(
            repo_id=repository_id, repo_type="model", private=False, exist_ok=True,
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
        """硬链接快照并在后台上传为远端 last checkpoint。"""
        self._raise_finished_failures()
        snapshot_directory = self.snapshot_root / f"step-{step}-{time.time_ns()}"
        snapshot_directory.mkdir()
        metadata_path = checkpoint_path.with_suffix(".json")
        os.link(checkpoint_path, snapshot_directory / "last.pt")
        os.link(metadata_path, snapshot_directory / "last.json")
        future = self.api.upload_folder(
            repo_id=self.repository_id, repo_type="model",
            folder_path=snapshot_directory, path_in_repo="",
            commit_message=f"上传训练 checkpoint step {step}",
            token=True, run_as_future=True,
        )
        future.add_done_callback(
            lambda _completed, directory=snapshot_directory: shutil.rmtree(
                directory, ignore_errors=True,
            ),
        )
        self.futures.append(future)

    def close(self) -> None:
        """等待已排队上传结束并传播错误。"""
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


@torch.inference_mode()
def _evaluate(
    loader: DataLoader,
    maximum_windows: int,
    policy,
    history_frames: int,
    action_horizon: int,
    task_text: str,
) -> dict[str, float]:
    """在留出集上报告 SFT loss 与贪心生成的关键动作一致率。"""
    total_loss = 0.0
    total_agreement = 0.0
    windows = 0
    for sample in loader:
        loss = _sft_loss(policy, sample, history_frames, action_horizon, task_text)
        total_loss += float(loss)
        context, future = _sample_to_context_and_target(
            sample, history_frames, action_horizon, task_text,
        )
        predicted, _text = policy.generate_actions(
            context, include_past_actions=True, temperature=0.0,
        )
        total_agreement += key_action_agreement_rate(predicted, future)
        windows += 1
        if windows >= maximum_windows:
            break
    if windows == 0:
        raise RuntimeError("验证 DataLoader 没有可评估窗口")
    return {
        "loss": total_loss / windows,
        "key_action_agreement": total_agreement / windows,
    }


def main() -> None:
    """下载指定 MineStudio 范围后无限对 Qwen3VL 动作策略做 LoRA SFT。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="runs/data/minestudio")
    parser.add_argument(
        "--dataset-group",
        choices=[c.dataset_group for c in MINESTUDIO_DATASET_GROUPS], default="10xx",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=["image", "action"],
        choices=("action", "meta_info", "image", "event", "motion", "segmentation"),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--max-image-shards", type=int, default=None,
        help="只下载前若干个图像 LMDB 分片供实验或联调；默认下载全部分片",
    )
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--output", default="runs/checkpoints/minecraft_qwen3vl_vla")
    parser.add_argument("--hub-repo-id", default="unjustify/minecraft-qwen3vl-vla-10xx")
    parser.add_argument("--resume", default="auto")
    parser.add_argument(
        "--allow-dataset-transfer",
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--action-format", default=ActionTokenFormat.COMPACT_TAG.value,
        choices=[fmt.value for fmt in ActionTokenFormat],
    )
    parser.add_argument("--action-horizon", type=int, default=5)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=252)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-windows", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if not {"image", "action"}.issubset(arguments.modalities):
        raise ValueError("无限训练必须同时下载 image 和 action")
    if arguments.history_frames < 1 or arguments.action_horizon < 1:
        raise ValueError("history-frames 和 action-horizon 必须大于零")
    if arguments.lora_rank < 1:
        raise ValueError("lora-rank 必须大于零(SFT 需要可训练适配器)")
    if arguments.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation 必须大于零")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("该训练入口需要支持 BF16 的 CUDA GPU")

    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
        maximum_image_shards=arguments.max_image_shards,
    )
    print(json.dumps({
        "event": "dataset_ready",
        "dataset_group": dataset_configuration.dataset_group,
        "image_shards": list(download_selection.image_shards),
        "data_directory": str(data_directory),
    }, ensure_ascii=False), flush=True)

    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_uploader = PublicHubCheckpointUploader(
        arguments.hub_repo_id, output_directory,
    )
    action_format = ActionTokenFormat(arguments.action_format)
    policy_configuration = Qwen3VLPolicyConfiguration(
        model_name=arguments.model_name,
        action_format=action_format,
        action_horizon=arguments.action_horizon,
        max_history_frames=arguments.history_frames,
        max_new_tokens=max(64, 24 * arguments.action_horizon),
    )
    policy = build_qwen3vl_action_policy(
        policy_configuration, device, arguments.cache_directory,
        lora_rank=arguments.lora_rank,
    )
    checkpoint_configuration = _CheckpointConfiguration(
        model_name=arguments.model_name,
        action_format=arguments.action_format,
        action_horizon=arguments.action_horizon,
        history_frames=arguments.history_frames,
        lora_rank=arguments.lora_rank,
    )
    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=arguments.learning_rate)

    resume_path = (
        output_directory / "last.pt"
        if arguments.resume == "auto" else Path(arguments.resume)
    )
    if arguments.resume not in {"", "auto"} and not resume_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {resume_path}")
    first_step = 1
    if arguments.resume and resume_path.is_file():
        completed = _load_adapter(resume_path, policy, checkpoint_configuration)
        first_step = completed + 1
        print(json.dumps({"event": "resumed", "step": completed}), flush=True)

    sequence_length = arguments.history_frames + arguments.action_horizon
    dataset = MineStudioLMDBDataset(
        data_directory=data_directory,
        sequence_length=sequence_length,
        image_size=(arguments.image_height, arguments.image_width),
        task_text=dataset_configuration.task_text,
        camera_max_degrees=CAMERA_SCALE * DEGREES_PER_MOUSE_PIXEL,
        split="train",
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    loader = _data_loader(
        dataset, arguments.workers, shuffle=True,
        prefetch_factor=arguments.prefetch_factor, generator=generator,
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
                split="validation",
                validation_fraction=arguments.validation_fraction,
                seed=arguments.seed,
            )
            validation_loader = _data_loader(
                validation_dataset, arguments.workers, shuffle=False,
                prefetch_factor=arguments.prefetch_factor,
            )
        except RuntimeError as error:
            if "没有可用于该 split 的共同 episode" not in str(error):
                raise
            print(json.dumps({"event": "validation_disabled", "reason": str(error)},
                             ensure_ascii=False), flush=True)

    trainable_count = sum(p.numel() for p in trainable)
    print(json.dumps({
        "event": "training_start",
        "dataset_group": dataset_configuration.dataset_group,
        "windows": len(dataset),
        "trainable_parameters_million": round(trainable_count / 1e6, 2),
        "action_format": arguments.action_format,
    }, ensure_ascii=False), flush=True)

    started = time.time()
    last_step = first_step - 1
    batches = _cycle_batches(loader)
    task_text = dataset_configuration.task_text
    try:
        for step in itertools.count(first_step):
            policy.model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _ in range(arguments.gradient_accumulation):
                sample = next(batches)
                loss = _sft_loss(
                    policy, sample, arguments.history_frames,
                    arguments.action_horizon, task_text,
                ) / arguments.gradient_accumulation
                loss.backward()
                accumulated += float(loss)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            last_step = step
            if step % arguments.log_every == 0:
                elapsed = max(time.time() - started, 1e-4)
                print(json.dumps({
                    "split": "train", "step": step,
                    "loss": round(accumulated, 5),
                    "steps_per_second": round((step - first_step + 1) / elapsed, 3),
                }), flush=True)
            if validation_loader is not None and step % arguments.validate_every == 0:
                policy.model.eval()
                validation = _evaluate(
                    validation_loader, arguments.validation_windows, policy,
                    arguments.history_frames, arguments.action_horizon, task_text,
                )
                print(json.dumps({
                    "split": "validation", "step": step,
                    **{k: round(v, 5) for k, v in validation.items()},
                }), flush=True)
            if step % arguments.save_every == 0:
                checkpoint_path = output_directory / "last.pt"
                _save_checkpoint(
                    checkpoint_path, policy, checkpoint_configuration, step,
                    dataset_configuration.dataset_group, download_selection.image_shards,
                )
                checkpoint_uploader.publish(checkpoint_path, step)
    except KeyboardInterrupt:
        if last_step >= first_step:
            checkpoint_path = output_directory / "last.pt"
            _save_checkpoint(
                checkpoint_path, policy, checkpoint_configuration, last_step,
                dataset_configuration.dataset_group, download_selection.image_shards,
            )
            checkpoint_uploader.publish(checkpoint_path, last_step)
        print(json.dumps({"event": "training_interrupted", "step": last_step}), flush=True)
    finally:
        checkpoint_uploader.close()


if __name__ == "__main__":
    main()
