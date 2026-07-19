"""在单机 AutoDL GPU 上流式下载并训练完整 MineStudio 课程。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

import torch

from datasets.vpt.minestudio_curriculum import curriculum_stage_names
from datasets.vpt.minestudio_download import (
    list_stage_image_shards,
    local_complete_image_shards,
    prepare_stage_shard,
    prune_completed_stage,
)
from train.minecraft.world_model_warm_start import (
    CHECKPOINT_VERSION,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
)


SCHEDULE_VERSION = "autodl_minecraft_curriculum_v1"
DEFAULT_STAGE_UPDATES = {
    "foundation": 100_000,
    "construction": 50_000,
    "long_horizon": 50_000,
}
DEFAULT_STAGE_LEARNING_RATES = {
    "foundation": 1e-4,
    "construction": 7e-5,
    "long_horizon": 5e-5,
}


@dataclass(frozen=True)
class CurriculumScheduleEntry:
    """单个图像分片的训练目标。"""

    stage: str
    image_shard: str
    image_shard_index: int
    image_shard_count: int
    updates: int
    target_step: int
    learning_rate: float


def distribute_updates(total_updates: int, shard_count: int) -> tuple[int, ...]:
    """把一个阶段的 optimizer update 尽量均匀地分配到所有图像分片。"""
    if total_updates < shard_count:
        raise ValueError("阶段 update 数必须至少等于图像分片数")
    if shard_count < 1:
        raise ValueError("图像分片数必须大于零")
    quotient, remainder = divmod(total_updates, shard_count)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(shard_count)
    )


def build_curriculum_schedule(
    stages: tuple[str, ...],
    stage_image_shards: dict[str, tuple[str, ...]],
    stage_updates: dict[str, int],
    stage_learning_rates: dict[str, float],
) -> tuple[CurriculumScheduleEntry, ...]:
    """构造具有全局累计 step 的确定性分片课程。"""
    schedule = []
    target_step = 0
    for stage in stages:
        image_shards = stage_image_shards[stage]
        allocations = distribute_updates(stage_updates[stage], len(image_shards))
        for index, (image_shard, updates) in enumerate(zip(image_shards, allocations)):
            target_step += updates
            schedule.append(CurriculumScheduleEntry(
                stage=stage,
                image_shard=image_shard,
                image_shard_index=index,
                image_shard_count=len(image_shards),
                updates=updates,
                target_step=target_step,
                learning_rate=stage_learning_rates[stage],
            ))
    return tuple(schedule)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """以原子替换写入 UTF-8 JSON。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_training_state(output_directory: Path) -> dict[str, object] | None:
    """读取训练器随 checkpoint 原子发布的轻量状态。"""
    metadata_path = output_directory / "last.json"
    checkpoint_path = output_directory / "last.pt"
    if metadata_path.is_file():
        if not checkpoint_path.is_file():
            raise RuntimeError("发现 last.json 但 checkpoint 文件不存在")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint_status = checkpoint_path.stat()
        if (
            metadata.get("checkpoint_size") == checkpoint_status.st_size
            and metadata.get("checkpoint_modified_ns") == checkpoint_status.st_mtime_ns
        ):
            return metadata
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True,
        )
        return {
            "version": checkpoint.get("version"),
            "step": checkpoint.get("step"),
            "curriculum_stage": checkpoint.get("curriculum_stage"),
            "image_shards": list(checkpoint.get("image_shards", ())),
        }
    return None


def _validate_cuda_runtime() -> None:
    """在下载大数据前验证 AutoDL 镜像具备 BF16 CUDA 训练能力。"""
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA；请在 AutoDL GPU 实例中运行课程训练器")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16；本训练基线不提供静默精度降级")
    try:
        probe = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
        _ = probe @ probe
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise RuntimeError(
            "CUDA BF16 探针失败；AutoDL 镜像中的 PyTorch/CUDA wheel 可能不包含该 GPU 架构",
        ) from error
    capability = torch.cuda.get_device_capability()
    properties = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "event": "cuda_runtime",
        "device": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bfloat16": True,
        "memory_gib": round(properties.total_memory / 1024**3, 1),
    }, ensure_ascii=False), flush=True)


def _schedule_payload(
    schedule: tuple[CurriculumScheduleEntry, ...],
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """返回用于拒绝意外改动恢复语义的持久化课程配置。"""
    return {
        "version": SCHEDULE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "entries": [asdict(entry) for entry in schedule],
        "training": {
            "effective_batch": (
                arguments.batch * arguments.gradient_accumulation_steps
            ),
            "history": arguments.history,
            "action_horizon": arguments.action_horizon,
            "image_height": arguments.image_height,
            "image_width": arguments.image_width,
            "window_stride": arguments.window_stride,
            "validation_fraction": arguments.validation_fraction,
            "world_weight": arguments.world_weight,
            "kl_weight": arguments.kl_weight,
            "vision_model": arguments.vision_model,
            "text_model": arguments.text_model,
            "small": arguments.small,
            "seed": arguments.seed,
        },
    }


def _persist_or_validate_schedule(path: Path, payload: dict[str, object]) -> None:
    """首次保存课程；恢复时拒绝会改变累计 step 含义的配置漂移。"""
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"{path} 与本次参数不同；为防止错配 checkpoint，请恢复原参数或使用新输出目录",
            )
        return
    _atomic_json(path, payload)


def _training_command(
    entry: CurriculumScheduleEntry,
    arguments: argparse.Namespace,
    checkpoint_path: Path,
) -> list[str]:
    """构造一个分片训练子进程命令。"""
    command = [
        sys.executable, "-m", "train.minecraft.world_model_warm_start",
        "--data-root", str(Path(arguments.data_root).resolve()),
        "--output", str(Path(arguments.output).resolve()),
        "--stage", entry.stage,
        "--steps", str(entry.target_step),
        "--batch", str(arguments.batch),
        "--gradient-accumulation-steps", str(arguments.gradient_accumulation_steps),
        "--workers", str(arguments.workers),
        "--window-stride", str(arguments.window_stride),
        "--validation-fraction", str(arguments.validation_fraction),
        "--history", str(arguments.history),
        "--action-horizon", str(arguments.action_horizon),
        "--image-height", str(arguments.image_height),
        "--image-width", str(arguments.image_width),
        "--learning-rate", str(entry.learning_rate),
        "--world-weight", str(arguments.world_weight),
        "--kl-weight", str(arguments.kl_weight),
        "--save-every", str(arguments.save_every),
        "--validate-every", str(arguments.validate_every),
        "--validation-batches", str(arguments.validation_batches),
        "--log-every", str(arguments.log_every),
        "--seed", str(arguments.seed),
        "--vision-model", arguments.vision_model,
        "--text-model", arguments.text_model,
    ]
    if checkpoint_path.is_file():
        command.extend(["--resume", str(checkpoint_path)])
    if not arguments.fused_optimizer:
        command.append("--no-fused-optimizer")
    if arguments.small:
        command.append("--small")
    return command


def main() -> None:
    """执行可恢复的 MineStudio 全课程分片训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="runs/data/minestudio")
    parser.add_argument("--output", default="runs/checkpoints/minecraft_dreamer_lite")
    parser.add_argument(
        "--stages", nargs="+", choices=curriculum_stage_names(),
        default=list(curriculum_stage_names()),
    )
    parser.add_argument(
        "--foundation-updates", type=int,
        default=DEFAULT_STAGE_UPDATES["foundation"],
    )
    parser.add_argument(
        "--construction-updates", type=int,
        default=DEFAULT_STAGE_UPDATES["construction"],
    )
    parser.add_argument(
        "--long-horizon-updates", type=int,
        default=DEFAULT_STAGE_UPDATES["long_horizon"],
    )
    parser.add_argument(
        "--foundation-learning-rate", type=float,
        default=DEFAULT_STAGE_LEARNING_RATES["foundation"],
    )
    parser.add_argument(
        "--construction-learning-rate", type=float,
        default=DEFAULT_STAGE_LEARNING_RATES["construction"],
    )
    parser.add_argument(
        "--long-horizon-learning-rate", type=float,
        default=DEFAULT_STAGE_LEARNING_RATES["long_horizon"],
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--world-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument(
        "--replace-image-shards", action=argparse.BooleanOptionalAction, default=True,
        help="下载新分片成功后删除当前阶段旧图像分片",
    )
    parser.add_argument(
        "--prune-completed-stages", action=argparse.BooleanOptionalAction, default=True,
        help="阶段 checkpoint 完成后删除该阶段本地数据，最后一个启用阶段除外",
    )
    parser.add_argument(
        "--fused-optimizer", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    stages = tuple(arguments.stages)
    expected_order = tuple(
        stage for stage in curriculum_stage_names() if stage in stages
    )
    if stages != expected_order or len(set(stages)) != len(stages):
        raise ValueError("stages 必须按 foundation、construction、long_horizon 顺序且不重复")
    stage_updates = {
        "foundation": arguments.foundation_updates,
        "construction": arguments.construction_updates,
        "long_horizon": arguments.long_horizon_updates,
    }
    stage_learning_rates = {
        "foundation": arguments.foundation_learning_rate,
        "construction": arguments.construction_learning_rate,
        "long_horizon": arguments.long_horizon_learning_rate,
    }
    if arguments.batch < 1 or arguments.gradient_accumulation_steps < 1:
        raise ValueError("batch 和 gradient-accumulation-steps 必须大于零")
    if arguments.workers < 0 or arguments.download_workers < 1:
        raise ValueError("workers 不能为负且 download-workers 必须大于零")
    if not 0.0 <= arguments.validation_fraction < 1.0:
        raise ValueError("validation-fraction 必须位于 [0,1)")
    if any(stage_updates[stage] < 1 for stage in stages):
        raise ValueError("所有启用阶段的 update 数必须大于零")
    if any(stage_learning_rates[stage] <= 0.0 for stage in stages):
        raise ValueError("所有启用阶段的 learning rate 必须大于零")
    if not arguments.dry_run:
        _validate_cuda_runtime()
    stage_image_shards = {
        stage: list_stage_image_shards(stage)
        for stage in stages
    }
    schedule = build_curriculum_schedule(
        stages, stage_image_shards, stage_updates, stage_learning_rates,
    )
    payload = _schedule_payload(schedule, arguments)
    if arguments.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return

    output_directory = Path(arguments.output).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    schedule_path = output_directory / "autodl_schedule.json"
    training_state = _read_training_state(output_directory)
    if training_state is not None and not schedule_path.is_file():
        raise RuntimeError("发现已有 checkpoint 但没有 autodl_schedule.json；请使用新输出目录")
    _persist_or_validate_schedule(schedule_path, payload)
    checkpoint_path = output_directory / "last.pt"
    completed_step = int(training_state["step"]) if training_state else 0
    prefetched_stages = {
        stage
        for stage in stages
        if len(local_complete_image_shards(arguments.data_root, stage)) > 1
    }
    if prefetched_stages:
        print(json.dumps({
            "event": "preserve_prefetched_images",
            "stages": sorted(prefetched_stages),
        }, ensure_ascii=False), flush=True)
    previous_target = 0
    final_stage = stages[-1]
    for entry in schedule:
        stage_is_complete = entry.image_shard_index == entry.image_shard_count - 1
        if completed_step >= entry.target_step:
            if (
                stage_is_complete
                and entry.stage != final_stage
                and arguments.prune_completed_stages
            ):
                prune_completed_stage(arguments.data_root, entry.stage)
            previous_target = entry.target_step
            continue
        if completed_step > previous_target and training_state is not None:
            active_shards = tuple(training_state.get("image_shards", ()))
            expected_shard = PurePosixPath(entry.image_shard).name
            if (
                training_state.get("curriculum_stage") != entry.stage
                or expected_shard not in active_shards
            ):
                raise RuntimeError("checkpoint 位于分片中途，但记录的阶段/分片与课程不一致")
        print(json.dumps({
            "event": "prepare_shard",
            **asdict(entry),
        }, ensure_ascii=False), flush=True)
        _, selection = prepare_stage_shard(
            stage_name=entry.stage,
            data_root=arguments.data_root,
            image_shard_index=entry.image_shard_index,
            maximum_workers=arguments.download_workers,
            replace_image_shards=(
                arguments.replace_image_shards
                and entry.stage not in prefetched_stages
            ),
        )
        if selection.image_shard != entry.image_shard:
            raise RuntimeError("远端图像分片列表在课程创建后发生变化，请使用新输出目录重建课程")
        subprocess.run(
            _training_command(entry, arguments, checkpoint_path),
            check=True,
        )
        training_state = _read_training_state(output_directory)
        if training_state is None:
            raise RuntimeError("训练子进程结束后没有发布 last.json")
        completed_step = int(training_state["step"])
        if completed_step != entry.target_step:
            raise RuntimeError(
                f"训练子进程结束于 step={completed_step}，预期 {entry.target_step}",
            )
        if (
            stage_is_complete
            and entry.stage != final_stage
            and arguments.prune_completed_stages
        ):
            removed = prune_completed_stage(arguments.data_root, entry.stage)
            print(json.dumps({
                "event": "prune_completed_stage",
                "stage": entry.stage,
                "removed": removed,
            }, ensure_ascii=False), flush=True)
        previous_target = entry.target_step
    print(json.dumps({
        "event": "curriculum_complete",
        "step": completed_step,
        "checkpoint": str(checkpoint_path),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
