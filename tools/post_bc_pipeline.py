"""等待 BC 完成，发布最佳 LoRA，并启动 2+6 强化训练流水线。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


def process_is_running(pid: int) -> bool:
    """检查进程存在且不是 zombie。"""
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    fields = stat.read_text(encoding="utf-8").split()
    return len(fields) > 2 and fields[2] != "Z"


def write_status(path: Path, stage: str, **values: Any) -> None:
    """原子更新后处理状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_at": time.time(), **values}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def best_training_result(output: Path) -> dict[str, Any]:
    """从 epoch checkpoint 中提取最佳验证 loss 和 checkpoint。"""
    evaluations: list[dict[str, Any]] = []
    for state_path in output.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for item in state.get("log_history", []):
            if "eval_loss" in item:
                evaluations.append(
                    {
                        "checkpoint": str(state_path.parent),
                        "epoch": float(item["epoch"]),
                        "eval_loss": float(item["eval_loss"]),
                        "step": int(item.get("step", state.get("global_step", 0))),
                    }
                )
    if not evaluations:
        raise RuntimeError("BC 结束后没有找到任何验证指标")
    return min(evaluations, key=lambda item: item["eval_loss"])


def model_card(best: dict[str, Any]) -> str:
    """生成公开 LoRA 的模型卡。"""
    return f"""---
base_model: unsloth/gemma-4-26B-A4B-it
library_name: peft
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- unsloth
- lora
- minecraft
- vision
- minestudio
datasets:
- unjustify/minestudio-trajectory-sft-237
---

# MineStudio Gemma 4 26B-A4B Trajectory LoRA

该 LoRA 使用 `unsloth/gemma-4-26B-A4B-it` 和
`minestudio-trajectory-sft-768.h5` 进行视觉行为克隆训练。

| 项目 | 值 |
|---|---:|
| 数据量 | 768 |
| 训练 / 验证 | 691 / 77 |
| 划分种子 | 3407 |
| LoRA rank / alpha | 32 / 32 |
| Micro-batch / 有效 batch | 4 / 8 |
| 最大 epoch | 10 |
| 早停 patience | 2 |
| 最佳 epoch | {best['epoch']:.0f} |
| 最佳 step | {best['step']} |
| 最佳 eval loss | {best['eval_loss']:.8f} |

验证集用于早停与最佳 checkpoint 选择，没有划分测试集。仓库只包含 PEFT LoRA，
不包含 Gemma 4 基座权重。
"""


def run_checked(command: list[str], log_path: Path) -> None:
    """执行阶段命令并把 stdout/stderr 追加到统一日志。"""
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--bc-output", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--rollout-output", type=Path, required=True)
    parser.add_argument("--rlhf-output", type=Path, required=True)
    parser.add_argument("--rlhf-hf-repo", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    arguments = parser.parse_args()
    log_path = arguments.status.with_suffix(".log")

    write_status(arguments.status, "waiting_for_bc", pid=arguments.wait_pid)
    while process_is_running(arguments.wait_pid):
        time.sleep(arguments.poll_seconds)

    adapter = arguments.bc_output / "lora_adapter"
    required = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        write_status(arguments.status, "bc_failed", missing=missing)
        raise RuntimeError("BC 进程结束，但最终 lora_adapter 不完整")

    best = best_training_result(arguments.bc_output)
    (adapter / "README.md").write_text(model_card(best), encoding="utf-8")
    write_status(arguments.status, "uploading", best=best, adapter=str(adapter))
    api = HfApi()
    repository = api.create_repo(arguments.hf_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=arguments.hf_repo,
        repo_type="model",
        folder_path=adapter,
        commit_message="Upload best A4B behavior cloning LoRA",
    )
    publication = {
        "repository": str(repository),
        "repo_id": arguments.hf_repo,
        "adapter": str(adapter),
        "best": best,
    }
    (arguments.bc_output / "publication.json").write_text(
        json.dumps(publication, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_status(arguments.status, "generating_rollouts", **publication)
    run_checked(
        [
            sys.executable,
            "-m",
            "tools.audits.terra_tree_approach_batch8",
            "--runtime",
            str(arguments.runtime),
            "--output",
            str(arguments.rollout_output),
            "--policy-adapter",
            arguments.hf_repo,
        ],
        log_path,
    )
    execution = arguments.rollout_output / "execution.json"
    if not execution.is_file():
        raise RuntimeError("2+6 runner 未生成 execution.json")

    write_status(
        arguments.status,
        "training_rlhf",
        execution=str(execution),
        rlhf_repository=f"https://huggingface.co/{arguments.rlhf_hf_repo}",
        **publication,
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "train.gemma_vision_rlhf",
            "--adapter",
            arguments.hf_repo,
            "--execution",
            str(execution),
            "--output-dir",
            str(arguments.rlhf_output),
            "--hf-repo",
            arguments.rlhf_hf_repo,
        ],
        log_path,
    )
    write_status(
        arguments.status,
        "complete",
        execution=str(execution),
        rlhf_repository=f"https://huggingface.co/{arguments.rlhf_hf_repo}",
        **publication,
    )


if __name__ == "__main__":
    main()
