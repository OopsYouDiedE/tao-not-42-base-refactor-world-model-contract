"""观察驱动 2+6 rollout 的可训练数据合同。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SourceRole = Literal["reference_expert", "policy_sample"]


@dataclass(frozen=True)
class RolloutSample:
    group_id: str
    candidate_id: str
    source_role: SourceRole
    action_text: str
    reward: float
    relative_advantage: float
    image_paths: tuple[Path, ...]
    original_width: int
    original_height: int
    response_token_ids: tuple[int, ...] = ()
    old_logprobs: tuple[float, ...] = ()
    policy_version: str | None = None
    sampling_parameters: tuple[tuple[str, str], ...] = ()

    @property
    def policy_eligible(self) -> bool:
        return self.source_role == "policy_sample"

    @property
    def behavior_cloning_eligible(self) -> bool:
        return self.source_role == "reference_expert"


def load_execution_group(path: str | Path) -> list[RolloutSample]:
    """读取一次真实执行结果，并严格验证 2 reference + 6 policy 边界。"""
    execution_path = Path(path)
    payload = json.loads(execution_path.read_text(encoding="utf-8"))
    group_id = str(payload["snapshot_id"])
    samples: list[RolloutSample] = []
    for item in payload["trajectories"]:
        role = item["source_role"]
        if role not in ("reference_expert", "policy_sample"):
            raise ValueError(f"未知 source_role：{role!r}")
        image_paths = tuple(execution_path.parent / frame["path"] for frame in item["frames"])
        if not image_paths or any(not image.is_file() for image in image_paths):
            raise FileNotFoundError(f"{item['candidate_id']} 的轨迹图片不完整")
        from PIL import Image

        with Image.open(image_paths[0]) as image:
            width, height = image.size
        samples.append(
            RolloutSample(
                group_id=group_id,
                candidate_id=str(item["candidate_id"]),
                source_role=role,
                action_text=str(item["action_text"]),
                reward=float(item["score"]),
                relative_advantage=float(item["relative_advantage"]),
                image_paths=image_paths,
                original_width=width,
                original_height=height,
                response_token_ids=tuple(
                    int(value) for value in item.get("response_token_ids", ())
                ),
                old_logprobs=tuple(float(value) for value in item.get("old_logprobs", ())),
                policy_version=(
                    None if item.get("policy_version") is None else str(item["policy_version"])
                ),
                sampling_parameters=tuple(
                    sorted(
                        (str(key), json.dumps(value, ensure_ascii=False, sort_keys=True))
                        for key, value in item.get("sampling_parameters", {}).items()
                    )
                ),
            )
        )
    _validate_group(samples)
    return samples


def _validate_group(samples: list[RolloutSample]) -> None:
    if len(samples) != 8:
        raise ValueError(f"2+6 组必须正好包含 8 条轨迹，实际为 {len(samples)}")
    if len({sample.group_id for sample in samples}) != 1:
        raise ValueError("一组轨迹必须共享 group_id")
    references = sum(sample.behavior_cloning_eligible for sample in samples)
    policies = sum(sample.policy_eligible for sample in samples)
    if (references, policies) != (2, 6):
        raise ValueError(f"来源组成必须为 2 reference + 6 policy，实际为 {references}+{policies}")


def require_on_policy_logprobs(samples: list[RolloutSample]) -> None:
    """PPO/GRPO 概率比训练前的硬门禁。"""
    missing = [
        sample.candidate_id
        for sample in samples
        if sample.policy_eligible
        and (
            not sample.old_logprobs
            or len(sample.old_logprobs) != len(sample.response_token_ids)
            or sample.policy_version is None
            or not sample.sampling_parameters
        )
    ]
    if missing:
        raise ValueError(
            "概率比训练缺少或未对齐 policy 逐 token old_logprobs/生成元数据："
            + ", ".join(missing)
        )


def masks(samples: list[RolloutSample]) -> dict[str, list[bool]]:
    return {
        "policy": [sample.policy_eligible for sample in samples],
        "reference_bc": [sample.behavior_cloning_eligible for sample in samples],
    }
