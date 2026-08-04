"""Validated 2-reference + 6-policy rollout training contract."""

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
    execution_path = Path(path)
    payload = json.loads(execution_path.read_text(encoding="utf-8"))
    samples: list[RolloutSample] = []
    for item in payload["trajectories"]:
        role = item["source_role"]
        if role not in ("reference_expert", "policy_sample"):
            raise ValueError(f"unknown source_role: {role!r}")
        images = tuple(execution_path.parent / frame["path"] for frame in item["frames"])
        if not images or any(not image.is_file() for image in images):
            raise FileNotFoundError(f"incomplete images for {item['candidate_id']}")
        from PIL import Image

        with Image.open(images[0]) as image:
            width, height = image.size
        samples.append(
            RolloutSample(
                str(payload["snapshot_id"]),
                str(item["candidate_id"]),
                role,
                str(item["action_text"]),
                float(item["score"]),
                float(item["relative_advantage"]),
                images,
                width,
                height,
                tuple(map(int, item.get("response_token_ids", ()))),
                tuple(map(float, item.get("old_logprobs", ()))),
                None if item.get("policy_version") is None else str(item["policy_version"]),
                tuple(
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
    composition = (
        sum(sample.behavior_cloning_eligible for sample in samples),
        sum(sample.policy_eligible for sample in samples),
    )
    if (
        len(samples) != 8
        or len({sample.group_id for sample in samples}) != 1
        or composition != (2, 6)
    ):
        raise ValueError(
            f"rollout group must contain one group with 2 reference + 6 policy; got {composition}"
        )


def require_on_policy_logprobs(samples: list[RolloutSample]) -> None:
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
        raise ValueError("missing aligned on-policy metadata: " + ", ".join(missing))


def masks(samples: list[RolloutSample]) -> dict[str, list[bool]]:
    return {
        "policy": [sample.policy_eligible for sample in samples],
        "reference_bc": [sample.behavior_cloning_eligible for sample in samples],
    }
