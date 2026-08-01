from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tao.baselines.codex.client import CodexInvocation
from tao.baselines.codex.contracts import (
    SCORE_DIMENSIONS,
    compile_teacher_action,
    generation_schema,
)
from tao.baselines.codex.teacher_only import (
    TRAJECTORY_LANGUAGE_STYLE_PROMPTS,
    TRAJECTORY_LANGUAGE_STYLE_TIERS,
    TeacherBatchRequest,
    TeacherOnlyPipeline,
)


def _teacher_action(*, horizon: int = 4, key: str = "W") -> dict[str, Any]:
    return {
        "protocol": "tap-v1",
        "horizon_ticks": horizon,
        "segments": [
            {
                "duration_ticks": horizon,
                "keys": [key],
                "mouse": [0, 0],
                "scroll": 0,
            }
        ],
        "summary": f"use {key}",
    }


def test_generation_schema_avoids_unsupported_unique_items_keyword() -> None:
    keys_schema = generation_schema(4)["properties"]["segments"]["items"]["properties"]["keys"]
    assert "uniqueItems" not in keys_schema


def test_teacher_action_compiler_rejects_wrong_horizon_without_padding() -> None:
    value = _teacher_action(horizon=3)
    value["horizon_ticks"] = 4
    with pytest.raises(ValueError, match="不补齐也不截断"):
        compile_teacher_action(
            value,
            candidate_id="T01",
            expected_horizon_ticks=4,
            generation_audit={},
        )


def test_teacher_action_compiler_rejects_unknown_key() -> None:
    value = _teacher_action(key="hard-coded-policy")
    with pytest.raises(ValueError, match="未知按键"):
        compile_teacher_action(
            value,
            candidate_id="T01",
            expected_horizon_ticks=4,
            generation_audit={},
        )


class FakeTeacherClient:
    keys = ("W", "A", "S", "D", "space", "shift", "ctrl", "MouseLeft")

    def __init__(self, *, invalid_first_generation: bool = False) -> None:
        self.calls: list[str] = []
        self.invalid_first_generation = invalid_first_generation

    def run_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        images: tuple[Path, ...] = (),
    ) -> CodexInvocation:
        self.calls.append(prompt)
        if '"operation": "generate_teacher_trajectory"' in prompt:
            slot = int(re.search(r'"trajectory_slot": (\d+)', prompt).group(1))
            result = _teacher_action(key=self.keys[slot - 1])
            if self.invalid_first_generation:
                self.invalid_first_generation = False
                result["segments"][0]["duration_ticks"] = 3
        else:
            anonymous_ids = schema["properties"]["scores"]["items"]["properties"][
                "anonymous_id"
            ]["enum"]
            result = {
                "scores": [
                    {
                        "anonymous_id": anonymous_id,
                        "dimensions": {name: 3 for name in SCORE_DIMENSIONS},
                        "rationale": "same score for deterministic tie break",
                        "safety_flags": [],
                    }
                    for anonymous_id in anonymous_ids
                ]
            }
        return CodexInvocation(
            result=result,
            model="fake",
            attempts=1,
            wall_ms=1.0,
            image_count=len(images),
        )


def test_teacher_only_retries_semantically_invalid_generation(tmp_path: Path) -> None:
    image = tmp_path / "initial.png"
    image.write_bytes(b"initial")
    request = TeacherBatchRequest(
        task="approach target",
        snapshot_id="same-snapshot",
        initial_image=image,
        initial_state={},
        horizon_ticks=4,
        output_directory=tmp_path,
    )
    client = FakeTeacherClient(invalid_first_generation=True)

    candidate = TeacherOnlyPipeline(client)._generate_candidate(request, image, 1, ())

    assert len(client.calls) == 2
    assert candidate.generation_audit["semantic_attempt"] == 2


def test_teacher_only_runs_eight_selects_four_and_skips_training(tmp_path: Path) -> None:
    image = tmp_path / "initial.png"
    image.write_bytes(b"initial")
    client = FakeTeacherClient()
    executed: list[str] = []

    def execute(candidate: Any) -> dict[str, Any]:
        executed.append(candidate.candidate_id)
        directory = tmp_path / candidate.candidate_id
        directory.mkdir()
        final_image = directory / "final.png"
        final_image.write_bytes(b"final")
        return {
            "candidate_id": candidate.candidate_id,
            "ticks": len(candidate.ticks),
            "execution_status": "SUCCESS",
            "success": True,
            "frames": [
                {"tick": len(candidate.ticks), "path": f"{candidate.candidate_id}/final.png"}
            ],
        }

    result = TeacherOnlyPipeline(client).run(
        TeacherBatchRequest(
            task="approach target",
            snapshot_id="same-snapshot",
            initial_image=image,
            initial_state={"x": 1.0, "y": 2.0, "z": 3.0},
            horizon_ticks=4,
            output_directory=tmp_path,
        ),
        execute,
    )

    assert len(client.calls) == 9
    generation_calls = [
        prompt
        for prompt in client.calls
        if '"operation": "generate_teacher_trajectory"' in prompt
    ]
    assert len(generation_calls) == 8
    assert len(set(TRAJECTORY_LANGUAGE_STYLE_PROMPTS)) == 8
    assert all(style.count("。") == 1 for style in TRAJECTORY_LANGUAGE_STYLE_PROMPTS)
    assert TRAJECTORY_LANGUAGE_STYLE_TIERS == ("preferred", "preferred") + ("diverse",) * 6
    for prompt, style in zip(generation_calls, TRAJECTORY_LANGUAGE_STYLE_PROMPTS, strict=True):
        assert prompt.count(style) == 1
    assert [
        candidate.generation_audit["language_style_prompt"]
        for candidate in result.candidates
    ] == list(TRAJECTORY_LANGUAGE_STYLE_PROMPTS)
    assert executed == [f"T{index:02d}" for index in range(1, 9)]
    assert result.selected_candidate_ids == ("T01", "T02", "T03", "T04")
    assert len(result.candidates) == 8
    assert len(result.executions) == 8
    assert all(score.total == 60.0 for score in result.scores)

    pipeline_result = json.loads((tmp_path / "pipeline_result.json").read_text("utf-8"))
    assert pipeline_result["training"] == {
        "model_load_attempted": False,
        "optimizer_steps": 0,
        "reason": "本机没有可加载的动作策略或审核策略模型；本轮只产出 BC 训练样本。",
        "status": "skipped_no_local_model",
        "training_skipped": True,
    }
    assert pipeline_result["dataset"]["behavior_cloning_samples"] == 4
    assert pipeline_result["dataset"]["rlhf_samples"] == 0
    assert set(pipeline_result["rlhf_tracks"]) == {
        "environment_policy",
        "review_policy",
    }
    assert (
        pipeline_result["teacher"]["trajectory_language_style_prompts"]
        == list(TRAJECTORY_LANGUAGE_STYLE_PROMPTS)
    )
    selected = [
        json.loads(line)
        for line in (tmp_path / "selected.jsonl").read_text("utf-8").splitlines()
    ]
    assert [row["candidate_id"] for row in selected] == ["T01", "T02", "T03", "T04"]
    assert all(row["sample_type"] == "behavior_cloning" for row in selected)
    assert all(row["old_logprobs"] is None for row in selected)
