"""八条 Codex 教师轨迹的生成、执行、评分、筛选与导出管线。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tao.baselines.codex.client import CodexClient
from tao.baselines.codex.contracts import (
    ALLOWED_KEYS,
    SCORE_DIMENSIONS,
    TeacherCandidate,
    TeacherScore,
    compile_teacher_action,
    generation_schema,
    parse_teacher_scores,
    scoring_schema,
)

BATCH_SIZE = 8
SELECTED_COUNT = 4


@dataclass(frozen=True)
class TeacherBatchRequest:
    task: str
    snapshot_id: str
    initial_image: Path
    initial_state: dict[str, Any]
    horizon_ticks: int
    output_directory: Path

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("教师任务不能为空")
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id 不能为空")
        if self.horizon_ticks < 1:
            raise ValueError("horizon_ticks 必须大于零")


@dataclass(frozen=True)
class TeacherOnlyResult:
    candidates: tuple[TeacherCandidate, ...]
    executions: tuple[dict[str, Any], ...]
    scores: tuple[TeacherScore, ...]
    selected_candidate_ids: tuple[str, ...]
    scoring_audit: dict[str, Any]
    output_directory: Path

    def pipeline_result(self) -> dict[str, Any]:
        return {
            "mode": "teacher_only",
            "batch_size": len(self.candidates),
            "selected_count": len(self.selected_candidate_ids),
            "selected_fraction": 0.5,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "teacher": {
                "provider": "codex-cli",
                "generation_sessions": sum(
                    int(candidate.generation_audit.get("semantic_attempt", 1))
                    for candidate in self.candidates
                ),
                "scoring_sessions": int(self.scoring_audit.get("semantic_attempt", 1)),
                "same_rubric_for_all": True,
                "candidate_source_hidden_during_scoring": True,
            },
            "training": {
                "training_skipped": True,
                "status": "skipped_no_local_model",
                "reason": "本机没有可加载的策略模型；本轮只产出 BC 训练样本。",
                "model_load_attempted": False,
                "optimizer_steps": 0,
            },
            "dataset": {
                "behavior_cloning_samples": len(self.selected_candidate_ids),
                "rlhf_samples": 0,
                "rlhf_exclusion_reason": (
                    "Codex 教师轨迹没有行为策略 old_logprobs，不能作为 clipped RLHF rollout。"
                ),
            },
            "artifacts": {
                "candidates": "candidates.json",
                "execution": "execution.json",
                "teacher_scores": "teacher_scores.json",
                "selected_bc_samples": "selected.jsonl",
                "pipeline_result": "pipeline_result.json",
            },
        }


class TeacherOnlyPipeline:
    """固定执行 8 条教师轨迹并选择评分最高的 4 条。"""

    def __init__(self, client: CodexClient, *, semantic_max_attempts: int = 3):
        if semantic_max_attempts < 1:
            raise ValueError("semantic_max_attempts 必须至少为 1")
        self.client = client
        self.semantic_max_attempts = semantic_max_attempts

    def run(
        self,
        request: TeacherBatchRequest,
        execute: Callable[[TeacherCandidate], dict[str, Any]],
    ) -> TeacherOnlyResult:
        request.output_directory.mkdir(parents=True, exist_ok=True)
        image = request.initial_image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"教师初始观察不存在：{image}")

        collected: list[TeacherCandidate] = []
        for index in range(1, BATCH_SIZE + 1):
            collected.append(
                self._generate_candidate(request, image, index, tuple(collected))
            )
        candidates = tuple(collected)
        _atomic_json(
            request.output_directory / "candidates.json",
            {
                "mode": "teacher_only",
                "batch_size": BATCH_SIZE,
                "source": "codex_teacher",
                "snapshot_id": request.snapshot_id,
                "horizon_ticks": request.horizon_ticks,
                "candidates": [_candidate_record(candidate) for candidate in candidates],
            },
        )

        executions: list[dict[str, Any]] = []
        for candidate in candidates:
            result = execute(candidate)
            if not isinstance(result, dict):
                raise TypeError("轨迹执行器必须返回字典")
            result_id = result.get("candidate_id", candidate.candidate_id)
            if result_id != candidate.candidate_id:
                raise ValueError("轨迹执行结果 candidate_id 与输入不一致")
            executions.append({**result, "candidate_id": candidate.candidate_id})

        scores, scoring_audit = self._score_batch(request, candidates, tuple(executions), image)
        score_by_id = {score.candidate_id: score for score in scores}
        ranked = sorted(scores, key=lambda score: (-score.total, score.candidate_id))
        selected_ids = tuple(score.candidate_id for score in ranked[:SELECTED_COUNT])
        if len(selected_ids) != SELECTED_COUNT:
            raise RuntimeError("teacher-only 管线必须选出恰好 4 条轨迹")

        execution_rows = []
        for execution in executions:
            score = score_by_id[execution["candidate_id"]]
            execution_rows.append(
                {
                    **execution,
                    "teacher_score": _score_record(score),
                    "selected_for_bc": execution["candidate_id"] in selected_ids,
                }
            )
        _atomic_json(
            request.output_directory / "execution.json",
            {
                "snapshot_id": request.snapshot_id,
                "same_snapshot_for_all": True,
                "trajectories": execution_rows,
            },
        )
        _atomic_json(
            request.output_directory / "teacher_scores.json",
            {
                "rubric": {
                    "scale": "每项 0-5；总分由本地代码计算",
                    "weights": SCORE_DIMENSIONS,
                },
                "anonymous_batch_scoring": True,
                "scoring_audit": scoring_audit,
                "scores": [_score_record(score) for score in ranked],
                "selection": {
                    "method": "total_desc_then_candidate_id_asc",
                    "top_fraction": 0.5,
                    "selected_count": SELECTED_COUNT,
                    "selected_candidate_ids": list(selected_ids),
                },
            },
        )
        selected_records = [
            _bc_sample(request, candidate, score_by_id[candidate.candidate_id], execution_rows)
            for candidate in candidates
            if candidate.candidate_id in selected_ids
        ]
        selected_records.sort(key=lambda row: selected_ids.index(row["candidate_id"]))
        _atomic_jsonl(request.output_directory / "selected.jsonl", selected_records)

        result = TeacherOnlyResult(
            candidates=candidates,
            executions=tuple(execution_rows),
            scores=scores,
            selected_candidate_ids=selected_ids,
            scoring_audit=scoring_audit,
            output_directory=request.output_directory,
        )
        _atomic_json(request.output_directory / "pipeline_result.json", result.pipeline_result())
        return result

    def _generate_candidate(
        self,
        request: TeacherBatchRequest,
        image: Path,
        index: int,
        existing: tuple[TeacherCandidate, ...],
    ) -> TeacherCandidate:
        failures: list[str] = []
        existing_actions = {candidate.action_text for candidate in existing}
        for semantic_attempt in range(1, self.semantic_max_attempts + 1):
            prompt = _generation_prompt(
                request,
                index,
                semantic_attempt=semantic_attempt,
                previous_failure=failures[-1] if failures else None,
                existing=existing,
            )
            invocation = self.client.run_structured(
                prompt,
                generation_schema(request.horizon_ticks),
                images=(image,),
            )
            try:
                candidate = compile_teacher_action(
                    invocation.result,
                    candidate_id=f"T{index:02d}",
                    expected_horizon_ticks=request.horizon_ticks,
                    generation_audit={
                        **invocation.audit_dict(),
                        "semantic_attempt": semantic_attempt,
                    },
                )
                if candidate.action_text in existing_actions:
                    raise ValueError("轨迹与同 batch 的已有轨迹完全重复")
                return candidate
            except ValueError as error:
                failures.append(str(error))
        raise RuntimeError(
            f"候选 T{index:02d} 在 {self.semantic_max_attempts} 次语义重试后仍不合法："
            + "；".join(failures)
        )

    def _score_batch(
        self,
        request: TeacherBatchRequest,
        candidates: tuple[TeacherCandidate, ...],
        executions: tuple[dict[str, Any], ...],
        initial_image: Path,
    ) -> tuple[tuple[TeacherScore, ...], dict[str, Any]]:
        shuffled = sorted(
            candidates,
            key=lambda candidate: hashlib.sha256(
                f"{request.snapshot_id}:{candidate.candidate_id}".encode()
            ).hexdigest(),
        )
        anonymous_to_candidate = {
            f"A{index:02d}": candidate.candidate_id
            for index, candidate in enumerate(shuffled, start=1)
        }
        candidate_to_anonymous = {
            candidate_id: anonymous_id
            for anonymous_id, candidate_id in anonymous_to_candidate.items()
        }
        execution_by_id = {row["candidate_id"]: row for row in executions}
        evidence_images: list[Path] = [initial_image]
        review_rows: list[dict[str, Any]] = []
        for candidate in shuffled:
            execution = execution_by_id[candidate.candidate_id]
            evidence = _final_evidence_image(request.output_directory, execution)
            image_index = None
            if evidence is not None:
                evidence_images.append(evidence)
                image_index = len(evidence_images)
            review_rows.append(
                {
                    "anonymous_id": candidate_to_anonymous[candidate.candidate_id],
                    "summary": candidate.summary,
                    "action_text": candidate.action_text,
                    "execution": _execution_for_review(execution),
                    "final_evidence_image_number": image_index,
                }
            )
        anonymous_ids = tuple(anonymous_to_candidate)
        failures: list[str] = []
        for semantic_attempt in range(1, self.semantic_max_attempts + 1):
            prompt = _scoring_prompt(
                request,
                review_rows,
                semantic_attempt=semantic_attempt,
                previous_failure=failures[-1] if failures else None,
            )
            invocation = self.client.run_structured(
                prompt,
                scoring_schema(anonymous_ids),
                images=tuple(evidence_images),
            )
            try:
                scores = parse_teacher_scores(
                    invocation.result,
                    anonymous_to_candidate=anonymous_to_candidate,
                )
                return scores, {
                    **invocation.audit_dict(),
                    "semantic_attempt": semantic_attempt,
                    "anonymous_ids": list(anonymous_ids),
                    "initial_image_number": 1,
                    "evidence_image_count": len(evidence_images) - 1,
                }
            except ValueError as error:
                failures.append(str(error))
        raise RuntimeError(
            f"统一评分在 {self.semantic_max_attempts} 次语义重试后仍不合法："
            + "；".join(failures)
        )


def _generation_prompt(
    request: TeacherBatchRequest,
    index: int,
    *,
    semantic_attempt: int,
    previous_failure: str | None,
    existing: tuple[TeacherCandidate, ...],
) -> str:
    payload = {
        "operation": "generate_teacher_trajectory",
        "protocol": "tap-v1",
        "trajectory_slot": index,
        "semantic_attempt": semantic_attempt,
        "batch_size": BATCH_SIZE,
        "task": request.task,
        "snapshot_id": request.snapshot_id,
        "horizon_ticks": request.horizon_ticks,
        "tick_duration_ms": 50,
        "initial_state": _initial_state_for_prompt(request.initial_state),
        "allowed_keys": sorted(ALLOWED_KEYS),
        "existing_trajectory_summaries": [candidate.summary for candidate in existing],
        "previous_rejection": previous_failure,
    }
    return (
        "你是 TAO 的离线视觉动作教师。第一张图片是当前快照的初始观察。"
        "请为指定槽位独立规划一条可真实执行的轨迹。动作段中的 keys、mouse、scroll "
        "会在 duration_ticks 内每 tick 重复执行。所有 duration_ticks 之和必须严格等于 "
        "horizon_ticks。不得补 tick、截断、使用未知键、假设图片和状态中不存在的资源，"
        "也不得输出解释性额外字段。八个槽位应采用有意义的不同策略。\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _scoring_prompt(
    request: TeacherBatchRequest,
    review_rows: list[dict[str, Any]],
    *,
    semantic_attempt: int,
    previous_failure: str | None,
) -> str:
    payload = {
        "operation": "score_teacher_trajectory_batch",
        "semantic_attempt": semantic_attempt,
        "previous_rejection": previous_failure,
        "task": request.task,
        "snapshot_id": request.snapshot_id,
        "rubric": {
            name: {"weight": weight, "scale": "integer 0-5"}
            for name, weight in SCORE_DIMENSIONS.items()
        },
        "image_mapping": (
            "图片 1 是所有候选共用的初始观察；每个候选的 final_evidence_image_number "
            "指向其执行末帧。"
        ),
        "candidates": review_rows,
    }
    return (
        "你是 TAO 的统一轨迹审核教师。候选已经匿名化，来源信息不可见。请在一次响应中"
        "使用完全相同的 rubric 审核全部八条轨迹。TAP 合法性、固定 horizon、同快照恢复"
        "和完整执行由本地代码硬校验；你只评估任务进展、安全性、视觉因果一致性、时序"
        "正确性和动作效率。不得自行给总分，总分由本地代码按权重计算。\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _initial_state_for_prompt(state: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
        "is_on_ground",
        "is_dead",
        "raycast_block",
        "raycast_position",
        "inventory",
        "mined_statistics",
        "nearby_block_counts",
    )
    return {key: state[key] for key in allowed if key in state}


def _execution_for_review(execution: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "ticks",
        "execution_status",
        "success",
        "initial_distance_to_observed_log",
        "final_distance_to_observed_log",
        "distance_progress",
        "required_distance_progress",
        "progress_metrics",
        "success_evidence",
        "final_state",
    )
    value = {key: execution[key] for key in allowed if key in execution}
    if isinstance(value.get("final_state"), dict):
        value["final_state"] = _initial_state_for_prompt(value["final_state"])
    return value


def _final_evidence_image(output: Path, execution: dict[str, Any]) -> Path | None:
    frames = execution.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    relative = frames[-1].get("path") if isinstance(frames[-1], dict) else None
    if not isinstance(relative, str):
        return None
    path = (output / relative).resolve()
    return path if path.is_file() else None


def _candidate_record(candidate: TeacherCandidate) -> dict[str, Any]:
    segments = []
    cursor = 0
    for segment in candidate.segments:
        tick = candidate.ticks[cursor]
        segments.append(
            {
                "duration_ticks": segment.duration_ticks,
                "keys": list(segment.keys),
                "mouse": list(segment.mouse),
                "scroll": tick.scroll,
            }
        )
        cursor += segment.duration_ticks
    return {
        "candidate_id": candidate.candidate_id,
        "source_role": candidate.source_role,
        "summary": candidate.summary,
        "protocol": "tap-v1",
        "ticks": len(candidate.ticks),
        "segments": segments,
        "action_text": candidate.action_text,
        "generation_audit": candidate.generation_audit,
        "rlhf_eligible": False,
        "rlhf_exclusion_reason": "教师轨迹没有行为策略 old_logprobs",
    }


def _score_record(score: TeacherScore) -> dict[str, Any]:
    return {
        "candidate_id": score.candidate_id,
        "anonymous_id": score.anonymous_id,
        "dimensions": score.dimensions,
        "total": score.total,
        "rationale": score.rationale,
        "safety_flags": list(score.safety_flags),
    }


def _bc_sample(
    request: TeacherBatchRequest,
    candidate: TeacherCandidate,
    score: TeacherScore,
    executions: list[dict[str, Any]],
) -> dict[str, Any]:
    execution = next(row for row in executions if row["candidate_id"] == candidate.candidate_id)
    return {
        "sample_type": "behavior_cloning",
        "source": "codex_teacher",
        "candidate_id": candidate.candidate_id,
        "snapshot_id": request.snapshot_id,
        "task": request.task,
        "initial_image": str(request.initial_image),
        "protocol": "tap-v1",
        "horizon_ticks": len(candidate.ticks),
        "action_text": candidate.action_text,
        "teacher_score": _score_record(score),
        "execution": _execution_for_review(execution),
        "training_status": "pending_bc_training",
        "rlhf_eligible": False,
        "old_logprobs": None,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)
