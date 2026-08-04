"""从真实 CraftGround 轨迹构建 fail-closed 的 2+6 训练合同。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from interaction_trajectory_review_agents import review_trajectory
from relative_advantage_comparison_training import build_comparison_group
from shared_tools import atomic_write_json


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象：{path}")
    return value


def _assert_shared_start(teacher: dict[str, Any], policy: dict[str, Any]) -> str:
    starts = (teacher.get("shared_start"), policy.get("shared_start"))
    if any(not isinstance(start, dict) for start in starts):
        raise ValueError("教师或策略结果缺少 shared_start 事实")
    typed = tuple(start for start in starts if isinstance(start, dict))
    if any(start.get("restore_probe_passed") is not True for start in typed):
        raise ValueError("所有轨迹运行都必须通过真实快照恢复探针")
    fingerprints: list[list[dict[str, Any]]] = []
    for start in typed:
        values = start.get("state_fingerprints")
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, dict) for value in values)
        ):
            raise ValueError("单次运行缺少有效起点状态")
        fingerprints.append(values)
    if any(any(value != values[0] for value in values) for values in fingerprints):
        raise ValueError("单次运行内的起点状态不一致")
    if fingerprints[0][0] != fingerprints[1][0]:
        raise ValueError("教师与本地策略的逻辑起点不一致")
    hashes = []
    for start in typed:
        baseline = start.get("baseline_world")
        instances = baseline.get("instances") if isinstance(baseline, dict) else None
        if not instances:
            raise ValueError("所有轨迹必须来自明确的基准世界")
        values = {item.get("source_sha256") for item in instances}
        if len(values) != 1 or None in values:
            raise ValueError("基准世界实例哈希缺失或不一致")
        hashes.append(values.pop())
    if hashes[0] != hashes[1]:
        raise ValueError("教师与本地策略的基准世界哈希不一致")
    return str(hashes[0])


def _accepted_control(trajectory: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    records = trajectory.get("generation_records")
    decisions = trajectory.get("model_decisions")
    contexts = trajectory.get("generation_contexts")
    if (
        not isinstance(records, list)
        or not isinstance(decisions, dict)
        or not isinstance(contexts, dict)
    ):
        raise ValueError("轨迹缺少模型生成审计记录")
    for record in records:
        if record.get("status") != "completed":
            continue
        generation_id = str(record.get("generation_id"))
        decision = decisions.get(generation_id)
        context = contexts.get(generation_id)
        if isinstance(decision, dict) and decision.get("control") and isinstance(context, dict):
            return str(decision["control"]), context
    raise ValueError("轨迹没有被执行器接受的模型动作")


def _frame_paths(context: dict[str, Any], output: Path) -> list[dict[str, str]]:
    paths = context.get("observation_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("生成记录缺少真实观察路径")
    frames = []
    for raw in paths:
        path = Path(str(raw)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append({"path": os.path.relpath(path, output.parent)})
    return frames


def build_execution(
    teacher_dir: Path, policy_dir: Path, output: Path, *, action_budget_ticks: int = 512
) -> dict[str, Any]:
    teacher_result = _load_json(teacher_dir / "result.json")
    policy_result = _load_json(policy_dir / "result.json")
    baseline_hash = _assert_shared_start(teacher_result, policy_result)
    policy_metadata = _load_json(policy_dir / "on-policy-generations.json")
    records = policy_metadata.get("generations")
    if not isinstance(records, list):
        raise ValueError("策略运行缺少 on-policy generation 元数据")
    specs = [
        *(
            ("reference_expert", item, teacher_dir)
            for item in teacher_result.get("trajectories", [])
        ),
        *(("policy_sample", item, policy_dir) for item in policy_result.get("trajectories", [])),
    ]
    if (
        sum(role == "reference_expert" for role, _, _ in specs),
        sum(role == "policy_sample" for role, _, _ in specs),
    ) != (2, 6):
        raise ValueError("真实 execution 必须恰好包含 2 reference + 6 policy")
    reviewed: list[tuple[str, dict[str, Any], dict[str, Any], str, dict[str, Any]]] = []
    for role, summary, directory in specs:
        if not isinstance(summary, dict):
            raise ValueError("轨迹摘要格式无效")
        trajectory_path = Path(str(summary.get("trajectory_json", "")))
        if not trajectory_path.is_file():
            trajectory_path = directory / str(summary.get("trajectory_id")) / "trajectory.json"
        trajectory = _load_json(trajectory_path)
        review = review_trajectory(trajectory, summary, action_budget_ticks=action_budget_ticks)
        if not review.contract_valid:
            raise ValueError(f"轨迹 {review.trajectory_id} 审核失败：{review.issues}")
        control, context = _accepted_control(trajectory)
        reviewed.append((role, summary, review.to_dict(), control, context))
    comparisons = build_comparison_group(
        review_trajectory(
            _load_json(Path(str(summary["trajectory_json"]))),
            summary,
            action_budget_ticks=action_budget_ticks,
        )
        for _, summary, _, _, _ in reviewed
    )
    by_id = {item.trajectory_id: item for item in comparisons}
    trajectories = []
    for role, summary, review, control, context in reviewed:
        trajectory_id = str(summary["trajectory_id"])
        comparison = by_id[trajectory_id]
        item: dict[str, Any] = {
            "candidate_id": trajectory_id,
            "source_role": role,
            "action_text": control,
            "score": comparison.score,
            "relative_advantage": comparison.relative_advantage,
            "frames": _frame_paths(context, output),
            "provenance": {
                "observation_source": "craftground_rgb",
                "action_generator": "real_teacher"
                if role == "reference_expert"
                else "local_vision_model",
                "environment_execution": "craftground_real_ticks",
                "score_source": "interaction_trajectory_review_agents.review_trajectory",
                "executed_ticks": review["executed_ticks"],
                "baseline_world_sha256": baseline_hash,
            },
        }
        if role == "policy_sample":
            matching = [
                record
                for record in records
                if record.get("trajectory_id") == trajectory_id
                and record.get("response_text") == control
            ]
            if not matching:
                raise ValueError(f"策略轨迹 {trajectory_id} 缺少已接受动作的真实 token 记录")
            record = matching[0]
            token_ids = record.get("response_token_ids")
            logprobs = record.get("old_logprobs")
            if (
                not isinstance(token_ids, list)
                or not isinstance(logprobs, list)
                or len(token_ids) != len(logprobs)
                or not token_ids
                or any(not math.isfinite(float(value)) for value in logprobs)
            ):
                raise ValueError(f"策略轨迹 {trajectory_id} 的 token/logprob 未对齐")
            item.update(
                {
                    "response_token_ids": token_ids,
                    "old_logprobs": logprobs,
                    "policy_version": policy_metadata.get("policy_version"),
                    "sampling_parameters": record.get("sampling_parameters"),
                }
            )
        trajectories.append(item)
    payload = {
        "snapshot_id": f"baseline-world:{baseline_hash}",
        "contract": "real-craftground-2-reference-6-policy/v1",
        "trajectories": trajectories,
    }
    atomic_write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-budget-ticks", type=int, default=512)
    arguments = parser.parse_args()
    build_execution(
        arguments.teacher_dir,
        arguments.policy_dir,
        arguments.output,
        action_budget_ticks=arguments.action_budget_ticks,
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
