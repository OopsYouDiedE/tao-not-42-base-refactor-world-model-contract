"""测试模型对三类生成题的作答格式，并计算与参考示范的诊断相似度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets.action_codec import LumineWindowAction, decode_lumine_action
from datasets.minestudio_finetune.review_questions import read_jsonl


def parse_answer(value: Any, expected_blocks: int) -> list[LumineWindowAction]:
    """解析 JSON 动作块列表并执行严格数量检查。"""
    if not isinstance(value, list) or len(value) != expected_blocks:
        raise ValueError(f"答案必须是恰好包含 {expected_blocks} 个动作块的 JSON 数组")
    if not all(isinstance(block, str) for block in value):
        raise ValueError("动作块必须是字符串")
    for block in value:
        if block.count("<|action_start|>") != 1 or block.count("<|action_end|>") != 1:
            raise ValueError("每个动作块必须恰好包含一对 action 标记")
    parsed = [decode_lumine_action(block) for block in value]
    if any(not action.chunks for action in parsed):
        raise ValueError("每个动作块必须至少包含一个 chunk")
    return parsed


def action_similarity(candidate: LumineWindowAction, reference: LumineWindowAction) -> float:
    """返回 0 到 1 的诊断分；按键集合占 80%，鼠标距离占 20%。"""
    chunk_scores: list[float] = []
    maximum_chunks = max(len(candidate.chunks), len(reference.chunks))
    for index in range(maximum_chunks):
        candidate_chunk = candidate.chunks[index] if index < len(candidate.chunks) else None
        reference_chunk = reference.chunks[index] if index < len(reference.chunks) else None
        if candidate_chunk is None or reference_chunk is None:
            chunk_scores.append(0.0)
            continue
        left, right = set(candidate_chunk.keys), set(reference_chunk.keys)
        key_similarity = 1.0 if not left and not right else len(left & right) / len(left | right)
        mouse_distance = abs(candidate_chunk.mouse[0] - reference_chunk.mouse[0]) + abs(
            candidate_chunk.mouse[1] - reference_chunk.mouse[1]
        )
        mouse_similarity = max(0.0, 1.0 - mouse_distance / 200.0)
        chunk_scores.append(0.8 * key_similarity + 0.2 * mouse_similarity)
    key_score = sum(chunk_scores) / len(chunk_scores)
    return round(key_score, 6)


def evaluate_responses(
    questions: list[dict[str, Any]],
    answer_key: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按题号评估回答。合理动作多解，相似度只用于发现离群答案。"""
    references = {record["id"]: record for record in answer_key}
    submitted = {record["id"]: record for record in responses}
    results: list[dict[str, Any]] = []
    for question in questions:
        sample_id = question["id"]
        reference_values = references[sample_id]["reference_action_sequence"]
        response = submitted.get(sample_id)
        if response is None:
            results.append({"id": sample_id, "format_valid": False, "error": "missing_response"})
            continue
        try:
            candidate = parse_answer(response.get("answer"), len(reference_values))
            reference = parse_answer(reference_values, len(reference_values))
        except (TypeError, ValueError) as error:
            results.append({"id": sample_id, "format_valid": False, "error": str(error)})
            continue
        scores = [action_similarity(left, right) for left, right in zip(candidate, reference)]
        results.append({
            "id": sample_id,
            "task_type": question["task_type"],
            "format_valid": True,
            "reference_similarity": round(sum(scores) / len(scores), 6),
            "requires_semantic_review": True,
            "note": "参考示范不是唯一合理答案；相似度不能单独决定正确性或训练准入。",
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="测试模型对 MineStudio 轨迹题的作答")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    response_source = parser.add_mutually_exclusive_group(required=True)
    response_source.add_argument("--responses", type=Path)
    response_source.add_argument(
        "--reference-replay", action="store_true",
        help="把隔离答案转换为模拟回答，只用于验证协议解析和评测链路",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    answer_key = read_jsonl(arguments.dataset_dir / "answer_key.jsonl")
    responses = (
        [
            {"id": record["id"], "answer": record["reference_action_sequence"]}
            for record in answer_key
        ]
        if arguments.reference_replay
        else read_jsonl(arguments.responses)
    )
    results = evaluate_responses(
        read_jsonl(arguments.dataset_dir / "questions.jsonl"),
        answer_key,
        responses,
    )
    with arguments.output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({
        "responses": len(results),
        "format_valid": sum(record["format_valid"] for record in results),
        "output": str(arguments.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
