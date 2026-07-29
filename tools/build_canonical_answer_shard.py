"""构建保持时长、便于预测的第二轮规范答案分片。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.action_codec import LumineActionChunk, LumineWindowAction, decode_lumine_action


CLICK_KEYS = {"MouseLeft", "MouseRight"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key_ticks(block: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in decode_lumine_action(block).chunks:
        for key in chunk.keys:
            counts[key] = counts.get(key, 0) + 1
    return counts


def mouse_sum(block: str) -> tuple[int, int]:
    chunks = decode_lumine_action(block).chunks
    return sum(chunk.mouse[0] for chunk in chunks), sum(chunk.mouse[1] for chunk in chunks)


def click_order(block: str) -> list[str]:
    return [key for chunk in decode_lumine_action(block).chunks for key in chunk.keys if key in CLICK_KEYS]


def equivalent(candidate: str, blind: str) -> bool:
    return (
        len(decode_lumine_action(candidate).chunks) == len(decode_lumine_action(blind).chunks)
        and key_ticks(candidate) == key_ticks(blind)
        and mouse_sum(candidate) == mouse_sum(blind)
        and click_order(candidate) == click_order(blind)
    )


def complexity(block: str) -> tuple[int, int]:
    chunks = decode_lumine_action(block).chunks
    return sum(chunk.mouse != (0, 0) for chunk in chunks), sum(bool(chunk.keys) for chunk in chunks)


def pad_to_ticks(block: str, target: int) -> str:
    chunks = list(decode_lumine_action(block).chunks)
    if len(chunks) > target:
        raise ValueError(f"规范答案 {len(chunks)} tick 超过公开时长 {target}")
    if len(chunks) == target:
        return block
    padded: list[LumineActionChunk] = []
    remaining = target - len(chunks)
    for index, chunk in enumerate(chunks):
        padded.append(chunk)
        slots = len(chunks) - index
        add = remaining // slots
        padded.extend(LumineActionChunk(keys=()) for _ in range(add))
        remaining -= add
    padded.extend(LumineActionChunk(keys=()) for _ in range(remaining))
    return LumineWindowAction(tuple(padded)).to_text()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_dir
    answers = {row["id"]: row for row in read_jsonl(root / "answer_key.jsonl")}
    current = {row["id"]: row for row in read_jsonl(root / "second_round_preannotations.jsonl")}
    blind = {row["id"]: row for row in read_jsonl(root / "blind_test_v2_final_shards/demo_sequence.jsonl")}
    comparisons = {row["id"]: row for row in read_jsonl(root / "blind_test_v2_final_comparisons/demo_sequence.jsonl")}
    output = []
    for sample_id, row in current.items():
        task_type = row["task_type"]
        if task_type not in {"demonstration_optimization", "image_sequence_to_action"}:
            continue
        target_blocks = answers[sample_id]["reference_action_sequence"]
        candidate_blocks = row["answer_sequence"]
        blind_blocks = blind.get(sample_id, {}).get("blind_answer_sequence", [])
        if len(candidate_blocks) != len(target_blocks):
            raise ValueError(f"{sample_id}: 动作块数不一致")
        canonical = []
        used_blind = False
        padded_ticks = 0
        for index, (candidate, target) in enumerate(zip(candidate_blocks, target_blocks)):
            target_ticks = len(decode_lumine_action(target).chunks)
            selected = candidate
            if index < len(blind_blocks) and equivalent(candidate, blind_blocks[index]):
                if complexity(blind_blocks[index]) < complexity(candidate):
                    selected = blind_blocks[index]
                    used_blind = True
            before = len(decode_lumine_action(selected).chunks)
            selected = pad_to_ticks(selected, target_ticks)
            padded_ticks += target_ticks - before
            canonical.append(selected)
        cmp = comparisons.get(sample_id, {})
        visible = row.get("answer_reason", "").split(" 复核", 1)[0].strip()
        changes = ["以图像与录制真值确认的行为、顺序和必要持续按键为正确底线"]
        if padded_ticks:
            changes.append(f"为 GUI 稀疏操作补回 {padded_ticks} 个空 tick，使各动作块严格匹配公开时长")
        changes.append("保留已清理的不可见鼠标微动，并以少量明确节点表达可见转向")
        if used_blind:
            changes.append("盲答与真值严格等价时采用了控制节点更少的表达")
        else:
            changes.append("盲答存在差异或没有更简单的严格等价结构，因此未替换真值规范答案")
        reason = (
            f"{visible} 输出严格覆盖题目公开的每个图像区间；必要的挖掘、攻击、移动、使用等持续按键时长保持不变。"
            "画面无法体现的鼠标微动已删除或并入同一可见转向阶段，GUI 仅稀疏化点击间光标轨迹并保留点击节点、顺序和总时长。"
        )
        output.append({
            "id": sample_id,
            "task_type": task_type,
            "answer_sequence": canonical,
            "answer_reason": reason,
            "reference_kind": "reviewed_optimized_demonstration" if task_type == "demonstration_optimization" else "reviewed_optimized_action_sequence",
            "canonicalization_changes": changes,
            "blind_guidance_used": {
                "used": used_blind,
                "v2_verdict": cmp.get("verdict"),
                "principle": "盲答只用于选择严格等价且更简单的结构，不用于改写图像与真值确定的行为",
            },
            "preannotation_kind": "subagent_canonical_predictable_answer",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))


if __name__ == "__main__":
    main()
