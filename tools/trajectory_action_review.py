"""Gradio 第二轮审核：按题型审核 AI 优化后的最终动作回答。"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from datasets.action_codec import LumineActionChunk, LumineWindowAction, decode_lumine_action
from datasets.minestudio_data.load import TrajectoryReader
from tools.trajectory_human_review import TASK_LABELS, TASK_PROMPTS_ZH, format_reference_actions, read_jsonl


CLICK_KEYS = frozenset({"MouseLeft", "MouseRight"})
OUTPUT_REQUIREMENTS = {
    "demonstration_optimization": "输出覆盖整段演示、保留可见意图和因果顺序的清理动作序列。",
    "image_sequence_to_action": "输出能够解释全部输入图像状态变化的一种完整动作序列。",
    "history_to_future_action": "只输出从最后一张历史图延拓到唯一未来关键帧的动作序列。",
    "single_frame_intent_to_action": "只输出从当前图推进人工给定意图、并到达未来关键帧的动作序列。",
}


def _sum_mouse(chunks: list[LumineActionChunk]) -> tuple[int, int]:
    return sum(chunk.mouse[0] for chunk in chunks), sum(chunk.mouse[1] for chunk in chunks)


def _spread_chunks(keys: tuple[str, ...], mouse: tuple[int, int], count: int) -> list[LumineActionChunk]:
    """把合并后的鼠标总偏移分摊到少量等效控制 tick。"""
    count = max(1, count)
    result = []
    remaining_x, remaining_y = mouse
    for index in range(count):
        slots = count - index
        current_x = int(round(remaining_x / slots))
        current_y = int(round(remaining_y / slots))
        result.append(LumineActionChunk(keys=keys, mouse=(current_x, current_y)))
        remaining_x -= current_x
        remaining_y -= current_y
    return result


def compress_gameplay_chunks(chunks: list[LumineActionChunk]) -> tuple[list[LumineActionChunk], dict[str, int]]:
    """保留普通游戏持续按键时长，只合并同段内的鼠标偏移表达。"""
    output: list[LumineActionChunk] = []
    removed_empty = 0
    merged_mouse = 0
    compressed_held = 0
    index = 0
    while index < len(chunks):
        keys = chunks[index].keys
        end = index + 1
        while end < len(chunks) and chunks[end].keys == keys and not chunks[end].scroll:
            end += 1
        run = chunks[index:end]
        if not keys and all(chunk.mouse == (0, 0) for chunk in run):
            removed_empty += len(run)
            index = end
            continue
        representative_ticks = len(run) if keys else 1
        nonzero_mouse = sum(chunk.mouse != (0, 0) for chunk in run)
        merged_mouse += max(0, nonzero_mouse - 1)
        if keys:
            output.append(LumineActionChunk(keys=keys, mouse=_sum_mouse(run)))
            output.extend(LumineActionChunk(keys=keys) for _ in range(len(run) - 1))
        else:
            output.extend(_spread_chunks(keys, _sum_mouse(run), representative_ticks))
        index = end
    return output, {
        "removed_empty_ticks": removed_empty,
        "compressed_held_ticks": compressed_held,
        "merged_mouse_ticks": merged_mouse,
    }


def compress_gui_chunks(chunks: list[LumineActionChunk]) -> tuple[list[LumineActionChunk], dict[str, int]]:
    """把 GUI 点击之间的光标轨迹合并到点击节点，保留点击顺序。"""
    output: list[LumineActionChunk] = []
    pending: list[LumineActionChunk] = []
    merged_mouse = 0
    removed_empty = 0
    for chunk in chunks:
        click_keys = tuple(key for key in chunk.keys if key in CLICK_KEYS)
        other_keys = tuple(key for key in chunk.keys if key not in CLICK_KEYS)
        if click_keys or other_keys or chunk.scroll:
            trajectory = pending + [chunk]
            nonzero = sum(item.mouse != (0, 0) for item in trajectory)
            merged_mouse += max(0, nonzero - 1)
            removed_empty += sum(item.mouse == (0, 0) and not item.keys for item in pending)
            output.append(LumineActionChunk(
                keys=other_keys + click_keys, mouse=_sum_mouse(trajectory), scroll=chunk.scroll,
            ))
            pending = []
        else:
            pending.append(chunk)
    if pending and any(chunk.mouse != (0, 0) for chunk in pending):
        merged_mouse += max(0, sum(chunk.mouse != (0, 0) for chunk in pending) - 1)
        output.append(LumineActionChunk(keys=(), mouse=_sum_mouse(pending)))
    else:
        removed_empty += len(pending)
    return output, {
        "removed_empty_ticks": removed_empty,
        "compressed_held_ticks": 0,
        "merged_mouse_ticks": merged_mouse,
    }


def optimize_action_sequence(
    blocks: list[str], gui_flags: list[bool] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """结合 GUI 状态压缩真值动作，输出任务所需的动作块。"""
    decoded = [list(decode_lumine_action(block).chunks) for block in blocks]
    total_raw = sum(len(chunks) for chunks in decoded)
    flags = list(gui_flags or [False] * total_raw)
    if len(flags) != total_raw:
        flags = [False] * total_raw
    output: list[str] = []
    totals = {
        "raw_ticks": total_raw, "optimized_ticks": 0, "removed_empty_ticks": 0,
        "compressed_held_ticks": 0, "merged_mouse_ticks": 0, "gui_ticks": sum(flags),
    }
    offset = 0
    for chunks in decoded:
        block_flags = flags[offset:offset + len(chunks)]
        offset += len(chunks)
        compressor = compress_gui_chunks if block_flags and all(block_flags) else compress_gameplay_chunks
        optimized, stats = compressor(chunks)
        if not optimized:
            optimized = [LumineActionChunk(keys=())]
        output.append(LumineWindowAction(tuple(optimized)).to_text())
        totals["optimized_ticks"] += len(optimized)
        for key, value in stats.items():
            totals[key] += value
    return output, totals


def optimization_reason(stats: dict[str, int]) -> str:
    mode = "GUI 光标轨迹按点击节点合并" if stats["gui_ticks"] else "普通游戏按稳定按键语义合并"
    return (
        f"{mode}；真值动作由 {stats['raw_ticks']} tick 压缩为 {stats['optimized_ticks']} tick。"
        f"剔除 {stats['removed_empty_ticks']} 个空动作 tick，普通游戏持续按键未缩短，并合并 "
        f"{stats['merged_mouse_ticks']} 个中途鼠标偏移；保留动作时长、动作顺序、点击节点和累计鼠标方向。"
    )


class ActionReviewStore:
    def __init__(self, dataset_directory: Path, raw_dataset_directory: Path | None = None) -> None:
        self.root = Path(dataset_directory)
        all_questions = read_jsonl(self.root / "questions.jsonl")
        first_reviews = {row["id"]: row for row in read_jsonl(self.root / "question_reviews.jsonl")}
        if set(first_reviews) != {row["id"] for row in all_questions}:
            raise ValueError("第一轮题目审核尚未覆盖全部候选题")
        self.questions = [row for row in all_questions if first_reviews[row["id"]]["decision"] == "approve"]
        self.answers = {row["id"]: row for row in read_jsonl(self.root / "answer_key.jsonl")}
        self.candidate_path = self.root / "second_round_preannotations.jsonl"
        self.review_path = self.root / "action_reviews.jsonl"
        existing = {row["id"]: row for row in read_jsonl(self.candidate_path)}
        reader = None
        if raw_dataset_directory:
            reader = TrajectoryReader(
                [Path(raw_dataset_directory)], ["action", "meta_info"], 320, 180,
            )
        try:
            self.candidates = {}
            for question in self.questions:
                sample_id = question["id"]
                if sample_id in existing:
                    self.candidates[sample_id] = existing[sample_id]
                    continue
                answer = self.answers[sample_id]
                start, end = question["target_interval"]
                gui_flags = None
                if reader and question["source"]["episode"] in set(reader.episode_names()):
                    metadata = reader.readers["meta_info"].read_frames(
                        question["source"]["episode"], start, end - start,
                    )
                    gui_flags = [bool(item.get("isGuiOpen")) for item in metadata]
                sequence, stats = optimize_action_sequence(answer["reference_action_sequence"], gui_flags)
                self.candidates[sample_id] = {
                    "id": sample_id, "task_type": question["task_type"],
                    "answer_sequence": sequence, "answer_reason": optimization_reason(stats),
                    "optimization_stats": stats, "ai_decision": "approve",
                    "preannotation_kind": "ai_second_round_answer_preannotation",
                }
        finally:
            if reader:
                reader.close()
        self.reviews = {row["id"]: row for row in read_jsonl(self.review_path)}
        blind_paths = [
            self.root / "blind_test_v2_final_shards" / "demo_sequence.jsonl",
            self.root / "blind_test_v2_final_shards" / "future_single.jsonl",
        ]
        comparison_paths = [
            self.root / "blind_test_v2_final_comparisons" / "demo_sequence.jsonl",
            self.root / "blind_test_v2_final_comparisons" / "future_single.jsonl",
        ]
        self.blind_answers = {
            row["id"]: row for path in blind_paths for row in read_jsonl(path)
        }
        self.blind_comparisons = {
            row["id"]: row for path in comparison_paths for row in read_jsonl(path)
        }
        self.index_by_id = {row["id"]: index for index, row in enumerate(self.questions)}
        self._lock = threading.Lock()
        self._write(self.candidate_path, [self.candidates[row["id"]] for row in self.questions])

    @staticmethod
    def _write(path: Path, records: list[dict[str, Any]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def parse_sequence(text: str) -> list[str]:
        try:
            blocks = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"回答必须是 JSON 数组：{error.msg}") from error
        if not isinstance(blocks, list) or not blocks or not all(isinstance(block, str) for block in blocks):
            raise ValueError("回答必须是非空动作字符串 JSON 数组")
        return [decode_lumine_action(block).to_text() for block in blocks]

    def save(self, sample_id: str, answer_text: str, decision: str, reason: str) -> None:
        if decision not in {"approve", "reject"}:
            raise ValueError("请选择批准或否决")
        concise_reason = reason.strip()
        if not concise_reason:
            raise ValueError("请填写题目与回答的审核理由")
        sequence = self.parse_sequence(answer_text)
        question = self.questions[self.index_by_id[sample_id]]
        expected_blocks = len(self.answers[sample_id]["reference_action_sequence"])
        if len(sequence) != expected_blocks:
            raise ValueError(f"该题型需要 {expected_blocks} 个动作块，当前为 {len(sequence)} 个")
        with self._lock:
            self.reviews[sample_id] = {
                "id": sample_id, "decision": decision, "reason": concise_reason,
                "review_kind": "human_second_round_answer_review",
                "reviewed_answer_sequence": sequence,
                "reference_kind": (
                    "reviewed_optimized_demonstration"
                    if question["task_type"] == "demonstration_optimization"
                    else "reviewed_optimized_action_sequence"
                ),
            }
            self._write(self.review_path, [
                self.reviews[row["id"]] for row in self.questions if row["id"] in self.reviews
            ])

    def counts(self) -> tuple[int, int, int]:
        approved = sum(row["decision"] == "approve" for row in self.reviews.values())
        rejected = sum(row["decision"] == "reject" for row in self.reviews.values())
        return approved, rejected, len(self.questions) - approved - rejected


def build_interface(store: ActionReviewStore) -> gr.Blocks:
    def answer_preview(answer_text: str, reason: str) -> str:
        try:
            sequence = store.parse_sequence(answer_text)
            rendered = json.dumps(sequence, ensure_ascii=False, indent=2)
            validation = "✅ 动作 JSON 可以解析。"
        except ValueError as error:
            rendered = answer_text
            validation = f"❌ {error}"
        return (
            "### 当前编辑后的最终回答\n\n"
            f"```json\n{rendered}\n```\n\n"
            f"**回答理由：** {reason.strip() or '尚未填写'}\n\n{validation}"
        )

    def progress() -> str:
        approved, rejected, pending = store.counts()
        return f"**第二轮进度：{approved + rejected} / {len(store.questions)}**　批准 {approved}　否决 {rejected}　待审 {pending}"

    def render(index: int) -> tuple[Any, ...]:
        index = max(0, min(int(index), len(store.questions) - 1))
        question = store.questions[index]
        sample_id = question["id"]
        frames = question["source"]["image_frames"]
        images = [(str(store.root / path), f"输入图 {i + 1} · 帧 {frame}") for i, (path, frame) in enumerate(zip(question["images"], frames))]
        references = question.get("review_reference_images", [])
        reference_frames = question["source"].get("review_reference_frames", [])
        if len(references) == 2 and len(reference_frames) == 2:
            images.append((str(store.root / references[1]), f"未来关键帧 · 帧 {reference_frames[1]}"))
        truth = store.answers[sample_id]["reference_action_sequence"]
        candidate = store.candidates[sample_id]
        review = store.reviews.get(sample_id, {})
        sequence = review.get("reviewed_answer_sequence", candidate["answer_sequence"])
        reason = review.get("reason", candidate["answer_reason"])
        task_text = "\n\n".join([
            f"### {TASK_LABELS[question['task_type']]}", TASK_PROMPTS_ZH[question["task_type"]],
            f"**本题输出要求：** {OUTPUT_REQUIREMENTS[question['task_type']]}",
            f"**AI 优化依据：** {candidate['answer_reason']}",
        ])
        blind = store.blind_answers.get(sample_id)
        comparison = store.blind_comparisons.get(sample_id)
        if blind and comparison:
            score_text = "　".join(
                f"{name} {float(comparison['scores'][name]):.2f}"
                for name in ("action_type", "duration", "ordering", "mouse_or_gui")
            )
            blind_text = "\n\n".join([
                "### 最终新版 Prompt 无答案泄露盲答参考",
                "```json\n" + json.dumps(blind["blind_answer_sequence"], ensure_ascii=False, indent=2) + "\n```",
                f"**盲答自评：** {blind['blind_reason']}",
                f"**后验判定：** `{comparison['verdict']}`　{score_text}",
                f"**比较摘要：** {comparison['summary']}",
                f"**关键差异：** {comparison['key_differences']}",
                f"**调整建议：** {comparison['recommendation']}",
                "该内容只作参考，不会覆盖手写答案或人工修改。",
            ])
        else:
            blind_text = "### 最终新版 Prompt 无答案泄露盲答参考\n\n当前题目没有可用的盲测结果。"
        preview_text = answer_preview(json.dumps(sequence, ensure_ascii=False, indent=2), reason)
        return (
            index, sample_id, f"## {sample_id}\n第 **{index + 1} / {len(store.questions)}** 道",
            images, task_text, format_reference_actions(truth, unoptimized=True), blind_text,
            preview_text, json.dumps(sequence, ensure_ascii=False, indent=2),
            review.get("decision"), reason, progress(),
        )

    def save(index: int, answer_text: str, decision: str, reason: str) -> tuple[str, str]:
        sample_id = store.questions[int(index)]["id"]
        try:
            store.save(sample_id, answer_text, decision, reason)
        except ValueError as error:
            return f"❌ {error}", progress()
        return f"✅ 已保存 `{sample_id}` 的题目与回答审核。", progress()

    def navigate(index: int, offset: int) -> tuple[Any, ...]:
        return render(int(index) + offset)

    with gr.Blocks(title="MineStudio 第二轮回答审核") as interface:
        gr.Markdown("# 第二轮：题目与最终回答审核\n结合完整图像轨迹和真值动作，审核 AI 给出的压缩动作回答；可以直接修改后批准，或填写理由否决。")
        index_state = gr.State(0)
        with gr.Row():
            previous = gr.Button("← 上一题")
            next_button = gr.Button("下一题 →", variant="primary")
            selector = gr.Dropdown(choices=[row["id"] for row in store.questions], value=store.questions[0]["id"], label="跳转到题目", filterable=True)
        title = gr.Markdown()
        gallery = gr.Gallery(label="完整图像轨迹", columns=6, height="auto", object_fit="contain")
        task = gr.Markdown()
        truth = gr.Markdown(label="真值动作")
        blind_reference = gr.Markdown()
        gr.Markdown("## 人工可编辑的最终回答")
        gr.Markdown("直接修改下方动作 JSON 和回答理由；上方预览会实时同步。")
        preview = gr.Markdown()
        answer = gr.Textbox(
            label="动作序列 JSON（可直接编辑）",
            placeholder='例如：["<|action_start|> ; W ; Mouse 4 2 W <|action_end|>"]',
            lines=20,
            max_lines=40,
            interactive=True,
        )
        decision = gr.Radio(choices=[("批准（包括修改后批准）", "approve"), ("否决", "reject")], label="审核决定")
        reason = gr.Textbox(
            label="回答理由（可直接编辑）",
            placeholder="说明图像证据、真值依据，以及保留、删除或合并了哪些动作。",
            lines=4,
            interactive=True,
        )
        with gr.Row():
            save_button = gr.Button("保存审核", variant="primary")
            save_next = gr.Button("保存并下一题")
        status = gr.Markdown()
        progress_text = gr.Markdown()
        outputs = [
            index_state, selector, title, gallery, task, truth, blind_reference,
            preview, answer, decision, reason, progress_text,
        ]
        interface.load(render, gr.State(0), outputs)
        previous.click(lambda index: navigate(index, -1), index_state, outputs)
        next_button.click(lambda index: navigate(index, 1), index_state, outputs)
        selector.change(lambda sample_id: render(store.index_by_id[sample_id]), selector, outputs)
        answer.change(answer_preview, [answer, reason], preview)
        reason.change(answer_preview, [answer, reason], preview)
        inputs = [index_state, answer, decision, reason]
        save_button.click(save, inputs, [status, progress_text])
        save_next.click(save, inputs, [status, progress_text]).then(lambda index: navigate(index, 1), index_state, outputs)
    return interface


def main() -> None:
    parser = argparse.ArgumentParser(description="MineStudio 第二轮题目与回答审核")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dataset-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    arguments = parser.parse_args()
    store = ActionReviewStore(arguments.dataset_dir, arguments.raw_dataset_dir)
    build_interface(store).launch(server_name=arguments.host, server_port=arguments.port, share=arguments.share, allowed_paths=[str(store.root.resolve())])


if __name__ == "__main__":
    main()
