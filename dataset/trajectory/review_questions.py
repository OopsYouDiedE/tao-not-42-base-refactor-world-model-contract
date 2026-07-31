"""Gradio 人工审核界面：逐题查看完整轨迹并保存审核结论。"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ModuleNotFoundError:
    gr = None  # type: ignore[assignment]
import numpy as np
from PIL import Image

from dataset.minestudio.reader import TrajectoryReader
from dataset.trajectory.generate_questions import (
    AI_REVIEW_PROMPT,
    HUMAN_REVIEW_PROMPT,
    action_node_frames,
    normalize_gui_clicks,
)
from lumine.action_codec import encode_lumine_action

TASK_LABELS = {
    "demonstration_optimization": "演示优化",
    "image_sequence_to_action": "图像序列反推动作",
    "history_to_future_action": "历史图像预测动作",
    "single_frame_intent_to_action": "单帧意图转动作",
}
TASK_PROMPTS_ZH = {
    "demonstration_optimization": (
        "任务推断内容：根据完整图像序列和首尾图之间的原始动作，判断演示在做什么、轨迹是否"
        "连贯，以及后续是否值得清理成更紧凑的标准动作。此题不预测图像序列之外的未来。"
    ),
    "image_sequence_to_action": (
        "任务推断内容：根据已经给出的完整图像序列，反推从第一张图变化到最后一张图所执行的"
        "动作。此题解释已发生的变化，不延拓未来。"
    ),
    "history_to_future_action": (
        "任务推断内容：根据全部历史图像，推断从最后一张历史图延拓到唯一未来关键帧之间的动作。"
    ),
    "single_frame_intent_to_action": (
        "任务推断内容：根据当前单帧和人工填写的意图，推断从当前帧延拓到唯一未来关键帧之间的动作。"
    ),
}
TOKEN_EXPLANATIONS = {
    "W": "前进",
    "S": "后退",
    "A": "向左移动",
    "D": "向右移动",
    "space": "跳跃",
    "shift": "潜行",
    "ctrl": "疾跑",
    "MouseLeft": "持续挖掘或攻击",
    "MouseRight": "使用或放置物品",
    "Inventory": "打开或关闭物品栏",
    "Drop": "丢弃物品",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def translate_intent(intent: str) -> str:
    replacements = {
        "hold the primary action to mine or attack the visible target": (
            "持续挖掘或攻击画面中的目标"
        ),
        "use or place the selected item": "使用或放置当前物品",
        "move forward": "向前移动",
        "move back": "向后移动",
        "move left": "向左移动",
        "move right": "向右移动",
        "jump": "跳跃",
        "sprint": "疾跑",
        "sneak": "潜行",
        "open or close the inventory": "打开或关闭物品栏",
        "drop the selected item": "丢弃当前物品",
        "adjust the view or GUI cursor toward the target": "调整视角或 GUI 光标位置",
        "; then ": "；然后",
    }
    translated = intent
    for source, target in replacements.items():
        translated = translated.replace(source, target)
    return translated


def summarize_actions(blocks: list[str], unoptimized: bool = False) -> str:
    text = " ".join(blocks)
    actions = [description for token, description in TOKEN_EXPLANATIONS.items() if token in text]
    mouse_moves = text.count("Mouse ")
    ticks = sum(block.count(";") for block in blocks)
    parts = [f"动作覆盖约 {ticks} 个 50 毫秒 tick"]
    if actions:
        parts.append("主要行为包括" + "、".join(actions))
    if mouse_moves:
        parts.append(f"包含 {mouse_moves} 次视角或光标位移")
    if unoptimized:
        parts.append("该序列仍是原始录制，只能审核是否值得保留，不能视为优化结果")
    return "；".join(parts) + "。"


def format_reference_actions(
    blocks: list[str],
    node_frames: list[int] | None = None,
    unoptimized: bool = False,
) -> str:
    """将完整参考动作逐块展示，便于人工核对每个 tick。"""
    lines = ["### 完整参考动作序列", summarize_actions(blocks, unoptimized)]
    for index, block in enumerate(blocks, 1):
        block_label = f"动作块 {index}"
        if node_frames is not None and index < len(node_frames):
            block_start = node_frames[index - 1]
            block_end = node_frames[index]
            block_label += (
                f" · 图像帧 {block_start} → {block_end} · 序列长度 {block_end - block_start} tick"
            )
        lines.extend([f"**{block_label}**", f"```text\n{block}\n```"])
    return "\n\n".join(lines)


class ReviewStore:
    """加载候选题并原子保存人工审核结果。"""

    def __init__(
        self,
        dataset_directory: Path,
        review_path: Path | None = None,
        raw_dataset_directory: Path | None = None,
        preannotation_path: Path | None = None,
    ) -> None:
        self.root = Path(dataset_directory)
        self.review_path = (
            Path(review_path) if review_path else self.root / "question_reviews.jsonl"
        )
        self.questions = read_jsonl(self.root / "questions.jsonl")
        self.answers = {
            record["id"]: record for record in read_jsonl(self.root / "answer_key.jsonl")
        }
        if not self.questions:
            raise ValueError(f"{self.root} 没有候选题")
        self.index_by_id = {record["id"]: index for index, record in enumerate(self.questions)}
        if len(self.index_by_id) != len(self.questions):
            raise ValueError("questions.jsonl 包含重复 ID")
        missing_answers = set(self.index_by_id) - set(self.answers)
        if missing_answers:
            raise ValueError(f"缺少参考答案：{sorted(missing_answers)[:5]}")
        self.reviews = {record["id"]: record for record in read_jsonl(self.review_path)}
        resolved_preannotation_path = (
            Path(preannotation_path)
            if preannotation_path
            else self.root / "ai_question_preannotations.jsonl"
        )
        self.preannotations = {
            record["id"]: record for record in read_jsonl(resolved_preannotation_path)
        }
        self._lock = threading.Lock()
        self.raw_reader = (
            TrajectoryReader(
                [Path(raw_dataset_directory)],
                ["action", "image", "meta_info"],
                320,
                180,
            )
            if raw_dataset_directory
            else None
        )
        self.raw_episodes = set(self.raw_reader.episode_names()) if self.raw_reader else set()
        if self.raw_reader is not None:
            self.ensure_review_references()

    def default_review(self, sample_id: str) -> dict[str, Any]:
        return {
            "id": sample_id,
            "decision": "pending",
            "reasons": [],
            "review_kind": "human_question_review",
        }

    def review(self, sample_id: str) -> dict[str, Any]:
        return {**self.default_review(sample_id), **self.reviews.get(sample_id, {})}

    def displayed_review(self, sample_id: str) -> dict[str, Any]:
        if sample_id in self.reviews:
            return self.review(sample_id)
        return {**self.default_review(sample_id), **self.preannotations.get(sample_id, {})}

    def save(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.reviews[record["id"]] = record
            ordered = [
                self.reviews[question["id"]]
                for question in self.questions
                if question["id"] in self.reviews
            ]
            self.review_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.review_path.with_suffix(self.review_path.suffix + ".tmp")
            temporary.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
                encoding="utf-8",
            )
            temporary.replace(self.review_path)

    def counts(self) -> dict[str, int]:
        counts = {"approve": 0, "reject": 0, "revise": 0, "pending": 0}
        for question in self.questions:
            decision = self.review(question["id"]).get("decision", "pending")
            counts[decision if decision in counts else "pending"] += 1
        return counts

    def can_adjust(self, sample_id: str) -> bool:
        question = self.questions[self.index_by_id[sample_id]]
        return question["source"]["episode"] in self.raw_episodes

    def close(self) -> None:
        if self.raw_reader is not None:
            self.raw_reader.close()

    @staticmethod
    def reference_frames(question: dict[str, Any]) -> list[int]:
        start = question["source"]["image_frames"][-1]
        end = question["target_interval"][1]
        if end <= start:
            raise ValueError("未来关键帧必须晚于最后一张输入图")
        return [start, end]

    def refresh_review_references(self, question: dict[str, Any]) -> None:
        if question["task_type"] not in {
            "single_frame_intent_to_action",
            "history_to_future_action",
        }:
            question.pop("review_reference_images", None)
            question["source"].pop("review_reference_frames", None)
            return
        if not self.raw_reader or not self.can_adjust(question["id"]):
            return
        frames = self.reference_frames(question)
        reference_directory = self.root / "review_images"
        reference_directory.mkdir(parents=True, exist_ok=True)
        end_relative = f"review_images/{question['id']}_end.jpg"
        array = self.raw_reader.readers["image"].read_frames(
            question["source"]["episode"],
            frames[1],
            1,
        )[0]
        Image.fromarray(np.asarray(array, dtype=np.uint8)).save(
            self.root / end_relative,
            quality=95,
        )
        question["review_reference_images"] = [
            question["images"][-1],
            end_relative,
        ]
        question["source"]["review_reference_frames"] = frames

    def ensure_review_references(self) -> None:
        changed = False
        for question in self.questions:
            if question["task_type"] in {
                "single_frame_intent_to_action",
                "history_to_future_action",
            } and self.can_adjust(question["id"]):
                expected_frames = self.reference_frames(question)
                current_frames = question["source"].get("review_reference_frames")
                paths = question.get("review_reference_images", [])
                if (
                    current_frames != expected_frames
                    or len(paths) != 2
                    or any(not (self.root / relative).is_file() for relative in paths)
                ):
                    self.refresh_review_references(question)
                    changed = True
        if changed:
            self._rewrite_dataset_files()

    def save_adjustment(
        self,
        sample_id: str,
        image_frames: list[int],
        target_end: int | None,
        rewrite_dataset: bool = True,
    ) -> None:
        if not self.raw_reader or not self.can_adjust(sample_id):
            raise ValueError("这道题没有可用的原始 LMDB，不能向前后重新取帧")
        question = self.questions[self.index_by_id[sample_id]]
        answer = self.answers[sample_id]
        if len(image_frames) != len(question["images"]):
            raise ValueError(f"图像帧必须保持 {len(question['images'])} 个")
        if any(right <= left for left, right in zip(image_frames, image_frames[1:], strict=False)):
            raise ValueError("图片帧必须严格递增")
        task_type = question["task_type"]
        if task_type == "image_sequence_to_action" or task_type == "demonstration_optimization":
            action_interval = [image_frames[0], image_frames[-1]]
        else:
            action_start = image_frames[-1]
            if target_end is None or int(target_end) <= action_start:
                raise ValueError("未来结束帧必须大于固定的动作起始帧")
            action_interval = [action_start, int(target_end)]
        if action_interval[1] - action_interval[0] > 64:
            raise ValueError("单题动作区间不能超过 64 帧")
        episode = question["source"]["episode"]
        episode_length = self.raw_reader.episode_length(episode)
        if any(frame < 0 or frame >= episode_length for frame in image_frames):
            raise ValueError(f"图像帧必须位于 0 到 {episode_length - 1}")
        if action_interval[0] < 0 or action_interval[1] > episode_length:
            raise ValueError(f"动作区间必须位于 0 到 {episode_length}")
        new_images = [
            self.raw_reader.readers["image"].read_frames(episode, frame, 1)[0]
            for frame in image_frames
        ]
        nodes = action_node_frames(task_type, image_frames, action_interval)
        actions = self.raw_reader.readers["action"].read_frames(
            episode,
            action_interval[0],
            action_interval[1] - action_interval[0],
        )
        metadata = self.raw_reader.readers["meta_info"].read_frames(
            episode,
            action_interval[0],
            action_interval[1] - action_interval[0],
        )
        normalized = normalize_gui_clicks(actions, metadata)
        blocks = []
        for left, right in zip(nodes, nodes[1:], strict=False):
            start_offset = left - action_interval[0]
            end_offset = right - action_interval[0]
            block = {
                field: np.asarray(values)[start_offset:end_offset]
                for field, values in normalized.items()
            }
            blocks.append(encode_lumine_action(block).to_text())
        with self._lock:
            for relative, array in zip(question["images"], new_images, strict=True):
                Image.fromarray(np.asarray(array, dtype=np.uint8)).save(
                    self.root / relative, quality=95
                )
            question["source"]["image_frames"] = image_frames
            question["target_interval"] = action_interval
            if task_type == "demonstration_optimization":
                question["inputs"]["raw_action_sequence"] = blocks
            answer["reference_action_sequence"] = blocks
            answer["source"]["action_start_frame"] = action_interval[0]
            answer["source"]["action_end_frame"] = action_interval[1]
            self.refresh_review_references(question)
            if rewrite_dataset:
                self._rewrite_dataset_files()

    def save_intent(self, sample_id: str, intent: str) -> None:
        question = self.questions[self.index_by_id[sample_id]]
        if question["task_type"] != "single_frame_intent_to_action":
            return
        normalized = intent.strip()
        question["inputs"]["intent"] = normalized
        question["inputs"]["intent_status"] = (
            "human_authored" if normalized else "pending_human_authoring"
        )
        with self._lock:
            self._rewrite_dataset_files()

    def _rewrite_dataset_files(self) -> None:
        self._atomic_write(self.root / "questions.jsonl", self.questions)
        ordered_answers = [self.answers[question["id"]] for question in self.questions]
        self._atomic_write(self.root / "answer_key.jsonl", ordered_answers)
        self._atomic_write(
            self.root / "ai_review_requests.jsonl",
            [
                {
                    "id": question["id"],
                    "system": AI_REVIEW_PROMPT,
                    "question": question,
                    "answer": self.answers[question["id"]],
                    "output_contract": {
                        "decision": "approve_or_reject",
                        "reason": "one_concise_string",
                    },
                }
                for question in self.questions
            ],
        )
        self._atomic_write(
            self.root / "human_review_templates.jsonl",
            [
                {
                    "id": question["id"],
                    "instructions": HUMAN_REVIEW_PROMPT,
                    "question": question,
                    "answer": self.answers[question["id"]],
                    "decision": "pending",
                    "reason": "",
                }
                for question in self.questions
            ],
        )

    @staticmethod
    def _atomic_write(path: Path, records: list[dict[str, Any]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(path)


def build_interface(store: ReviewStore) -> gr.Blocks:
    if gr is None:
        raise ModuleNotFoundError("启动人工审核界面需要安装 review 可选依赖 gradio")
    """构建第一阶段题目审核界面，并展示完整轨迹与参考动作。"""

    def render(index: int) -> tuple[Any, ...]:
        index = max(0, min(int(index), len(store.questions) - 1))
        question = store.questions[index]
        sample_id = question["id"]
        review = store.displayed_review(sample_id)
        frames = question["source"]["image_frames"]
        task_type = question["task_type"]
        input_kind = "历史输入" if task_type == "history_to_future_action" else "模型输入"
        images = [
            (str(store.root / relative), f"{input_kind} {image_index + 1} · 帧 {frame}")
            for image_index, (relative, frame) in enumerate(
                zip(question["images"], frames, strict=True)
            )
        ]
        details = [
            f"### {TASK_LABELS[question['task_type']]}",
            TASK_PROMPTS_ZH[question["task_type"]],
        ]
        if sample_id not in store.reviews and sample_id in store.preannotations:
            preannotation = store.preannotations[sample_id]
            suggested = "通过" if preannotation.get("decision") == "approve" else "不通过"
            details.extend(
                [
                    "### AI 预标注",
                    f"建议：**{suggested}**。动作类型："
                    f"`{preannotation.get('action_class', 'unknown')}`。"
                    "该结果尚未计入人工审核，点击保存后才会成为人工结论。",
                ]
            )
        intent = question.get("inputs", {}).get("intent")
        if not intent and sample_id not in store.reviews and sample_id in store.preannotations:
            intent = store.preannotations[sample_id].get("suggested_intent", "")
        if intent:
            details.extend(["### 给定意图", translate_intent(intent)])
        details.append(f"**来源 episode：** `{question['source']['episode']}`")
        can_adjust = store.can_adjust(sample_id)
        interval = question["target_interval"]
        if task_type == "image_sequence_to_action":
            interval_text = f"动作区间自动绑定第一张和最后一张图片：`[{frames[0]}, {frames[-1]}]`"
        elif task_type == "single_frame_intent_to_action":
            interval_text = f"动作起点固定为观察帧 {frames[0]}，只可调整结束帧。"
        elif task_type == "history_to_future_action":
            interval_text = f"未来动作起点固定为最后一张历史图 {frames[-1]}，只可调整结束帧。"
        else:
            interval_text = (
                f"演示动作区间自动绑定第一张和最后一张图片：`[{frames[0]}, {frames[-1]}]`。"
            )
        frame_updates = [
            gr.update(
                value=frames[position] if position < len(frames) else 0,
                visible=position < len(frames),
                interactive=can_adjust,
            )
            for position in range(5)
        ]
        reference_frames = question["source"].get("review_reference_frames", [])
        reference_images = question.get("review_reference_images", [])
        future_gallery = (
            [(str(store.root / reference_images[1]), f"未来关键帧 · 帧 {reference_frames[1]}")]
            if len(reference_frames) == 2 and len(reference_images) == 2
            else []
        )
        display_sequence = images + future_gallery
        reference_actions = store.answers[sample_id]["reference_action_sequence"]
        action_nodes = action_node_frames(task_type, frames, interval)
        return (
            index,
            sample_id,
            f"## {sample_id}\n第 **{index + 1} / {len(store.questions)}** 道",
            display_sequence,
            *frame_updates,
            gr.update(
                value=interval[1],
                visible=task_type in {"history_to_future_action", "single_frame_intent_to_action"},
                interactive=can_adjust,
            ),
            interval_text,
            "✅ 可从 7xx 原始数据重新取帧。"
            if can_adjust
            else "ℹ️ 缺少对应原始 LMDB，当前题目只读。",
            "\n\n".join(details),
            format_reference_actions(
                reference_actions,
                node_frames=action_nodes,
                unoptimized=task_type == "demonstration_optimization",
            ),
            gr.update(
                value=intent or "",
                visible=task_type == "single_frame_intent_to_action",
                interactive=task_type == "single_frame_intent_to_action",
            ),
            review.get("decision") if review.get("decision") in {"approve", "reject"} else None,
            review.get("reason", ""),
            progress_markdown(store),
        )

    def save_review(index: int, intent: str, decision: str, reason: str) -> tuple[str, str]:
        question = store.questions[int(index)]
        if decision not in {"approve", "reject"}:
            return "❌ 请选择通过或不通过。", progress_markdown(store)
        concise_reason = str(reason).strip()
        if decision == "approve" and not concise_reason:
            concise_reason = "题目图片清楚，帧序列和预测区间设置合理。"
        if decision == "reject" and not concise_reason:
            return "❌ 不通过时请填写简短理由。", progress_markdown(store)
        if (
            question["task_type"] == "single_frame_intent_to_action"
            and decision == "approve"
            and not str(intent).strip()
        ):
            return "❌ 单帧意图题通过前，必须填写人工意图。", progress_markdown(store)
        store.save_intent(question["id"], str(intent))
        store.save(
            {
                "id": question["id"],
                "decision": decision,
                "reason": concise_reason,
                "reasons": [concise_reason],
                "review_kind": "human_question_review",
                "image_frames": question["source"]["image_frames"],
                "target_interval": question["target_interval"],
            }
        )
        return f"✅ 已保存题目审核 `{question['id']}`。", progress_markdown(store)

    def adjust_frames(index: int, *values: Any) -> tuple[Any, ...]:
        question = store.questions[int(index)]
        frame_values = values[:5]
        target_end = values[5]
        frames = [int(frame_values[position]) for position in range(len(question["images"]))]
        try:
            store.save_adjustment(
                question["id"], frames, int(target_end) if target_end is not None else None
            )
        except ValueError as error:
            return (*render(index), f"❌ 调整失败：{error}")
        return (*render(index), "✅ 已重新取图，并按题型规则同步更新动作区间。")

    def navigate(index: int, offset: int) -> tuple[Any, ...]:
        return render(int(index) + offset)

    def jump(sample_id: str) -> tuple[Any, ...]:
        return render(store.index_by_id[sample_id])

    def fill_reason(decision: str, current: str) -> str:
        if decision == "approve" and not current.strip():
            return "题目图片清楚，帧序列和预测区间设置合理。"
        return current

    with gr.Blocks(title="MineStudio 轨迹题目审核") as interface:
        gr.Markdown(
            "# 第一阶段：题目审核\n只审核题目、图片帧与预测区间。标准答案将在题目通过后另行生成。"
        )
        index_state = gr.State(0)
        with gr.Row():
            previous = gr.Button("← 上一题")
            next_button = gr.Button("下一题 →", variant="primary")
            sample_selector = gr.Dropdown(
                choices=[question["id"] for question in store.questions],
                value=store.questions[0]["id"],
                label="跳转到题目",
                filterable=True,
                scale=4,
            )
        title = gr.Markdown()
        gallery = gr.Gallery(
            label="完整展示序列（延拓题只额外包含一张未来关键帧；点击查看大图）",
            columns=6,
            height="auto",
            object_fit="contain",
        )
        with gr.Row():
            frame_inputs = [
                gr.Number(label=f"图 {position + 1} 帧号", precision=0, step=1)
                for position in range(5)
            ]
        with gr.Row():
            target_end = gr.Number(label="未来最终帧", precision=0, step=1)
            apply_adjustment = gr.Button("应用帧调整", variant="secondary")
        interval_description = gr.Markdown()
        adjustment_help = gr.Markdown()
        adjustment_status = gr.Markdown()
        prompt = gr.Markdown(label="中文题目")
        reference_actions = gr.Markdown()
        intent_input = gr.Textbox(
            label="人工填写意图（只在单帧意图题显示）",
            placeholder="例如：持续挖掘准星指向的石块",
            lines=2,
        )
        gr.Markdown("## 题目审核")
        decision = gr.Radio(choices=[("通过", "approve"), ("不通过", "reject")], label="审核决定")
        reason = gr.Textbox(label="简短理由", lines=3, max_lines=5)
        with gr.Row():
            save_button = gr.Button("保存审核", variant="primary")
            save_next_button = gr.Button("保存并下一题")
        status = gr.Markdown()
        progress = gr.Markdown()

        render_outputs = [
            index_state,
            sample_selector,
            title,
            gallery,
            *frame_inputs,
            target_end,
            interval_description,
            adjustment_help,
            prompt,
            reference_actions,
            intent_input,
            decision,
            reason,
            progress,
        ]
        interface.load(render, gr.State(0), render_outputs)
        previous.click(lambda index: navigate(index, -1), index_state, render_outputs)
        next_button.click(lambda index: navigate(index, 1), index_state, render_outputs)
        sample_selector.change(jump, sample_selector, render_outputs)
        decision.change(fill_reason, [decision, reason], reason)
        save_inputs = [index_state, intent_input, decision, reason]
        save_button.click(save_review, save_inputs, [status, progress])
        save_next_button.click(save_review, save_inputs, [status, progress]).then(
            lambda index: navigate(index, 1),
            index_state,
            render_outputs,
        )
        apply_adjustment.click(
            adjust_frames,
            [index_state, *frame_inputs, target_end],
            [*render_outputs, adjustment_status],
        )
    return interface


def progress_markdown(store: ReviewStore) -> str:
    counts = store.counts()
    reviewed = counts["approve"] + counts["reject"] + counts["revise"]
    return (
        f"**进度：{reviewed} / {len(store.questions)}**　"
        f"通过 {counts['approve']}　拒绝 {counts['reject']}　"
        f"修订 {counts['revise']}　待审核 {counts['pending']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MineStudio 轨迹题 Gradio 人工审核")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--raw-dataset-dir", type=Path)
    parser.add_argument("--preannotations", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--username")
    parser.add_argument("--password")
    arguments = parser.parse_args()
    if bool(arguments.username) != bool(arguments.password):
        raise SystemExit("--username 与 --password 必须同时设置")
    store = ReviewStore(
        arguments.dataset_dir,
        arguments.reviews,
        arguments.raw_dataset_dir,
        arguments.preannotations,
    )
    interface = build_interface(store)
    try:
        interface.launch(
            server_name=arguments.host,
            server_port=arguments.port,
            share=arguments.share,
            auth=(arguments.username, arguments.password) if arguments.username else None,
            allowed_paths=[str(store.root.resolve())],
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
