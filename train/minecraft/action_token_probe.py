"""探测 Qwen3VL 最擅长的动作 token 表示：格式 × horizon × 上下文条件的对照实验。

对外接口：main（命令行入口）。

协议（对应用户验收口径）：随机抽取若干个含图像与动作的数据窗口；对每种候选文本格式、
每个未来 horizon（默认 5 与 20 帧）、每种历史上下文条件（正确历史 / 无历史 / 随机历史），
让 Qwen3VL 生成动作并解码，重复若干次（默认 10）。用"关键动作一致率"评分——只比移动
方向、转向、姿态、攻击 / 使用、快捷栏、相机粗方向，抖动级差异不计。同时报告重复之间的
自一致性，以及 5 帧与 20 帧的差距。输出 JSON 行与 markdown 汇总。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import torch
from PIL import Image

from datasets.minestudio.groups import get_dataset_group
from datasets.minestudio.dataset import MineStudioLMDBDataset
from net.action_token_codec import (
    CAMERA_NEUTRAL_BIN,
    ActionTokenFormat,
    StructuredAction,
)
from net.qwen3vl_policy import (
    HistoryContext,
    Qwen3VLPolicyConfiguration,
    build_qwen3vl_action_policy,
)
from train.minecraft.action_supervision import (
    CAMERA_SCALE,
    DEGREES_PER_MOUSE_PIXEL,
    vpt_actions_to_structured,
)
from train.minecraft.evaluation import (
    KEY_ACTION_FIELDS,
    key_action_agreement_rate,
    key_action_field_matches,
    key_action_signature,
)

CONTEXT_CONDITIONS = ("correct_history", "no_history", "random_history")

# 独特性打分权重：越稀有 / 越有语义的动作权重越高，用于优先挑选含独特动作的窗口，
# 避免抽到清一色"前进 / noop"的平淡窗口。
_DISTINCTIVENESS_WEIGHTS = {
    "inventory": 6.0,
    "drop": 5.0,
    "hotbar": 4.0,
    "jump": 3.0,
    "use": 2.0,
    "attack": 1.0,
    "camera": 1.0,
    "turn": 0.5,
}


def _distinctiveness_score(actions: list[StructuredAction]) -> float:
    """给未来动作序列打独特性分：出现越多稀有 / 显著动作分越高。

    统计整段里 inventory / drop / hotbar 切换 / jump / use / attack / 大幅相机 / 转向
    这些"值得一看"的事件，各按权重累加，并对种类多样性给额外奖励，使得同时包含多种
    独特动作的窗口优先被选中。
    """
    if not actions:
        return 0.0
    total = 0.0
    present_kinds: set[str] = set()
    for action in actions:
        if action.keys["inventory"]:
            total += _DISTINCTIVENESS_WEIGHTS["inventory"]; present_kinds.add("inventory")
        if action.keys["drop"]:
            total += _DISTINCTIVENESS_WEIGHTS["drop"]; present_kinds.add("drop")
        if any(action.keys[f"hotbar.{index}"] for index in range(1, 10)):
            total += _DISTINCTIVENESS_WEIGHTS["hotbar"]; present_kinds.add("hotbar")
        if action.keys["jump"]:
            total += _DISTINCTIVENESS_WEIGHTS["jump"]; present_kinds.add("jump")
        if action.keys["use"]:
            total += _DISTINCTIVENESS_WEIGHTS["use"]; present_kinds.add("use")
        if action.keys["attack"]:
            total += _DISTINCTIVENESS_WEIGHTS["attack"]; present_kinds.add("attack")
        if action.keys["left"] or action.keys["right"]:
            total += _DISTINCTIVENESS_WEIGHTS["turn"]; present_kinds.add("turn")
        if (abs(action.camera_yaw_bin - CAMERA_NEUTRAL_BIN) >= 2
                or abs(action.camera_pitch_bin - CAMERA_NEUTRAL_BIN) >= 2):
            total += _DISTINCTIVENESS_WEIGHTS["camera"]; present_kinds.add("camera")
    # 种类多样性奖励：每多一种独特动作类型再加 2 分。
    return total + 2.0 * len(present_kinds)


def _compact_frame(action: StructuredAction) -> str:
    """把单帧动作压成一行可读串，用于序列对照展示。"""
    parts = []
    for key, symbol in (("forward", "F"), ("back", "B"), ("left", "L"), ("right", "R")):
        if action.keys[key]:
            parts.append(symbol)
    for key in ("jump", "sneak", "sprint", "attack", "use", "drop", "inventory"):
        if action.keys[key]:
            parts.append(key)
    for index in range(1, 10):
        if action.keys[f"hotbar.{index}"]:
            parts.append(f"h{index}")
    yaw = action.camera_yaw_bin - CAMERA_NEUTRAL_BIN
    pitch = action.camera_pitch_bin - CAMERA_NEUTRAL_BIN
    if yaw or pitch:
        parts.append(f"cam({yaw:+d},{pitch:+d})")
    return " ".join(parts) if parts else "noop"


def _frames_to_images(frames: torch.Tensor) -> list[Image.Image]:
    """把 ``[T,3,H,W]`` uint8 帧转为 PIL 图像列表。"""
    images = []
    for index in range(frames.shape[0]):
        array = frames[index].permute(1, 2, 0).contiguous().to(torch.uint8).cpu().numpy()
        images.append(Image.fromarray(array))
    return images


def _random_structured_actions(count: int, generator: random.Random) -> list[StructuredAction]:
    """构造若干帧随机但结构合法的动作，用于随机历史条件。"""
    actions = []
    for _ in range(count):
        keys = {key: False for key in StructuredAction().keys}
        if generator.random() < 0.6:
            keys[generator.choice(["forward", "back"])] = True
        if generator.random() < 0.3:
            keys[generator.choice(["left", "right"])] = True
        if generator.random() < 0.3:
            keys[generator.choice(["sneak", "sprint"])] = True
        for key in ("jump", "attack", "use"):
            if generator.random() < 0.2:
                keys[key] = True
        actions.append(StructuredAction(
            camera_yaw_bin=generator.randint(0, 10),
            camera_pitch_bin=generator.randint(0, 10),
            keys=keys,
        ))
    return actions


def _build_context(
    history_images: list[Image.Image],
    task_text: str,
    condition: str,
    correct_past: list[StructuredAction],
    generator: random.Random,
) -> HistoryContext:
    """按上下文条件构造历史上下文。"""
    if condition == "correct_history":
        past = correct_past
    elif condition == "no_history":
        past = []
    elif condition == "random_history":
        past = _random_structured_actions(len(correct_past), generator)
    else:
        raise ValueError(f"未知上下文条件 {condition}")
    return HistoryContext(frames=history_images, task_text=task_text, past_actions=past)


def _self_consistency(repetitions: list[list[StructuredAction]]) -> float:
    """多次生成之间的平均逐帧关键动作众数占比（自一致性）。"""
    horizon = len(repetitions[0])
    per_frame = []
    for frame_index in range(horizon):
        signatures = [
            json.dumps(key_action_signature(rep[frame_index]), sort_keys=True)
            for rep in repetitions
        ]
        most_common = Counter(signatures).most_common(1)[0][1]
        per_frame.append(most_common / len(repetitions))
    return sum(per_frame) / len(per_frame)


def _run_cell(
    policy,
    history_images: list[Image.Image],
    task_text: str,
    reference_future: list[StructuredAction],
    condition: str,
    correct_past: list[StructuredAction],
    repetitions: int,
    temperature: float,
    generator: random.Random,
    base_seed: int,
) -> dict[str, object]:
    """跑一个(格式×horizon×条件)单元：重复生成、评关键动作一致率与自一致性。"""
    include_past = condition != "no_history"
    horizon = len(reference_future)
    agreements = []
    field_hits = {name: 0 for name in KEY_ACTION_FIELDS}
    repetition_actions: list[list[StructuredAction]] = []
    first_text = ""
    for repetition in range(repetitions):
        context = _build_context(
            history_images, task_text, condition, correct_past, generator,
        )
        actions, text = policy.generate_actions(
            context, include_past_actions=include_past,
            temperature=temperature, seed=base_seed + repetition,
        )
        if repetition == 0:
            first_text = text
        repetition_actions.append(actions)
        agreements.append(key_action_agreement_rate(actions, reference_future))
        for prediction, target in zip(actions, reference_future):
            for name, matched in key_action_field_matches(prediction, target).items():
                field_hits[name] += int(matched)
    frames_scored = repetitions * horizon
    # 取一致率最高的一次重复作为序列对照的代表预测。
    best_index = max(range(len(agreements)), key=lambda index: agreements[index])
    representative = repetition_actions[best_index]
    per_frame_match = [
        key_action_agreement_rate([prediction], [target]) == 1.0
        for prediction, target in zip(representative, reference_future)
    ]
    return {
        "condition": condition,
        "horizon": horizon,
        "key_action_agreement_mean": sum(agreements) / len(agreements),
        "key_action_agreement_max": max(agreements),
        "self_consistency": _self_consistency(repetition_actions),
        "field_agreement": {
            name: field_hits[name] / frames_scored for name in KEY_ACTION_FIELDS
        },
        "sequence_comparison": {
            "reference": [_compact_frame(action) for action in reference_future],
            "predicted": [_compact_frame(action) for action in representative],
            "per_frame_match": per_frame_match,
            "raw_text_first_rep": first_text.strip()[:600],
        },
    }


def _markdown_report(records: list[dict[str, object]]) -> str:
    """把逐单元记录汇总成 markdown 表格与结论。"""
    lines = ["# Qwen3VL 动作 token 表示实验报告", ""]
    lines.append("关键动作一致率：只比移动/转向/姿态/攻击/使用/快捷栏/相机粗方向，抖动不计。")
    lines.append("")
    lines.append("| 格式 | horizon | 上下文 | 关键动作一致率(均值) | 一致率(最好) | 自一致性 |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for record in records:
        lines.append(
            f"| {record['format']} | {record['horizon']} | {record['condition']} | "
            f"{record['key_action_agreement_mean']:.3f} | "
            f"{record['key_action_agreement_max']:.3f} | "
            f"{record['self_consistency']:.3f} |"
        )
    lines.append("")
    # 按格式在 correct_history 条件下的平均一致率排序，给出推荐。
    by_format: dict[str, list[float]] = {}
    for record in records:
        if record["condition"] == "correct_history":
            by_format.setdefault(record["format"], []).append(
                record["key_action_agreement_mean"],
            )
    if by_format:
        ranked = sorted(
            by_format.items(),
            key=lambda item: sum(item[1]) / len(item[1]),
            reverse=True,
        )
        best_format = ranked[0][0]
        lines.append(f"**推荐格式（correct_history 平均一致率最高）：`{best_format}`**")
        lines.append("")
        for name, scores in ranked:
            lines.append(f"- `{name}`: 平均 {sum(scores) / len(scores):.3f}")
    return "\n".join(lines) + "\n"


def _sequence_comparison_section(
    records: list[dict[str, object]],
    window_distinctiveness: dict[int, float],
) -> str:
    """构造逐帧序列对照段：真实动作 vs 模型代表预测，标出每帧关键动作命中。

    只展示 correct_history 条件（最有信息量），按窗口独特性从高到低排列，方便人工判断
    模型在含开背包 / 跳跃 / 切物品栏等独特动作的片段上到底预测了什么。
    """
    lines = ["", "## 逐帧序列对照（真实 vs 预测，correct_history）", ""]
    lines.append("每格式取代表预测（该单元一致率最高的一次采样）。`✓`=该帧关键动作命中，`✗`=未命中。")
    lines.append("窗口按独特性得分从高到低排列（越高含越多稀有 / 显著动作）。")
    selected = [
        record for record in records if record["condition"] == "correct_history"
    ]
    selected.sort(
        key=lambda record: (
            window_distinctiveness.get(record["window_index"], 0.0),
            record["horizon"],
        ),
        reverse=True,
    )
    for record in selected:
        comparison = record["sequence_comparison"]
        score = window_distinctiveness.get(record["window_index"], 0.0)
        lines.append("")
        lines.append(
            f"### 窗口 {record['window_index']} · 格式 `{record['format']}` · "
            f"horizon {record['horizon']} · 独特性 {score:.1f}"
        )
        lines.append("")
        lines.append("| 帧 | 真实动作 | 模型预测 | 命中 |")
        lines.append("|---:|---|---|:--:|")
        for index, (reference, predicted, matched) in enumerate(zip(
            comparison["reference"], comparison["predicted"], comparison["per_frame_match"],
        )):
            mark = "✓" if matched else "✗"
            lines.append(f"| t{index} | {reference} | {predicted} | {mark} |")
        lines.append("")
        lines.append(f"<details><summary>首次采样原始文本</summary>\n\n```\n"
                     f"{comparison['raw_text_first_rep']}\n```\n</details>")
    return "\n".join(lines) + "\n"


def main() -> None:
    """运行动作 token 表示对照实验并输出 JSON 行与 markdown 报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", required=True,
                        help="已下载的某个 dataset-group 目录，例如 runs/data/minestudio/10xx")
    parser.add_argument("--dataset-group", default="10xx", choices=("7xx", "9xx", "10xx"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--formats", nargs="+",
                        default=[fmt.value for fmt in ActionTokenFormat],
                        choices=[fmt.value for fmt in ActionTokenFormat])
    parser.add_argument("--windows", type=int, default=3,
                        help="最终参与实验的窗口数（从候选池按独特性选出）")
    parser.add_argument("--candidate-multiplier", type=int, default=12,
                        help="候选池大小相对 --windows 的倍数；越大越可能选到含独特动作的窗口")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--image-height", type=int, default=252)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default="runs/action_token_probe/report.md")
    parser.add_argument("--json-log", default="runs/action_token_probe/records.jsonl")
    arguments = parser.parse_args()
    if arguments.history_frames < 1 or arguments.windows < 1 or arguments.repetitions < 1:
        raise ValueError("history-frames、windows、repetitions 必须大于零")
    if not torch.cuda.is_available():
        raise RuntimeError("该实验需要 CUDA")

    device = torch.device("cuda")
    generator = random.Random(arguments.seed)
    torch.manual_seed(arguments.seed)
    dataset_configuration = get_dataset_group(arguments.dataset_group)
    max_horizon = max(arguments.horizons)
    sequence_length = arguments.history_frames + max_horizon
    dataset = MineStudioLMDBDataset(
        data_directory=arguments.data_directory,
        sequence_length=sequence_length,
        image_size=(arguments.image_height, arguments.image_width),
        task_text=dataset_configuration.task_text,
        camera_max_degrees=CAMERA_SCALE * DEGREES_PER_MOUSE_PIXEL,
        split="all",
    )
    if len(dataset) == 0:
        raise RuntimeError("数据窗口为空，检查下载分片")
    # 从随机候选池里按独特性打分，优先选含独特动作的窗口。用最大 horizon 打分，
    # 保证被选窗口在最长序列上也足够"值得一看"。
    candidate_count = min(len(dataset), max(arguments.windows * arguments.candidate_multiplier,
                                            arguments.windows))
    candidate_indices = generator.sample(range(len(dataset)), k=candidate_count)
    scored_candidates: list[tuple[float, int]] = []
    for candidate in candidate_indices:
        sample = dataset[candidate]
        future = vpt_actions_to_structured(
            sample["act_agg"][arguments.history_frames - 1:
                              arguments.history_frames - 1 + max_horizon],
        )
        if len(future) != max_horizon:
            continue
        scored_candidates.append((_distinctiveness_score(future), candidate))
    if not scored_candidates:
        raise RuntimeError("没有满足最大 horizon 的候选窗口，减小 --horizons 或增大数据")
    scored_candidates.sort(reverse=True)
    window_indices = [index for _score, index in scored_candidates[:arguments.windows]]
    window_distinctiveness = {index: score for score, index in scored_candidates}
    print(json.dumps({
        "event": "windows_selected",
        "windows": [
            {"window_index": index, "distinctiveness": round(window_distinctiveness[index], 2)}
            for index in window_indices
        ],
    }, ensure_ascii=False), flush=True)
    records: list[dict[str, object]] = []
    report_path = Path(arguments.report)
    json_path = Path(arguments.json_log)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    for action_format_value in arguments.formats:
        action_format = ActionTokenFormat(action_format_value)
        for horizon in arguments.horizons:
            configuration = Qwen3VLPolicyConfiguration(
                model_name=arguments.model_name,
                action_format=action_format,
                action_horizon=horizon,
                max_history_frames=arguments.history_frames,
                max_new_tokens=max(96, 40 * horizon),
            )
            policy = build_qwen3vl_action_policy(
                configuration, device, arguments.cache_directory,
            )
            for window_position, window_index in enumerate(window_indices):
                sample = dataset[window_index]
                frames = sample["img"]
                actions = sample["act_agg"]
                history_images = _frames_to_images(frames[:arguments.history_frames])
                correct_past = vpt_actions_to_structured(
                    actions[:arguments.history_frames - 1],
                ) if arguments.history_frames > 1 else []
                reference_future = vpt_actions_to_structured(
                    actions[arguments.history_frames - 1:
                            arguments.history_frames - 1 + horizon],
                )
                if len(reference_future) != horizon:
                    continue
                for condition in CONTEXT_CONDITIONS:
                    cell = _run_cell(
                        policy, history_images, dataset_configuration.task_text,
                        reference_future, condition, correct_past,
                        arguments.repetitions, arguments.temperature, generator,
                        base_seed=arguments.seed + 1000 * window_position,
                    )
                    record = {
                        "format": action_format_value,
                        "window_index": int(window_index),
                        **cell,
                    }
                    records.append(record)
                    with json_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            del policy
            torch.cuda.empty_cache()

    # 聚合到 (格式, horizon, 条件) 层面再写报告，避免单窗口噪声。
    aggregated: dict[tuple, list[dict]] = {}
    for record in records:
        aggregated.setdefault(
            (record["format"], record["horizon"], record["condition"]), [],
        ).append(record)
    summary = []
    for (format_value, horizon, condition), group in aggregated.items():
        summary.append({
            "format": format_value,
            "horizon": horizon,
            "condition": condition,
            "key_action_agreement_mean": sum(
                item["key_action_agreement_mean"] for item in group) / len(group),
            "key_action_agreement_max": max(
                item["key_action_agreement_max"] for item in group),
            "self_consistency": sum(
                item["self_consistency"] for item in group) / len(group),
        })
    report_text = _markdown_report(summary) + _sequence_comparison_section(
        records, window_distinctiveness,
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(json.dumps({
        "event": "probe_complete",
        "records": len(records),
        "report": str(report_path),
        "json_log": str(json_path),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

