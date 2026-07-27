# -*- coding: utf-8 -*-
"""把当前代码会发给模型的提示词原样导出成可读文件，便于人工审查是否夹带答案。

    python -m rl_training_environments.craftground.dump_prompt > prompt_snapshot.txt
"""
from __future__ import annotations

import sys

from rl_training_environments.craftground.llm_segment_controller import (
    describe_control_state,
)
from rl_training_environments.craftground.segment_prompt_builder import (
    PromptBuilder,
    SegmentRecord,
    render_request_zone,
)
from rl_training_environments.craftground.segment_text_codec import (
    canonical_segment_text,
    parse_segment_text,
)


class _FakeBlock:
    def __init__(self, key, x, y, z):
        self.translation_key, self.x, self.y, self.z = key, x, y, z


class _FakeRaycast:
    type = 1

    def __init__(self, block):
        self.target_block = block
        self.target_entity = None


class _FakeFull:
    x, y, z = -405.5, 69.0, -56.5
    yaw, pitch = 12.0, -8.0
    is_on_ground = True
    raycast_result = _FakeRaycast(_FakeBlock("block.minecraft.oak_log", -406, 70, -53))


def main() -> None:
    # 提示词全是中文，Windows 控制台默认 cp936 会在写出时 UnicodeEncodeError。
    # 显式转 UTF-8，免得调用方每次都要记着设 PYTHONUTF8=1。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    task_text = ("收集 4 个原木。你出生在一片森林里。\n"
                 "  每轮会告诉你背包里有什么，原木数攒到 4 即算完成。\n"
                 "  攒够 4 个原木后，再合成出 1 个工作台（crafting_table）。")
    builder = PromptBuilder(task_text)


    parsed = parse_segment_text(
        "for: 120/20s\nhold: W\nMouse: 6/20s +45,+0\nlook: 40/20s, 80/20s, 120/20s\n"
        "stop_if: stuck\nafter: stop 20/20s\nwhy: 转向右侧那棵树并走过去"
    )
    builder.append_record(SegmentRecord(
        index=1, canonical_text=canonical_segment_text(parsed), why_text=parsed.why_text,
        executed_ticks=94, planned_ticks=120, tripped_guard="stuck",
        observation_frames=[], state_note="背包变化：oak_log +1",
        truncation_notes=["位移 6.42 米，当前朝向 yaw +12.0°、pitch +8.0°（正=抬头）。"],
    ))
    builder.facts.record_measured(
        "aim_cap", "视角每 tick 最多 18°。你写过一次超额的 Mouse，结果少转了 27°。")

    text = builder.render_text_only_preview(
        describe_control_state(None), render_request_zone(32, 128, 4.2, 60),
    )
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
