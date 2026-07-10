# -*- coding: utf-8 -*-
"""慢塔设计 2 契约(grpo_pixel)验收:解析容错 / 状态行 / prev_done 回填语义。

无 LLM:全部用罐头字符串。真实 Omni 上的格式合规率验证属"大模型配合"项,另跑。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.craftground.grpo_pixel import (DECISIONS, parse_slow_reply,  # noqa: E402
                                          state_line)


def test_parse_full_reply():
    r = parse_slow_reply('{"prev_done": true, "decision": "continue", '
                         '"subgoal": "chop the oak tree", "aim": [612, 430], '
                         '"done_when": "oak_log in inventory"}')
    assert r["prev_done"] is True and r["decision"] == "continue"
    assert r["subgoal"] == "chop the oak tree" and r["aim"] == [612.0, 430.0]
    assert r["done_when"] == "oak_log in inventory" and r["parsed"]


def test_parse_degrades_per_field():
    """字段逐个降级,不整体作废:坏 aim → 中心;坏 decision → switch;缺 prev_done → False。"""
    r = parse_slow_reply('{"subgoal": "go", "aim": [99999, -5], "decision": "dance"}')
    assert r["aim"] == [1000.0, 0.0]                  # clip 而非作废
    assert r["decision"] == "switch" and r["decision"] in DECISIONS
    assert r["prev_done"] is False
    r2 = parse_slow_reply("total garbage no json")
    assert r2["subgoal"] == "" and r2["aim"] == [500.0, 500.0] and not r2["parsed"]


def test_parse_json_embedded_in_prose():
    r = parse_slow_reply('Sure! Here: {"subgoal": "mine stone", "aim": [500, 700]} done')
    assert r["subgoal"] == "mine stone" and r["parsed"]


def test_state_line_marks_and_bounds():
    """状态行:done 标记读第 6 位;只带最近 3 条子目标、最近 6 件库存。"""
    goal_log = [[0, "find tree", [500, 500], "", "switch", True],
                [20, "chop tree", [600, 400], "log in inv", "switch", False],
                [40, "chop more", [610, 410], "", "continue", False],
                [60, "make planks", [500, 500], "", "switch", False]]
    inv = {f"item{i}": i * 10 for i in range(8)}
    pose = [[0, 64, 0], [3, 64, 4]]
    s = state_line(80, inv, pose, goal_log)
    assert s.startswith("STATE t=80")
    assert "'find tree'" not in s                     # 只留最近 3 条
    assert "'chop tree'(open)" in s and "'make planks'(open)" in s
    assert "displacement:7blocks" in s
    assert "item0" not in s and "item7@70" in s       # 库存最近 6 件


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
