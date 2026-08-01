"""episode 划分的不变量测试：解析、无泄漏、占比逼近与确定性。"""

from __future__ import annotations

import pytest

from dataset.organization.split import (
    _select_groups_by_frames,
    _stable_order,
    parse_episode_identity,
)


def test_parse_episode_identity_splits_all_fields() -> None:
    """episode 名按 <前缀>-<hex>-<日期>-<时间> 拆开，前缀允许含连字符。"""
    identity = parse_episode_identity("lovely-persimmon-angora-02e496ce4abb-20220421-092639")
    assert identity.prefix == "lovely-persimmon-angora"
    assert identity.session == "02e496ce4abb"
    assert identity.date == "20220421"
    assert identity.time == "092639"


def test_parse_episode_identity_rejects_malformed_name() -> None:
    """结构不符时报错，不静默返回半解析结果。"""
    with pytest.raises(ValueError):
        parse_episode_identity("not-a-valid-episode-name")


def test_group_selection_hits_exact_ratio_when_possible() -> None:
    """存在精确解时枚举必须找到它，而不是停在近似值上。"""
    groups = {"a": 50, "b": 30, "c": 10, "d": 10}
    # 目标 20%：c + d = 20 恰好命中。
    assert _select_groups_by_frames(groups, 0.2) == ["c", "d"]


def test_group_selection_prefers_closest_subset_on_skewed_data() -> None:
    """帧数分布极偏时选偏差最小的子集，不是简单贪心的第一个。"""
    groups = {"huge": 900, "medium": 60, "small": 40}
    selected = _select_groups_by_frames(groups, 0.1)
    assert selected == ["medium", "small"]  # 100/1000 = 精确 10%


def test_group_selection_never_returns_all_groups() -> None:
    """选择结果必须留下至少一个组给训练集。"""
    groups = {"a": 10, "b": 10, "c": 10}
    selected = _select_groups_by_frames(groups, 0.99)
    assert len(selected) < len(groups)


def test_group_selection_rejects_empty_total() -> None:
    """总帧数为零时报错，不产生除零。"""
    with pytest.raises(ValueError):
        _select_groups_by_frames({"a": 0, "b": 0}, 0.1)


def test_stable_order_is_deterministic_and_input_order_independent() -> None:
    """稳定哈希排序与输入顺序无关，同种子结果恒定。"""
    names = ["alpha", "beta", "gamma", "delta"]
    first = _stable_order(names, seed=7)
    assert first == _stable_order(list(reversed(names)), seed=7)
    assert first == _stable_order(names, seed=7)


def test_stable_order_changes_with_seed() -> None:
    """换种子应得到不同顺序，否则种子参数形同虚设。"""
    names = [f"episode-{index}" for index in range(24)]
    assert _stable_order(names, seed=1) != _stable_order(names, seed=2)


def test_stable_order_is_a_permutation() -> None:
    """排序不增删元素。"""
    names = [f"episode-{index}" for index in range(16)]
    assert sorted(_stable_order(names, seed=3)) == sorted(names)
