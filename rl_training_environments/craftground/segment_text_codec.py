# -*- coding: utf-8 -*-
"""行式动作段文本 ←→ CraftGround V2 逐 tick 动作序列。

对外接口:
    PHYSICAL_KEY_TO_V2 — 物理键名 → V2 二值键（本文件是唯一知道物理键名的地方之一）。
    UNAVAILABLE_KEYS   — 本设备不存在的物理键（写了会被丢弃并告警）。
    ParsedSegment      — 解析结果；解析永不抛异常，脏输入记入 warnings。
    parse_segment_text — 模型文本 → ParsedSegment。
    compile_parsed_segment — ParsedSegment → 逐 tick 动作 dict 列表 + 观察点 tick。
    canonical_segment_text — ParsedSegment → 规范化文本（落盘/训练标签用）。

时间口径：全部绝对，从本段第一 tick 起算，单位 tick（20Hz 下 1 tick = 50ms）。
文本写作 `<帧号>/20s`；也宽容接受 `<秒>s` 与裸数字，均吸附到最近 tick。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rl_training_environments.craftground.action_contract import V2_KEYS

TICK_HZ = 20.0
MILLISECONDS_PER_TICK = 1000.0 / TICK_HZ

# 单段硬上限：防止模型写出无限长的段（10 秒）。超出直接裁剪并告警。
MAX_SEGMENT_TICKS = 200
MIN_SEGMENT_TICKS = 1
DEFAULT_SEGMENT_TICKS = 40

# 视角每 tick 上限，与 action_contract.CAM_MAX_DEG 同口径。
AIM_DEGREES_PER_TICK_CAP = 18.0
# ── GUI 光标：以下全部是实测标定，不是估算 ────────────────────────────────
# 1) 光标由相机增量驱动，且**只走整数像素**：1 px = 0.15°（Minecraft 鼠标灵敏度
#    的原生量子）。请求 0.34°/tick 实测只走 2px 而不是 2.267px——小数被截断且
#    不累积。所以规划必须按整数像素做，按度数平均摊会成比例丢量。
CURSOR_DEGREES_PER_PIXEL = 0.15
# 2) 浮点余量：degrees/0.15 在二进制下常略小于整数（0.45/0.15 = 2.9999…→2px），
#    折算回度数时加一点补偿才能拿到目标像素数。
CURSOR_DEGREE_EPSILON = 1e-4
# 3) 按 E 之后 GUI 要 2 tick 才真正打开，这期间的相机增量转的是世界视角而不是
#    光标。规划光标时必须跳过这 2 tick。
INVENTORY_OPEN_DELAY_TICKS = 2
# 4) 背包每次打开，光标复位到屏幕正中。
CURSOR_HOME = (0.5, 0.5)
# 5) 参考分辨率。光标按像素规划，因此需要知道真实屏幕尺寸。
CURSOR_SCREEN_WIDTH_PIXELS = 640
CURSOR_SCREEN_HEIGHT_PIXELS = 360

MAX_OBSERVATION_POINTS = 6
DEFAULT_LEASE_TICKS = 20

# 物理键 → V2 二值键。V2 没有取方块/副手/玩家列表/暂停通道，故不在表内。
PHYSICAL_KEY_TO_V2: Dict[str, str] = {
    "W": "forward", "A": "left", "S": "back", "D": "right",
    "Space": "jump", "Shift": "sneak", "Ctrl": "sprint",
    "Mouse_L": "attack", "Mouse_R": "use",
    "Q": "drop", "E": "inventory",
}
PHYSICAL_KEY_TO_V2.update({str(slot): f"hotbar.{slot}" for slot in range(1, 10)})

UNAVAILABLE_KEYS = frozenset(("Mouse_M", "F", "Tab", "Esc"))

# 互斥对：同时出现则双双丢弃（AGENTS §5 动作结构有界）。
EXCLUSIVE_KEY_PAIRS: Tuple[Tuple[str, str], ...] = (("W", "S"), ("A", "D"))

STOP_TRIGGERS = frozenset(("scene_changed", "flash", "stuck"))
TAIL_MODES = frozenset(("hold", "stop", "freeze"))

# 规范书写顺序：组内按表内顺序，组间按组号。
CANONICAL_KEY_ORDER: Tuple[str, ...] = (
    "W", "A", "S", "D",
    "Space", "Shift", "Ctrl",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "Mouse_L", "Mouse_R",
    "Q", "E",
)
_KEY_RANK = {key: index for index, key in enumerate(CANONICAL_KEY_ORDER)}

# 键名别名：模型可能写小写或换种拼法，宽容侧统一吸附。
_KEY_ALIASES: Dict[str, str] = {
    "SPACE": "Space", "SHIFT": "Shift", "CTRL": "Ctrl", "CONTROL": "Ctrl",
    "SNEAK": "Shift", "SPRINT": "Ctrl", "JUMP": "Space",
    "MOUSE_L": "Mouse_L", "MOUSE_LEFT": "Mouse_L", "LMB": "Mouse_L",
    "MOUSE_R": "Mouse_R", "MOUSE_RIGHT": "Mouse_R", "RMB": "Mouse_R",
    "MOUSE_M": "Mouse_M", "MOUSE_MIDDLE": "Mouse_M",
}


@dataclass
class HoldSpec:
    """一个按住键及其窗口（tick 闭开区间 [start_tick, end_tick)）。"""

    key: str
    start_tick: int
    end_tick: int


@dataclass
class TapSpec:
    """一次点按：在 tick 上按下一帧。"""

    tick: int
    keys: List[str]


@dataclass
class AxisSpec:
    """轴键的一个截止项：在 deadline_tick 之前完成 (x, y) 这个量。"""

    deadline_tick: int
    x: float
    y: float


@dataclass
class ParsedSegment:
    """解析后的动作段。解析器永不抛异常；所有问题记入 warnings。"""

    duration_ticks: int = DEFAULT_SEGMENT_TICKS
    holds: List[HoldSpec] = field(default_factory=list)
    taps: List[TapSpec] = field(default_factory=list)
    aim_items: List[AxisSpec] = field(default_factory=list)
    cursor_items: List[AxisSpec] = field(default_factory=list)
    observation_ticks: List[int] = field(default_factory=list)
    stop_triggers: List[str] = field(default_factory=list)
    tail_mode: str = "stop"
    lease_ticks: int = DEFAULT_LEASE_TICKS
    learn_text: str = ""
    why_text: str = ""
    warnings: List[str] = field(default_factory=list)
    raw_text: str = ""


_FRACTION_PATTERN = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*s?$")
_SECONDS_PATTERN = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*s$")
_BARE_NUMBER_PATTERN = re.compile(r"^([+-]?\d+(?:\.\d+)?)$")


def _parse_time_to_ticks(token: str) -> Optional[int]:
    """把一个时间 token 解析成 tick 数。

    接受 `30/20s`（分子=帧号，分母=控制帧率）、`1.5s`（秒，吸附到最近 tick）、
    裸数字（按秒解释）。无法解析返回 None。
    """
    text = token.strip()
    if not text:
        return None
    match = _FRACTION_PATTERN.match(text)
    if match is not None:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator <= 0:
            return None
        # 分母是书写时假定的帧率；换算成本环境 tick 数以保持时长语义不变。
        return int(round(numerator / denominator * TICK_HZ))
    match = _SECONDS_PATTERN.match(text)
    if match is not None:
        return int(round(float(match.group(1)) * TICK_HZ))
    match = _BARE_NUMBER_PATTERN.match(text)
    if match is not None:
        return int(round(float(match.group(1)) * TICK_HZ))
    return None


def _normalise_key(token: str) -> Optional[str]:
    """把一个键名 token 吸附到规范物理键名；不认识返回 None。"""
    text = token.strip()
    if not text:
        return None
    if text in _KEY_RANK or text in UNAVAILABLE_KEYS:
        return text
    upper = text.upper()
    if upper in _KEY_ALIASES:
        return _KEY_ALIASES[upper]
    for candidate in tuple(_KEY_RANK) + tuple(UNAVAILABLE_KEYS):
        if candidate.upper() == upper:
            return candidate
    return None


def _parse_degrees(token: str) -> Optional[float]:
    """解析 `+52`、`-6`、`+52deg`、`0` 这类度数 token。"""
    text = token.strip().lower().replace("deg", "").replace("°", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _split_entries(value: str) -> List[str]:
    """按逗号切分成条目，去掉空条目。"""
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_hold_line(value: str, parsed: ParsedSegment) -> None:
    """解析 hold 行：`W, D, Shift` 或带窗口 `Mouse_L 3/20s-9/20s`。"""
    for entry in _split_entries(value):
        tokens = entry.split()
        key = _normalise_key(tokens[0]) if tokens else None
        if key is None:
            parsed.warnings.append(f"hold 行无法识别的键：{entry!r}，已丢弃")
            continue
        if key in UNAVAILABLE_KEYS:
            parsed.warnings.append(f"hold 行的 {key} 在本设备不存在，已丢弃")
            continue
        start_tick = 0
        end_tick = parsed.duration_ticks
        if len(tokens) >= 2 and "-" in tokens[1]:
            left, _, right = tokens[1].partition("-")
            start_candidate = _parse_time_to_ticks(left)
            end_candidate = _parse_time_to_ticks(right)
            if start_candidate is None or end_candidate is None:
                parsed.warnings.append(f"hold 窗口无法解析：{entry!r}，按整段按住")
            else:
                start_tick, end_tick = start_candidate, end_candidate
        parsed.holds.append(HoldSpec(key=key, start_tick=start_tick, end_tick=end_tick))


def _parse_tap_line(value: str, parsed: ParsedSegment) -> None:
    """解析 tap 行：`30/20s E+P, 32/20s Q+E`（时刻 + 同刻多键用 +）。"""
    for entry in _split_entries(value):
        tokens = entry.split()
        if not tokens:
            continue
        tick = _parse_time_to_ticks(tokens[0])
        key_text = " ".join(tokens[1:]) if tick is not None else entry
        if tick is None:
            tick = 0
            parsed.warnings.append(f"tap 条目缺少时刻：{entry!r}，按第 0 tick 处理")
        keys: List[str] = []
        for token in key_text.replace("+", " ").split():
            key = _normalise_key(token)
            if key is None:
                parsed.warnings.append(f"tap 行无法识别的键：{token!r}，已丢弃")
            elif key in UNAVAILABLE_KEYS:
                parsed.warnings.append(f"tap 行的 {key} 在本设备不存在，已丢弃")
            else:
                keys.append(key)
        if keys:
            parsed.taps.append(TapSpec(tick=tick, keys=keys))


def _parse_axis_line(value: str, parsed: ParsedSegment, target: List[AxisSpec], label: str) -> None:
    """解析轴键行：`20/20s +0,+2, 45/20s +0,+3`（截止时刻 + 两个分量）。

    逗号既分隔条目又分隔分量，因此按 token 流解析：遇到时间 token 开新条目。
    """
    tokens = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    index = 0
    while index < len(tokens):
        deadline = _parse_time_to_ticks(tokens[index])
        if deadline is not None and index + 2 < len(tokens):
            first = _parse_degrees(tokens[index + 1])
            second = _parse_degrees(tokens[index + 2])
            consumed = 3
        else:
            # 没写时刻：整段末为截止时刻。
            deadline = parsed.duration_ticks
            first = _parse_degrees(tokens[index])
            second = _parse_degrees(tokens[index + 1]) if index + 1 < len(tokens) else None
            consumed = 2
        if first is None or second is None:
            parsed.warnings.append(f"{label} 行无法解析的分量：{tokens[index:index + 3]}，已丢弃")
            index += max(1, consumed)
            continue
        target.append(AxisSpec(deadline_tick=deadline, x=first, y=second))
        index += consumed


def _resolve_exclusive_conflicts(parsed: ParsedSegment) -> None:
    """互斥键同段按住则双双丢弃（窗口重叠才算冲突）。"""
    for left_key, right_key in EXCLUSIVE_KEY_PAIRS:
        left_holds = [hold for hold in parsed.holds if hold.key == left_key]
        right_holds = [hold for hold in parsed.holds if hold.key == right_key]
        conflicted = False
        for left in left_holds:
            for right in right_holds:
                if left.start_tick < right.end_tick and right.start_tick < left.end_tick:
                    conflicted = True
        if conflicted:
            parsed.holds = [hold for hold in parsed.holds if hold.key not in (left_key, right_key)]
            parsed.warnings.append(
                f"{left_key} 与 {right_key} 互斥且窗口重叠，两者已双双丢弃"
            )


def _clip_and_sort(parsed: ParsedSegment) -> None:
    """把所有时刻裁进 [0, duration_ticks]，排序去重，并记录裁剪。"""
    limit = parsed.duration_ticks
    for hold in parsed.holds:
        hold.start_tick = max(0, min(hold.start_tick, limit))
        hold.end_tick = max(hold.start_tick, min(hold.end_tick, limit))
    kept_taps: List[TapSpec] = []
    for tap in parsed.taps:
        if tap.tick > limit - 1:
            parsed.warnings.append(f"tap @{tap.tick} 超出段长 {limit}，已裁到 {max(0, limit - 1)}")
        tap.tick = max(0, min(tap.tick, limit - 1))
        kept_taps.append(tap)
    parsed.taps = sorted(kept_taps, key=lambda item: item.tick)
    for items, label in ((parsed.aim_items, "Mouse"), (parsed.cursor_items, "Cursor")):
        for item in items:
            if item.deadline_tick > limit:
                parsed.warnings.append(f"{label} 截止 @{item.deadline_tick} 超出段长，已裁到 {limit}")
            item.deadline_tick = max(1, min(item.deadline_tick, limit))
        items.sort(key=lambda item: item.deadline_tick)
    # Cursor 是归一化绝对位置，越界通常意味着模型把它当成了相对增量或写了像素值。
    # codec 面向模型宽容：夹回 0..1 并明确告警，而不是静默照做。
    for item in parsed.cursor_items:
        clamped_x = min(1.0, max(0.0, item.x))
        clamped_y = min(1.0, max(0.0, item.y))
        if (clamped_x, clamped_y) != (item.x, item.y):
            parsed.warnings.append(
                f"Cursor 位置 ({item.x:.3f},{item.y:.3f}) 超出 0..1，已夹到 "
                f"({clamped_x:.3f},{clamped_y:.3f})。Cursor 写的是屏幕绝对位置，不是位移量"
            )
        item.x, item.y = clamped_x, clamped_y
    observation_ticks = sorted({max(1, min(tick, limit)) for tick in parsed.observation_ticks})
    if len(observation_ticks) > MAX_OBSERVATION_POINTS:
        parsed.warnings.append(
            f"观察点 {len(observation_ticks)} 个超过上限 {MAX_OBSERVATION_POINTS}，已保留最后 {MAX_OBSERVATION_POINTS} 个"
        )
        observation_ticks = observation_ticks[-MAX_OBSERVATION_POINTS:]
    parsed.observation_ticks = observation_ticks


def parse_segment_text(text: str) -> ParsedSegment:
    """把模型输出的行式动作段解析成 ParsedSegment。永不抛异常。"""
    parsed = ParsedSegment(raw_text=text)
    # 先扫一遍找 for，因为 hold 的默认窗口和各种裁剪都依赖段长。
    lines = [line.strip() for line in text.replace("\r", "").split("\n")]
    lines = [line for line in lines if line and not line.startswith("#")]
    pending: List[Tuple[str, str]] = []
    for line in lines:
        name, separator, value = line.partition(":")
        if not separator:
            continue
        pending.append((name.strip().lower(), value.strip()))

    for name, value in pending:
        if name == "for":
            ticks = _parse_time_to_ticks(value)
            if ticks is None:
                parsed.warnings.append(f"for 无法解析：{value!r}，按默认 {DEFAULT_SEGMENT_TICKS} tick")
            else:
                if ticks > MAX_SEGMENT_TICKS:
                    parsed.warnings.append(f"for {ticks} tick 超过上限 {MAX_SEGMENT_TICKS}，已裁剪")
                parsed.duration_ticks = max(MIN_SEGMENT_TICKS, min(ticks, MAX_SEGMENT_TICKS))
            break

    for name, value in pending:
        if name == "for":
            continue
        if name == "hold":
            _parse_hold_line(value, parsed)
        elif name == "tap":
            _parse_tap_line(value, parsed)
        elif name == "mouse":
            _parse_axis_line(value, parsed, parsed.aim_items, "Mouse")
        elif name == "cursor":
            _parse_axis_line(value, parsed, parsed.cursor_items, "Cursor")
        elif name == "look":
            for entry in _split_entries(value):
                tick = _parse_time_to_ticks(entry)
                if tick is None:
                    parsed.warnings.append(f"look 无法解析的时刻：{entry!r}，已丢弃")
                else:
                    parsed.observation_ticks.append(tick)
        elif name == "stop_if":
            for entry in _split_entries(value):
                trigger = entry.split()[0].lower()
                if trigger in STOP_TRIGGERS:
                    parsed.stop_triggers.append(trigger)
                else:
                    parsed.warnings.append(f"stop_if 未知触发条件：{entry!r}，已丢弃")
        elif name == "after":
            tokens = value.split()
            if tokens and tokens[0].lower() in TAIL_MODES:
                parsed.tail_mode = tokens[0].lower()
            elif tokens:
                parsed.warnings.append(f"after 未知处置：{tokens[0]!r}，按 stop 处理")
            if len(tokens) >= 2:
                lease = _parse_time_to_ticks(tokens[1])
                if lease is not None:
                    parsed.lease_ticks = max(1, lease)
        elif name == "learn":
            parsed.learn_text = value
        elif name == "why":
            parsed.why_text = value

    if not parsed.observation_ticks:
        parsed.observation_ticks = [parsed.duration_ticks]
        parsed.warnings.append("未写 look，已自动在段末插入一个观察点")
    _resolve_exclusive_conflicts(parsed)
    _clip_and_sort(parsed)
    return parsed


def _spread_axis_items(
    items: Sequence[AxisSpec], duration_ticks: int, cap_per_tick: float,
) -> Tuple[List[float], List[float], float, float]:
    """把轴键的截止项摊布到逐 tick 增量。

    Returns
    -------
    per_tick_x, per_tick_y : List[float]
        长度 = duration_ticks 的逐 tick 增量（度）。
    truncated_x, truncated_y : float
        因每 tick 上限而未能完成的残余量（度），用于如实记账。
    """
    per_tick_x = [0.0] * duration_ticks
    per_tick_y = [0.0] * duration_ticks
    truncated_x = 0.0
    truncated_y = 0.0
    window_start = 0
    for item in items:
        window_end = max(window_start + 1, min(item.deadline_tick, duration_ticks))
        tick_count = window_end - window_start
        if tick_count <= 0:
            truncated_x += item.x
            truncated_y += item.y
            continue
        budget = cap_per_tick * tick_count
        share_x = max(-budget, min(item.x, budget))
        share_y = max(-budget, min(item.y, budget))
        truncated_x += item.x - share_x
        truncated_y += item.y - share_y
        for offset in range(tick_count):
            per_tick_x[window_start + offset] += share_x / tick_count
            per_tick_y[window_start + offset] += share_y / tick_count
        window_start = window_end
    return per_tick_x, per_tick_y, truncated_x, truncated_y


def _pixels_to_degrees(pixels: int) -> float:
    """整数像素折回度数。

    加 epsilon 是因为 degrees/0.15 在二进制下常略小于整数（0.45/0.15 = 2.9999…），
    底层再向下取整就会少走 1 px；补一点余量才能拿到目标像素数。符号跟着位移走。
    """
    epsilon = CURSOR_DEGREE_EPSILON if pixels > 0 else -CURSOR_DEGREE_EPSILON
    return pixels * CURSOR_DEGREES_PER_PIXEL + epsilon


def _spread_cursor_items(
    items: Sequence[AxisSpec],
    duration_ticks: int,
    cap_per_tick: float,
    start: Tuple[float, float],
    inventory_toggle_ticks: Sequence[int] = (),
) -> Tuple[List[float], List[float], float, float, Tuple[float, float]]:
    """把 Cursor 的**绝对目标位置**摊成逐 tick 相机增量。

    与 _spread_axis_items 的关键区别：Cursor 项是「光标要去哪」而不是「挪多少」，
    所以每项都得先减去当前光标位置才是本段该转的度数。段内维护光标状态，
    多个 Cursor 项因此不会像相对量那样累积漂移。

    符号口径：光标 y 向下为正，V2 的 camera_pitch 也是向下为正（实测标定），
    所以这里返回的 y 增量**不需要取反**——取反是世界视角 Mouse 的事。

    量化：光标只走整数像素（1 px = 0.15°），小数截断且不累积。因此这里按整数
    像素分配，而不是把度数平均摊到每 tick——后者会成比例丢量（实测只到位八成）。

    Returns
    -------
    per_tick_x, per_tick_y : List[float]
        逐 tick 的 camera_yaw / camera_pitch 增量（度），已按上限截断。
    truncated_x, truncated_y : float
        因每 tick 上限没走完的残余（屏幅），用于如实记账。
    end_cursor : Tuple[float, float]
        段末光标实际所在（含截断影响），供下一段作为 start 传入。
    """
    max_pixels_per_tick = int(cap_per_tick / CURSOR_DEGREES_PER_PIXEL)

    def plan_pixels(total_pixels: int, tick_count: int) -> List[int]:
        """把整数像素位移分配到 tick 上：**尽量晚走，按截止时刻到达**。

        两个理由：
        - 语义是"到位后停住"。上一项的截止时刻就是下一项的起点，若下一项一开始
          就冲过去，落在两个截止时刻之间的点击会打在半路上——这正是实测里
          "点击落到下一个槽位"的原因。晚走则光标停在上一个目标上等着被点。
        - 按整数像素走满上限，不平均摊，避免每 tick 的小数像素被底层截断。
        """
        remaining = abs(total_pixels)
        sign = 1 if total_pixels >= 0 else -1
        plan = [0] * tick_count
        for offset in range(tick_count - 1, -1, -1):
            if remaining <= 0:
                break
            move = min(remaining, max_pixels_per_tick)
            plan[offset] = sign * move
            remaining -= move
        return plan

    per_tick_x = [0.0] * duration_ticks
    per_tick_y = [0.0] * duration_ticks
    truncated_x = 0.0
    truncated_y = 0.0
    current_x, current_y = start
    window_start = 0
    toggles = sorted(set(inventory_toggle_ticks))
    for item in items:
        window_end = max(window_start + 1, min(item.deadline_tick, duration_ticks))
        # 按 E 会重开 GUI 并把光标弹回正中，所以 E 之前的移动全部作废；且 GUI 要
        # INVENTORY_OPEN_DELAY_TICKS 个 tick 才真正打开，这期间转的是世界视角，
        # 光标规划必须从 E + 延迟 之后才起算。
        inside = [tick for tick in toggles if window_start <= tick < window_end]
        if inside:
            current_x, current_y = CURSOR_HOME
            window_start = max(inside) + INVENTORY_OPEN_DELAY_TICKS
            window_end = max(window_start + 1, window_end)
            toggles = [tick for tick in toggles if tick > max(inside)]
        window_start = min(window_start, duration_ticks - 1)
        window_end = min(max(window_start + 1, window_end), duration_ticks)
        tick_count = window_end - window_start
        target_x = min(1.0, max(0.0, item.x))
        target_y = min(1.0, max(0.0, item.y))
        # 目标与现状的差折成整数像素——光标的位移量子就是像素。
        want_pixels_x = int(round((target_x - current_x) * CURSOR_SCREEN_WIDTH_PIXELS))
        want_pixels_y = int(round((target_y - current_y) * CURSOR_SCREEN_HEIGHT_PIXELS))
        plan_x = plan_pixels(want_pixels_x, tick_count)
        plan_y = plan_pixels(want_pixels_y, tick_count)
        for offset in range(tick_count):
            if plan_x[offset]:
                per_tick_x[window_start + offset] += _pixels_to_degrees(plan_x[offset])
            if plan_y[offset]:
                per_tick_y[window_start + offset] += _pixels_to_degrees(plan_y[offset])
        moved_pixels_x = sum(plan_x)
        moved_pixels_y = sum(plan_y)
        truncated_x += (want_pixels_x - moved_pixels_x) / CURSOR_SCREEN_WIDTH_PIXELS
        truncated_y += (want_pixels_y - moved_pixels_y) / CURSOR_SCREEN_HEIGHT_PIXELS
        current_x += moved_pixels_x / CURSOR_SCREEN_WIDTH_PIXELS
        current_y += moved_pixels_y / CURSOR_SCREEN_HEIGHT_PIXELS
        window_start = window_end
    return per_tick_x, per_tick_y, truncated_x, truncated_y, (current_x, current_y)


def _blank_action() -> Dict[str, object]:
    action: Dict[str, object] = {key: False for key in V2_KEYS}
    action["camera_yaw"] = 0.0
    action["camera_pitch"] = 0.0
    return action


@dataclass
class CompiledSegment:
    """编译产物：逐 tick 动作 + 观察点 + 如实记账的截断量。"""

    actions: List[Dict[str, object]]
    observation_ticks: List[int]
    aim_truncated_yaw_deg: float
    aim_truncated_pitch_deg: float
    cursor_truncated_x: float
    cursor_truncated_y: float
    cursor_end: Tuple[float, float]


def compile_parsed_segment(
    parsed: ParsedSegment,
    cursor_start: Tuple[float, float] = CURSOR_HOME,
) -> CompiledSegment:
    """ParsedSegment → 逐 tick V2 动作 dict 列表。

    hold 在窗口内每 tick 置真；tap 只在其 tick 置真一帧；Mouse/Cursor 摊成相机增量。
    Cursor 是**绝对位置**，按实测标定折成相机度数；因为光标状态跨段延续，
    调用方需把上一段的 cursor_end 作为本段的 cursor_start 传进来，
    否则每段都会以为光标还在正中。
    """
    duration = parsed.duration_ticks
    actions = [_blank_action() for _ in range(duration)]

    for hold in parsed.holds:
        v2_key = PHYSICAL_KEY_TO_V2.get(hold.key)
        if v2_key is None:
            continue
        for tick in range(hold.start_tick, min(hold.end_tick, duration)):
            actions[tick][v2_key] = True

    for tap in parsed.taps:
        if not 0 <= tap.tick < duration:
            continue
        for key in tap.keys:
            v2_key = PHYSICAL_KEY_TO_V2.get(key)
            if v2_key is not None:
                actions[tap.tick][v2_key] = True

    aim_x, aim_y, aim_truncated_x, aim_truncated_y = _spread_axis_items(
        parsed.aim_items, duration, AIM_DEGREES_PER_TICK_CAP,
    )
    (
        cursor_x, cursor_y, cursor_truncated_x, cursor_truncated_y, cursor_end,
    ) = _spread_cursor_items(
        parsed.cursor_items, duration, AIM_DEGREES_PER_TICK_CAP, cursor_start,
        inventory_toggle_ticks=[
            tap.tick for tap in parsed.taps if "E" in tap.keys
        ],
    )
    for tick in range(duration):
        actions[tick]["camera_yaw"] = aim_x[tick] + cursor_x[tick]
        # 两个 y 分量口径不同，必须分别处理，不能先相加再取反：
        #   Mouse 的 +pitch = 抬头，而 V2 camera_pitch 正方向向下，故取负；
        #   Cursor 的 +y = 光标向下，与 V2 camera_pitch 同向，故原样相加。
        actions[tick]["camera_pitch"] = -aim_y[tick] + cursor_y[tick]

    return CompiledSegment(
        actions=actions,
        observation_ticks=list(parsed.observation_ticks),
        aim_truncated_yaw_deg=aim_truncated_x,
        aim_truncated_pitch_deg=aim_truncated_y,
        cursor_truncated_x=cursor_truncated_x,
        cursor_truncated_y=cursor_truncated_y,
        cursor_end=cursor_end,
    )


def tail_action(tail_mode: str) -> Dict[str, object]:
    """段末处置对应的一帧动作。

    hold 无法在无状态的 V2 dict 里表达"保持上一帧"，运行侧负责复制上一帧；
    本函数只给 stop（全松）与 freeze（松移动键，其余由运行侧保留）的空动作。
    """
    return _blank_action()


def _format_tick(tick: int) -> str:
    return f"{tick}/{int(TICK_HZ)}s"


def canonical_segment_text(parsed: ParsedSegment) -> str:
    """把 ParsedSegment 渲染成规范形式文本（落盘 / 训练标签用）。

    键按 CANONICAL_KEY_ORDER 排序，时刻绝对升序去重，同刻多键合并为 `+` 组。
    """
    lines: List[str] = [f"for: {_format_tick(parsed.duration_ticks)}"]

    if parsed.holds:
        entries: List[str] = []
        for hold in sorted(parsed.holds, key=lambda item: (_KEY_RANK.get(item.key, 99), item.start_tick)):
            if hold.start_tick == 0 and hold.end_tick >= parsed.duration_ticks:
                entries.append(hold.key)
            else:
                entries.append(f"{hold.key} {_format_tick(hold.start_tick)}-{_format_tick(hold.end_tick)}")
        lines.append("hold: " + ", ".join(entries))

    if parsed.taps:
        merged: Dict[int, List[str]] = {}
        for tap in parsed.taps:
            merged.setdefault(tap.tick, []).extend(tap.keys)
        entries = []
        for tick in sorted(merged):
            keys = sorted(dict.fromkeys(merged[tick]), key=lambda key: _KEY_RANK.get(key, 99))
            entries.append(f"{_format_tick(tick)} " + "+".join(keys))
        lines.append("tap: " + ", ".join(entries))

    for items, label in ((parsed.aim_items, "Mouse"), (parsed.cursor_items, "Cursor")):
        if not items:
            continue
        entries = [
            f"{_format_tick(item.deadline_tick)} {item.x:+g},{item.y:+g}" for item in items
        ]
        lines.append(f"{label}: " + ", ".join(entries))

    lines.append("look: " + ", ".join(_format_tick(tick) for tick in parsed.observation_ticks))
    if parsed.stop_triggers:
        lines.append("stop_if: " + ", ".join(dict.fromkeys(parsed.stop_triggers)))
    lines.append(f"after: {parsed.tail_mode} {_format_tick(parsed.lease_ticks)}")
    if parsed.learn_text:
        lines.append(f"learn: {parsed.learn_text}")
    if parsed.why_text:
        lines.append(f"why: {parsed.why_text}")
    return "\n".join(lines)
