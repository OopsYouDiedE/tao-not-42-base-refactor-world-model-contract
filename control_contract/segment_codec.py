# -*- coding: utf-8 -*-
"""决策段与大模型文本之间的鲁棒编解码，以及给大模型的格式说明生成。

对外接口：
    encode_segment — Segment → 单个 JSON 对象文本（SFT 目标）。
    decode_segment — 大模型自由文本 → 合法 Segment，任何脏输出都产出可执行结果。
    describe_segment_format — 生成格式说明（含 profile 能力段），进入 prompt。
    describe_execution_report — 把上一段执行结果渲染成 prompt 片段。
    describe_control_state — 把当前 latch / 光标 / 槽位状态渲染成 prompt 片段。

鲁棒性纪律（沿用 AGENTS §5）：解码端永不抛错、永不返回非法结构。识别失败退化为
"一个安全空步 + RELEASE_ALL 尾策略"的最小段；未知角色丢弃；越界数值截断。位移用极坐标
表达，因此结构上不存在互斥冲突需要消解。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from control_contract.binding_profile import BindingProfile, describe_capabilities
from control_contract.decision_segment import (
    Aim,
    BUILTIN_CHANNELS,
    DEFAULT_LEASE_MS,
    ExecutionReport,
    Guard,
    GuardComparison,
    Hold,
    MAX_LEASE_MS,
    MAX_STEP_DURATION_MS,
    MIN_STEP_DURATION_MS,
    Move,
    Point,
    Press,
    Release,
    Segment,
    Select,
    Step,
    TailPolicy,
    TextInput,
    ControlState,
)

# 解码失败时的兜底步时长（毫秒）：足够短，让运行时尽快回到推理循环重新看画面。
FALLBACK_STEP_DURATION_MS = 100


def _encode_step(step: Step) -> Dict[str, Any]:
    """把一个 Step 编成 JSON 可序列化 dict，只写非默认字段以压缩 token。"""
    payload: Dict[str, Any] = {"ms": step.duration_ms}
    if step.aim is not None:
        payload["aim"] = {"yaw": round(step.aim.yaw_deg, 2),
                          "pitch": round(step.aim.pitch_deg, 2)}
    if step.move is not None:
        payload["move"] = {"dir": round(step.move.direction_deg, 1),
                           "power": round(step.move.power, 2)}
    if step.point is not None:
        payload["point"] = {"x": round(step.point.x, 3), "y": round(step.point.y, 3)}
    if step.presses:
        payload["press"] = [
            item.role if item.repeat == 1 else {"role": item.role, "repeat": item.repeat}
            for item in step.presses
        ]
    if step.holds:
        payload["hold"] = [item.role for item in step.holds]
    if step.releases:
        payload["release"] = [item.role for item in step.releases]
    if step.select is not None:
        payload["select"] = step.select.slot
    if step.text is not None:
        payload["text"] = step.text.text
    return payload


def _encode_guard(guard: Guard) -> Dict[str, Any]:
    """把一个 Guard 编成 JSON 可序列化 dict。"""
    payload: Dict[str, Any] = {
        "channel": guard.channel,
        "when": guard.comparison.value,
        "threshold": guard.threshold,
    }
    if guard.sustain_ms:
        payload["sustain_ms"] = guard.sustain_ms
    if guard.label:
        payload["label"] = guard.label
    return payload


def encode_segment(segment: Segment) -> str:
    """把 Segment 编码为单个 JSON 对象文本（SFT 监督目标与历史动作回灌都用它）。

    Parameters
    ----------
    segment : Segment

    Returns
    -------
    str
        一行 JSON；字段顺序固定，便于大模型模仿。
    """
    payload: Dict[str, Any] = {}
    if segment.intent:
        payload["intent"] = segment.intent
    payload["steps"] = [_encode_step(step) for step in segment.steps]
    if segment.guards:
        payload["guards"] = [_encode_guard(guard) for guard in segment.guards]
    payload["tail"] = segment.tail.value
    payload["lease_ms"] = segment.lease_ms
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))


def _extract_json_objects(text: str) -> List[Mapping[str, Any]]:
    """从自由文本中扫出所有顶层花括号平衡且可解析的 JSON 对象。

    容忍 ```json 代码围栏、前后散文与多余对象；引号内的花括号不计入配对。
    """
    found: List[Mapping[str, Any]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start:index + 1])
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        found.append(parsed)
                    start = -1
    return found


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """把任意值转成 float，失败返回 default。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):  # NaN / inf
        return default
    return result


def _coerce_int(value: Any, default: int) -> int:
    """把任意值转成 int，失败返回 default。"""
    return int(round(_coerce_float(value, float(default))))


def _clamp(value: float, low: float, high: float) -> float:
    """区间截断。"""
    return max(low, min(high, value))


def _resolve_or_none(profile: BindingProfile, name: Any) -> Optional[str]:
    """解析角色名，非法返回 None（解码端丢弃而非抛错）。"""
    if not isinstance(name, str):
        return None
    try:
        return profile.resolve_role(name)
    except ValueError:
        return None


def _role_list(profile: BindingProfile, value: Any) -> List[str]:
    """把角色名（单个或数组）解析为合法核心角色列表，去重保序。"""
    candidates = value if isinstance(value, list) else [value]
    result: List[str] = []
    for item in candidates:
        role = _resolve_or_none(profile, item)
        if role is not None and role not in result:
            result.append(role)
    return result


def _decode_step(payload: Mapping[str, Any], profile: BindingProfile) -> Optional[Step]:
    """解码单个步；无任何可识别内容返回 None。"""
    duration = _coerce_int(payload.get("ms", payload.get("duration_ms", 0)), 0)
    aim = None
    aim_payload = payload.get("aim")
    if isinstance(aim_payload, Mapping):
        aim = Aim(
            yaw_deg=_coerce_float(aim_payload.get("yaw", aim_payload.get("yaw_deg"))),
            pitch_deg=_coerce_float(aim_payload.get("pitch", aim_payload.get("pitch_deg"))),
        )
    move = None
    move_payload = payload.get("move")
    if isinstance(move_payload, Mapping):
        move = Move(
            direction_deg=_coerce_float(
                move_payload.get("dir", move_payload.get("direction_deg"))),
            power=_clamp(_coerce_float(move_payload.get("power", 1.0), 1.0), 0.0, 1.0),
        )
    point = None
    point_payload = payload.get("point")
    if isinstance(point_payload, Mapping):
        point = Point(
            x=_clamp(_coerce_float(point_payload.get("x", 0.5), 0.5), 0.0, 1.0),
            y=_clamp(_coerce_float(point_payload.get("y", 0.5), 0.5), 0.0, 1.0),
        )
    presses: List[Press] = []
    press_payload = payload.get("press")
    if press_payload is not None:
        items = press_payload if isinstance(press_payload, list) else [press_payload]
        for item in items:
            if isinstance(item, Mapping):
                role = _resolve_or_none(profile, item.get("role"))
                repeat = max(1, _coerce_int(item.get("repeat", 1), 1))
            else:
                role = _resolve_or_none(profile, item)
                repeat = 1
            if role is not None:
                presses.append(Press(role=role, repeat=repeat))
    holds = [Hold(role=role) for role in _role_list(profile, payload.get("hold", []))]
    releases = [Release(role=role) for role in _role_list(profile, payload.get("release", []))]
    held = {item.role for item in holds}
    releases = [item for item in releases if item.role not in held]
    select = None
    if payload.get("select") is not None and profile.slot_count > 0:
        slot = _coerce_int(payload.get("select"), 0)
        if 1 <= slot <= profile.slot_count:
            select = Select(slot=slot)
    text = None
    if isinstance(payload.get("text"), str) and profile.supports_text and payload["text"]:
        text = TextInput(text=payload["text"])
    if not any((aim, move, point, presses, holds, releases, select, text)) and duration <= 0:
        return None
    duration = int(_clamp(duration or FALLBACK_STEP_DURATION_MS,
                          MIN_STEP_DURATION_MS, MAX_STEP_DURATION_MS))
    return Step(duration_ms=duration, aim=aim, move=move, point=point, presses=presses,
                holds=holds, releases=releases, select=select, text=text)


def _decode_guard(payload: Mapping[str, Any], known_channels: frozenset) -> Optional[Guard]:
    """解码单个守卫；通道未声明或比较方式不可识别则返回 None（丢弃该守卫）。"""
    channel = payload.get("channel")
    if not isinstance(channel, str) or channel not in known_channels:
        return None
    raw_comparison = str(payload.get("when", payload.get("comparison", ""))).strip().lower()
    try:
        comparison = GuardComparison(raw_comparison)
    except ValueError:
        return None
    return Guard(
        channel=channel,
        comparison=comparison,
        threshold=_coerce_float(payload.get("threshold")),
        sustain_ms=max(0, _coerce_int(payload.get("sustain_ms", 0), 0)),
        label=str(payload.get("label", "")),
    )


def _fallback_segment() -> Segment:
    """解码完全失败时的最小安全段：短暂静止后立刻回到推理循环。"""
    return Segment(
        steps=[Step(duration_ms=FALLBACK_STEP_DURATION_MS)],
        guards=[],
        tail=TailPolicy.RELEASE_ALL,
        lease_ms=0,
        intent="decode failed: idle and re-observe",
    )


def decode_segment(
    text: str,
    profile: BindingProfile,
    signal_channels: frozenset = frozenset(),
) -> Segment:
    """把大模型自由文本解码为一个合法可执行的决策段。

    永不抛错：识别失败返回 ``_fallback_segment()``（一个短静止步 + RELEASE_ALL），
    保证运行时总能拿到结构合法的段（AGENTS §5）。

    Parameters
    ----------
    text : str
        大模型原始输出，可含代码围栏与散文。
    profile : BindingProfile
        用于解析角色别名、判定槽位与文本能力。
    signal_channels : frozenset
        环境额外声明的数值信号通道名；与内建像素通道共同构成合法守卫通道集合。

    Returns
    -------
    Segment
        结构合法的决策段。
    """
    known_channels = frozenset(BUILTIN_CHANNELS) | frozenset(signal_channels)
    for payload in _extract_json_objects(text):
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            continue
        steps = [
            step for step in (
                _decode_step(item, profile) for item in raw_steps if isinstance(item, Mapping)
            ) if step is not None
        ]
        if not steps:
            continue
        raw_guards = payload.get("guards")
        guards: List[Guard] = []
        if isinstance(raw_guards, list):
            for item in raw_guards:
                if isinstance(item, Mapping):
                    guard = _decode_guard(item, known_channels)
                    if guard is not None:
                        guards.append(guard)
        try:
            tail = TailPolicy(str(payload.get("tail", "")).strip().lower())
        except ValueError:
            tail = TailPolicy.RELEASE_ALL
        lease = _coerce_int(payload.get("lease_ms", DEFAULT_LEASE_MS), DEFAULT_LEASE_MS)
        lease = int(_clamp(lease, 0, MAX_LEASE_MS))
        intent = payload.get("intent")
        return Segment(
            steps=steps, guards=guards, tail=tail, lease_ms=lease,
            intent=intent if isinstance(intent, str) else "",
        )
    return _fallback_segment()


_FORMAT_HEADER = """You control the game by emitting ONE decision segment per turn, as a single
JSON object. A segment is a short time-extended program, not a single frame: you will not be
asked again until it finishes or a guard interrupts it.

{
  "intent": "<one short line: what this segment is for>",
  "steps": [ {"ms": <duration>, ...primitives...}, ... ],
  "guards": [ {"channel": "<name>", "when": "below|above|delta_above",
               "threshold": <number>, "sustain_ms": <int>, "label": "<why>"} ],
  "tail": "hold|release_move|release_all",
  "lease_ms": <int>
}

Steps run in order; every primitive inside one step runs concurrently.
Primitives:
  "aim":   {"yaw": <deg>, "pitch": <deg>}  relative turn, +yaw right, +pitch up
  "move":  {"dir": <deg>, "power": 0..1}   0=forward, 90=right, 180=back, 270=left
  "point": {"x": 0..1, "y": 0..1}          UI position, (0,0) top-left
  "press": ["<role>", ...]                 tap now, released inside this step
  "hold":  ["<role>", ...]                 keep pressed across later steps and segments
  "release": ["<role>", ...]               let go of something you were holding
  "select": <slot>                         switch to a discrete slot
  "text":  "<string>"                      type text
Angles are degrees. Never mention frames, ticks or key names: the runtime maps roles to this
device for you."""

_DURATION_TEMPLATE = (
    "Durations are milliseconds, between {min_ms} and {max_ms} per step."
)

_TIMING_TEMPLATE = """Timing you must plan around:
  Deciding costs about {latency_ms} ms of real game time. The frame you are looking at is
  already {lag_ms} ms old by the time your segment starts running.
  A segment covering less than about {latency_ms} ms of game time wastes most of its time
  waiting for you, so prefer segments of {min_useful_ms}..{max_useful_ms} ms.
  You cannot react inside a segment, so put a guard on anything that must interrupt it.
  "tail" is what keeps running after your last step while the next decision is computed:
  "hold" to keep moving, "release_move" to coast to a stop, "release_all" to freeze.
  "lease_ms" is a dead-man switch: if nothing new arrives within that time everything is
  released. Keep it short whenever running blind would be dangerous."""

_GUARD_TEMPLATE = """Guards are checked continuously by the runtime at no cost to you, and are
the only way to get fast reactions. Always guard segments longer than about {guard_hint_ms} ms.
Always available channels:
  "pixel.change" — how much the picture changes between consecutive frames, 0..1.
      Use "above" to stop on a sudden event, "below" with sustain_ms to detect being stuck.
  "pixel.drift" — how far the picture has moved away from the start of this segment, 0..1.
      Use "above" to stop once the scene has clearly changed (menu opened, area loaded)."""


def describe_segment_format(
    profile: BindingProfile,
    inference_latency_ms: int,
    signal_descriptions: Optional[Mapping[str, str]] = None,
) -> str:
    """生成给大模型的完整控制说明：格式 + 设备能力 + 时序预算 + 守卫通道。

    时序段用真实推理延迟参数化——大模型据此自己选择段长，而不是由代码硬编码一个 horizon。
    这是本契约替代"逐帧 horizon"的关键：段长由延迟与任务共同决定。

    Parameters
    ----------
    profile : BindingProfile
        目标游戏 / 设备的能力声明。
    inference_latency_ms : int
        实测的单轮推理墙钟延迟（毫秒），须为正。
    signal_descriptions : Optional[Mapping[str, str]]
        环境额外声明的守卫通道 → 一句话说明（如 ``{"health": "0..1 player health"}``）。

    Returns
    -------
    str
        可直接放入 system prompt 的多行英文说明。

    Raises
    ------
    ValueError
        inference_latency_ms 非正。
    """
    if inference_latency_ms <= 0:
        raise ValueError(f"inference_latency_ms 必须为正，收到 {inference_latency_ms}")
    sections = [
        _FORMAT_HEADER,
        _DURATION_TEMPLATE.format(
            min_ms=MIN_STEP_DURATION_MS, max_ms=MAX_STEP_DURATION_MS),
        describe_capabilities(profile),
        _TIMING_TEMPLATE.format(
            latency_ms=inference_latency_ms,
            lag_ms=inference_latency_ms,
            min_useful_ms=2 * inference_latency_ms,
            max_useful_ms=8 * inference_latency_ms,
        ),
        _GUARD_TEMPLATE.format(guard_hint_ms=2 * inference_latency_ms),
    ]
    if signal_descriptions:
        lines = ["This game also reports:"]
        for channel, description in signal_descriptions.items():
            lines.append(f'  "{channel}" — {description}')
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def describe_control_state(state: ControlState, profile: BindingProfile) -> str:
    """把当前控制状态渲染成 prompt 片段，让大模型知道自己正按着什么。

    latch 是跨段延续的，模型若不知道当前状态就会重复 hold 或忘记 release，这个片段是
    闭环的必要部分。
    """
    parts = []
    if state.latched_roles:
        parts.append(f"still holding: {', '.join(sorted(state.latched_roles))}")
    else:
        parts.append("holding nothing")
    if profile.slot_count > 0 and state.current_slot:
        parts.append(f"slot {state.current_slot} selected")
    if profile.menu_cursor and not profile.cursor_jumps_in_one_tick:
        # 光标能一 tick 直达时其当前位置对决策无用（下一次 point 直接覆盖），不占 prompt。
        parts.append(f"cursor at ({state.cursor_x:.2f}, {state.cursor_y:.2f})")
    return "Control state: " + "; ".join(parts) + "."


def describe_execution_report(report: ExecutionReport) -> str:
    """把上一段执行结果渲染成 prompt 片段（"我闭眼那段时间发生了什么"）。"""
    if report.tripped_guard is not None:
        guard = report.tripped_guard
        reason = guard.label or f"{guard.channel} {guard.comparison.value} {guard.threshold}"
        outcome = (
            f"interrupted after {report.executed_ms} ms of {report.planned_ms} ms "
            f"by guard: {reason}"
        )
    elif report.executed_ms < report.planned_ms:
        outcome = f"cut short after {report.executed_ms} ms of {report.planned_ms} ms"
    else:
        outcome = f"ran to completion in {report.executed_ms} ms"
    lines = [f"Previous segment {outcome}."]
    if report.tail_ms:
        lines.append(f"Then the tail policy ran for another {report.tail_ms} ms.")
    if report.observation_lag_ms:
        lines.append(
            f"The current frame is about {report.observation_lag_ms} ms behind live play."
        )
    return " ".join(lines)
