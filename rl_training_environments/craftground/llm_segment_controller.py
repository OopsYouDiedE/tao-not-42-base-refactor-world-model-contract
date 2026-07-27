# -*- coding: utf-8 -*-
"""大模型逐段控制 CraftGround 的闭环运行器：提示词 → 段文本 → 逐 tick 执行 → 回执。

对外接口:
    ControllerLimits    — 硬上限（轮数 / 总 tick / 墙钟），防止模型无限执行。
    GuardDetector       — 从帧序列判定 stuck / flash / scene_changed。
    AnthropicSegmentClient — 调用 Messages API 取一段动作文本。
    SegmentControllerSession — 一个回合的完整闭环。

本模块是 craftground 环境的运行期脚本层，只被人工/脚本调用，不被训练代码 import。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rl_training_environments.craftground.segment_prompt_builder import (
    PromptBuilder,
    SegmentRecord,
    render_request_zone,
)
from rl_training_environments.craftground.segment_text_codec import (
    AIM_DEGREES_PER_TICK_CAP,
    CURSOR_DEGREES_PER_PIXEL,
    CURSOR_HOME,
    CURSOR_SCREEN_HEIGHT_PIXELS,
    CURSOR_SCREEN_WIDTH_PIXELS,
    MAX_SEGMENT_TICKS,
    PHYSICAL_KEY_TO_V2,
    TICK_HZ,
    CompiledSegment,
    ParsedSegment,
    canonical_segment_text,
    compile_parsed_segment,
    parse_segment_text,
)

# 守卫判定阈值。stuck：帧间平均绝对差低于此值且持续足够久。
STUCK_MEAN_ABSOLUTE_DIFFERENCE = 1.2
STUCK_CONSECUTIVE_TICKS = 12
# stuck 改为位置判定：只有"按着移动键却没位移"才算卡住。像素判定会把
# "站定挖矿"误判成卡住（v2 的教训：31 轮里 14 轮被误砍）。
STUCK_POSITION_EPSILON_PER_TICK = 0.02
STUCK_POSITION_WINDOW_TICKS = 16
# flash：帧间亮度突增比例。
FLASH_BRIGHTNESS_RATIO = 1.28
# scene_changed：帧间平均绝对差超过此值。
SCENE_CHANGED_MEAN_ABSOLUTE_DIFFERENCE = 26.0
# 守卫在段头若干 tick 内不生效，避免上一段余速或渲染延迟误触发。
GUARD_WARMUP_TICKS = 8
# 建议段长下界的天花板：留出至少 20 tick 给上界，避免高延迟下区间倒挂。
SUGGESTED_MINIMUM_CEILING = 160
# 超过这个盲执行时长的段必须带 stop_if（3 秒看不到画面已经够久）。
GUARD_REQUIRED_ABOVE_TICKS = 60


@dataclass
class ControllerLimits:
    """硬上限。任一触及即停止本回合，绝不让模型无限执行。"""

    max_rounds: int = 40
    max_total_ticks: int = 6000
    max_wall_clock_seconds: float = 2400.0
    max_consecutive_same_guard: int = 3


class GuardDetector:
    """从相邻帧判定守卫触发。灰度下采样后比较，避免整帧逐像素开销。"""

    MOVEMENT_KEYS = ("forward", "back", "left", "right")

    def __init__(self, enabled_triggers: Sequence[str]) -> None:
        self.enabled = set(enabled_triggers)
        self._previous_small: Optional[np.ndarray] = None
        self._still_ticks = 0
        self._position_history: List[Tuple[float, float, float]] = []

    @staticmethod
    def _to_small_gray(frame_rgb: np.ndarray) -> np.ndarray:
        gray = frame_rgb.astype(np.float32).mean(axis=2)
        return gray[::8, ::8]

    def observe(self, frame_rgb: np.ndarray, tick_in_segment: int,
                position: Optional[Tuple[float, float, float]] = None,
                action: Optional[Dict[str, object]] = None) -> Optional[str]:
        """喂一帧（含位置与本 tick 动作），返回触发的守卫名或 None。"""
        small = self._to_small_gray(frame_rgb)
        previous = self._previous_small
        self._previous_small = small
        if position is not None:
            self._position_history.append(position)
        if previous is None or previous.shape != small.shape:
            return None
        difference = float(np.abs(small - previous).mean())
        if tick_in_segment < GUARD_WARMUP_TICKS:
            return None
        if "scene_changed" in self.enabled and difference > SCENE_CHANGED_MEAN_ABSOLUTE_DIFFERENCE:
            return "scene_changed"
        if "flash" in self.enabled:
            previous_brightness = max(float(previous.mean()), 1e-4)
            if float(small.mean()) / previous_brightness > FLASH_BRIGHTNESS_RATIO:
                return "flash"
        if "stuck" in self.enabled and self._is_position_stuck(action):
            return "stuck"
        return None

    def _is_position_stuck(self, action: Optional[Dict[str, object]]) -> bool:
        """按着移动键却在窗口内几乎没位移 = 卡住。没按移动键则永不算卡住。"""
        if action is None or not any(action.get(key) for key in self.MOVEMENT_KEYS):
            return False
        if len(self._position_history) <= STUCK_POSITION_WINDOW_TICKS:
            return False
        start = self._position_history[-STUCK_POSITION_WINDOW_TICKS - 1]
        end = self._position_history[-1]
        travelled = float(np.linalg.norm(np.array(end) - np.array(start)))
        return travelled < STUCK_POSITION_EPSILON_PER_TICK * STUCK_POSITION_WINDOW_TICKS


class AnthropicSegmentClient:
    """Anthropic Messages API 客户端（标准库 urllib，不引入新依赖）。"""

    def __init__(self, base_url: str, auth_token: str, model_name: str,
                 max_output_tokens: int = 700, timeout_seconds: float = 180.0,
                 max_attempts: int = 8, retry_backoff_seconds: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def request_segment(
        self, system_blocks: List[Dict[str, object]], messages: List[Dict[str, object]],
    ) -> Tuple[str, float, Dict[str, int]]:
        """发一次请求（含重试），返回 (文本, 耗时秒, usage)。重试用尽抛 RuntimeError。

        重试存在的理由：段是盲执行的，中途 API 抖动会让玩家停在半途，回合报废。
        重试期间玩家静止不动，等价于多等几秒，代价远小于报废整个回合。
        """
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._request_once(system_blocks, messages)
            except RuntimeError as error:
                last_error = str(error)
                if attempt >= self.max_attempts:
                    break
                delay = self.retry_backoff_seconds * attempt
                print(f"[client] 第 {attempt} 次请求失败（{last_error[:120]}），{delay:.0f}s 后重试",
                      flush=True)
                time.sleep(delay)
        raise RuntimeError(f"重试 {self.max_attempts} 次仍失败：{last_error}")

    def _request_once(
        self, system_blocks: List[Dict[str, object]], messages: List[Dict[str, object]],
    ) -> Tuple[str, float, Dict[str, int]]:
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.auth_token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"API HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"API 网络错误：{error}") from error
        except json.JSONDecodeError as error:
            raise RuntimeError(f"API 返回非 JSON：{error}") from error
        elapsed = time.time() - started
        blocks = body.get("content", [])
        text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        usage = body.get("usage", {})
        return text.strip(), elapsed, usage


def summarise_inventory(observation_full) -> Tuple[str, Dict[str, int]]:
    """从 obs["full"] 提取背包摘要文本与 {物品名: 数量} 字典。"""
    counts: Dict[str, int] = {}
    for item in getattr(observation_full, "inventory", []) or []:
        name = str(getattr(item, "translation_key", "") or "")
        count = int(getattr(item, "count", 0) or 0)
        if count <= 0 or not name:
            continue
        short = name.split(".")[-1]
        counts[short] = counts.get(short, 0) + count
    if not counts:
        return "空", counts
    parts = [f"{name}×{count}" for name, count in sorted(counts.items())]
    return ", ".join(parts), counts


def describe_control_state(last_action: Optional[Dict[str, object]]) -> str:
    """把上一段末的动作 dict 渲染成控制状态描述。"""
    if last_action is None:
        return "无按住的键"
    v2_to_physical = {value: key for key, value in PHYSICAL_KEY_TO_V2.items()}
    pressed = [
        v2_to_physical.get(key, key)
        for key, value in last_action.items()
        if value is True and key in v2_to_physical
    ]
    if not pressed:
        return "无按住的键"
    return "按住中：" + ", ".join(pressed)


def _blank_v2_action() -> Dict[str, object]:
    from craftground.environment.action_space import no_op_v2

    return dict(no_op_v2())


def _position_of(observation) -> Tuple[float, float, float]:
    full = observation["full"]
    return (float(full.x), float(full.y), float(full.z))


def _short_block_name(translation_key: str) -> str:
    """`block.minecraft.oak_log` → `oak_log`。空值返回空串。"""
    if not translation_key:
        return ""
    return translation_key.split(".")[-1]


# 准心射线命中类型：0=未命中，1=命中方块，2=命中实体（HitResult.Type）。
HIT_RESULT_MISS = 0
HIT_RESULT_BLOCK = 1
HIT_RESULT_ENTITY = 2


def probe_raycast(observation) -> Dict[str, object]:
    """读取准心射线，**仅用于落盘诊断**。

    这份读数不进提示词——模型只能看画面。它的用途是让我事后核对
    「模型以为自己瞄着什么」与「实际瞄着什么」的差距，据此判断是提示词
    表达不清还是代码设计有误。把它喂给模型等于替模型完成目标识别。
    """
    full = observation["full"]
    record: Dict[str, object] = {
        "position": [round(float(full.x), 2), round(float(full.y), 2), round(float(full.z), 2)],
        "is_on_ground": bool(full.is_on_ground),
    }
    raycast = getattr(full, "raycast_result", None)
    if raycast is None or raycast.type == HIT_RESULT_MISS:
        record["crosshair"] = None
        return record
    if raycast.type == HIT_RESULT_ENTITY:
        record["crosshair"] = {
            "kind": "entity",
            "name": _short_block_name(raycast.target_entity.translation_key),
        }
        return record
    block = raycast.target_block
    distance = float(np.linalg.norm(np.array([
        block.x + 0.5 - full.x, block.y + 0.5 - (full.y + 1.62), block.z + 0.5 - full.z,
    ])))
    record["crosshair"] = {
        "kind": "block",
        "name": _short_block_name(block.translation_key),
        "at": [block.x, block.y, block.z],
        "distance": round(distance, 2),
    }
    return record


@dataclass
class RoundOutcome:
    """一轮的完整记录，落盘成轨迹证据。"""

    round_index: int
    raw_model_text: str
    canonical_text: str
    parse_warnings: List[str]
    planned_ticks: int
    executed_ticks: int
    tripped_guard: Optional[str]
    latency_seconds: float
    usage: Dict[str, int]
    image_count: int
    inventory_before: Dict[str, int]
    inventory_after: Dict[str, int]
    position_before: Tuple[float, float, float]
    position_after: Tuple[float, float, float]
    yaw_before: float
    yaw_after: float
    pitch_before: float
    pitch_after: float
    requested_yaw_delta: float
    requested_pitch_delta: float
    observation_frame_paths: List[str] = field(default_factory=list)
    current_frame_path: str = ""
    truncation_notes: List[str] = field(default_factory=list)
    prompt_preview: str = ""
    # 诊断用：段首/段末的准心射线读数。模型看不到这两项，只供我事后核对。
    raycast_before: Dict[str, object] = field(default_factory=dict)
    raycast_after: Dict[str, object] = field(default_factory=dict)


class SegmentControllerSession:
    """一个回合的闭环：重置 → 逐轮取段 → 盲执行 → 记账。"""

    def __init__(self, environment, client: AnthropicSegmentClient, task_text: str,
                 limits: ControllerLimits, frame_output_directory,
                 initial_extra_commands: Sequence[str]) -> None:
        self.environment = environment
        self.client = client
        self.limits = limits
        self.frame_output_directory = frame_output_directory
        self.initial_extra_commands = list(initial_extra_commands)
        self.prompt_builder = PromptBuilder(task_text)
        self.rounds: List[RoundOutcome] = []
        self.total_ticks = 0
        self.started_at = time.time()
        self._last_latency: Optional[float] = None
        self._last_action: Optional[Dict[str, object]] = None
        self._consecutive_guard: Tuple[Optional[str], int] = (None, 0)
        self._current_observation = None
        self._stop_reason = ""
        # GUI 光标位置（归一化屏幕坐标）。背包每次打开都复位到正中。
        self._cursor_position: Tuple[float, float] = CURSOR_HOME

    # ── 环境交互 ──────────────────────────────────────────────────────────
    def fast_reset(self) -> None:
        """快速回档：/kill @p 重生 + 重跑初始命令（亚秒级，不重置方块）。"""
        observation, _ = self.environment.reset(options={
            "fast_reset": True,
            "extra_commands": list(self.initial_extra_commands),
        })
        for _ in range(8):
            observation, _, _, _, _ = self.environment.step(_blank_v2_action())
        self._current_observation = observation
        self._last_action = None

    def _step(self, action: Dict[str, object]):
        observation, _, _, _, _ = self.environment.step(action)
        self._current_observation = observation
        self.total_ticks += 1
        return observation

    # ── 自我修复 ──────────────────────────────────────────────────────────
    def _frame_of(observation) -> np.ndarray:
        return np.ascontiguousarray(observation["rgb"])

    # ── 段执行 ────────────────────────────────────────────────────────────
    def execute_segment(
        self, parsed: ParsedSegment, compiled: CompiledSegment,
    ) -> Tuple[int, Optional[str], List[Tuple[int, np.ndarray]], Optional[Tuple[int, np.ndarray]]]:
        """盲执行一段。返回 (已执行 tick, 触发的守卫, 观察帧, 中断帧)。"""
        detector = GuardDetector(parsed.stop_triggers)
        observation_ticks = set(compiled.observation_ticks)
        observation_frames: List[Tuple[int, np.ndarray]] = []
        interrupt_frame: Optional[Tuple[int, np.ndarray]] = None
        tripped: Optional[str] = None
        executed = 0

        # 段首先喂一帧建立守卫基线（不消耗 tick 预算之外的东西）。
        if self._current_observation is not None:
            detector.observe(self._frame_of(self._current_observation), 0,
                             position=_position_of(self._current_observation))

        for tick_index, action in enumerate(compiled.actions, start=1):
            observation = self._step(action)
            self._last_action = action
            executed = tick_index
            frame = self._frame_of(observation)
            if tick_index in observation_ticks:
                observation_frames.append((tick_index, frame.copy()))
            tripped = detector.observe(frame, tick_index,
                                       position=_position_of(observation), action=action)
            if tripped is not None:
                interrupt_frame = (tick_index, frame.copy())
                break
            if self.total_ticks >= self.limits.max_total_ticks:
                self._stop_reason = "触及总 tick 上限"
                break

        # 段末处置。
        if parsed.tail_mode == "stop":
            self._step(_blank_v2_action())
            self._last_action = None
        elif parsed.tail_mode == "freeze":
            frozen = dict(self._last_action or _blank_v2_action())
            for key in ("forward", "back", "left", "right"):
                frozen[key] = False
            frozen["camera_yaw"] = 0.0
            frozen["camera_pitch"] = 0.0
            self._step(frozen)
            self._last_action = frozen
        # hold：什么都不做，保留 _last_action 供下一段继承描述。

        if not observation_frames:
            observation_frames.append((executed, self._frame_of(self._current_observation).copy()))
        return executed, tripped, observation_frames, interrupt_frame

    # ── 真值记账 ──────────────────────────────────────────────────────────
    def _record_facts(self, parsed: ParsedSegment, compiled: CompiledSegment,
                      outcome: RoundOutcome) -> List[str]:
        """把可机械提取的偏差写进真值台账，返回本轮的截断说明。"""
        notes: List[str] = []
        ledger = self.prompt_builder.facts

        if abs(compiled.aim_truncated_yaw_deg) > 0.5 or abs(compiled.aim_truncated_pitch_deg) > 0.5:
            notes.append(
                f"视角被截断：yaw 少转 {compiled.aim_truncated_yaw_deg:+.1f}°、"
                f"pitch 少转 {compiled.aim_truncated_pitch_deg:+.1f}°。"
            )
            ledger.record_measured(
                "aim_cap",
                f"视角每 tick 最多 {AIM_DEGREES_PER_TICK_CAP:.0f}°。你写过一次超额的 Mouse，"
                f"结果少转了 {abs(compiled.aim_truncated_yaw_deg):.0f}°。大角度要给够 tick："
                f"转 N° 至少需要 ceil(N/{AIM_DEGREES_PER_TICK_CAP:.0f}) 个 tick。",
            )

        # 实际转角 vs 请求转角：这是"真值和写的角度有距离"的机械度量。
        actual_yaw_delta = ((outcome.yaw_after - outcome.yaw_before + 180.0) % 360.0) - 180.0
        # Minecraft 原始 pitch 负值 = 抬头，与契约 +pitch=抬头 相反，取负换到契约口径，
        # 否则会报出纯属符号错位的假偏差（v1 的教训）。
        actual_pitch_delta = -(outcome.pitch_after - outcome.pitch_before)
        if abs(outcome.requested_yaw_delta) > 3.0:
            error = actual_yaw_delta - outcome.requested_yaw_delta
            notes.append(
                f"yaw：请求 {outcome.requested_yaw_delta:+.1f}°，实际 {actual_yaw_delta:+.1f}°"
                f"（差 {error:+.1f}°）。"
            )
            if abs(error) > 5.0:
                ledger.record_measured(
                    "yaw_fidelity",
                    f"你请求的 yaw 与实际转动有偏差：请求 {outcome.requested_yaw_delta:+.0f}°"
                    f"实际只转了 {actual_yaw_delta:+.0f}°。写视角时把这个偏差算进去。",
                )
        if abs(outcome.requested_pitch_delta) > 3.0:
            error = actual_pitch_delta - outcome.requested_pitch_delta
            notes.append(
                f"pitch：请求 {outcome.requested_pitch_delta:+.1f}°，实际 {actual_pitch_delta:+.1f}°"
                f"（差 {error:+.1f}°）。"
            )

        moved = float(np.linalg.norm(
            np.array(outcome.position_after) - np.array(outcome.position_before)
        ))
        # 朝向也用契约口径报告：pitch 正 = 抬头，与模型书写的方向一致。
        notes.append(
            f"位移 {moved:.2f} 米，当前朝向 yaw {outcome.yaw_after:+.1f}°、"
            f"pitch {-outcome.pitch_after:+.1f}°（正=抬头）。"
        )
        # 光标真值：模型看不见箭头去哪了，必须明确告诉它落点，否则它只能靠猜。
        if parsed.cursor_items:
            end_x, end_y = compiled.cursor_end
            notes.append(f"GUI 光标现在停在 ({end_x:.3f}, {end_y:.3f})。")
            if abs(compiled.cursor_truncated_x) > 0.01 or abs(compiled.cursor_truncated_y) > 0.01:
                pixels_per_tick = AIM_DEGREES_PER_TICK_CAP / CURSOR_DEGREES_PER_PIXEL
                notes.append(
                    f"光标没走到位：x 差 {compiled.cursor_truncated_x:+.3f} 屏、"
                    f"y 差 {compiled.cursor_truncated_y:+.3f} 屏。给的 tick 不够——"
                    f"光标每 tick 最多走 {pixels_per_tick:.0f} 像素"
                    f"（{pixels_per_tick / CURSOR_SCREEN_WIDTH_PIXELS:.2f} 屏宽 / "
                    f"{pixels_per_tick / CURSOR_SCREEN_HEIGHT_PIXELS:.2f} 屏高）。"
                )
                ledger.record_measured(
                    "cursor_cap",
                    f"光标每 tick 最多走 {pixels_per_tick:.0f} 像素 = "
                    f"{pixels_per_tick / CURSOR_SCREEN_WIDTH_PIXELS:.2f} 屏宽 / "
                    f"{pixels_per_tick / CURSOR_SCREEN_HEIGHT_PIXELS:.2f} 屏高。"
                    f"跨半屏至少要 "
                    f"{int(0.5 * CURSOR_SCREEN_HEIGHT_PIXELS / pixels_per_tick) + 1} tick。",
                )

        if parsed.learn_text:
            ledger.record_inferred(f"learn_{outcome.round_index}", parsed.learn_text)
        return notes

    # ── 主循环 ────────────────────────────────────────────────────────────
    def _should_stop(self) -> Optional[str]:
        if len(self.rounds) >= self.limits.max_rounds:
            return f"触及轮数上限 {self.limits.max_rounds}"
        if self.total_ticks >= self.limits.max_total_ticks:
            return f"触及总 tick 上限 {self.limits.max_total_ticks}"
        elapsed = time.time() - self.started_at
        if elapsed >= self.limits.max_wall_clock_seconds:
            return f"触及墙钟上限 {self.limits.max_wall_clock_seconds:.0f}s"
        return None

    def _suggested_range(self) -> Tuple[int, int, int]:
        """按实测延迟推荐段长区间与需要守卫的门槛（单位 tick）。"""
        latency = self._last_latency if self._last_latency is not None else 0.8
        # 先把下界夹进合法区间，再推上界——否则高延迟下会出现下界 > 上界的矛盾建议。
        minimum = max(4, min(int(round(latency * 2.0 * TICK_HZ)), SUGGESTED_MINIMUM_CEILING))
        maximum = max(minimum + 20, min(int(round(latency * 8.0 * TICK_HZ)), MAX_SEGMENT_TICKS))
        # 守卫门槛由"盲执行多久"决定，与推理延迟无关：超过这个时长看不到画面就该有中断条件。
        return minimum, maximum, GUARD_REQUIRED_ABOVE_TICKS

    def run(self, goal_check) -> str:
        """跑到目标达成或触及上限。goal_check(counts, observation) → (done, note)。"""
        from PIL import Image

        while True:
            stop_reason = self._should_stop()
            if stop_reason is not None:
                return stop_reason

            observation = self._current_observation
            full = observation["full"]
            inventory_text, counts_before = summarise_inventory(full)
            position_before = (float(full.x), float(full.y), float(full.z))
            yaw_before, pitch_before = float(full.yaw), float(full.pitch)

            done, note = goal_check(counts_before, observation)
            if done:
                return f"目标达成：{note}"

            minimum, maximum, guard_above = self._suggested_range()
            request_text = render_request_zone(minimum, maximum, self._last_latency, guard_above)
            control_text = describe_control_state(self._last_action)
            current_frame = self._frame_of(observation)
            # 诊断读数：只落盘，不进提示词。
            raycast_before = probe_raycast(observation)
            system_blocks, messages, image_count = self.prompt_builder.build_messages(
                current_frame, control_text, request_text, inventory_text,
            )
            preview = self.prompt_builder.render_text_only_preview(control_text, request_text)

            round_index = len(self.rounds) + 1
            try:
                model_text, latency, usage = self.client.request_segment(system_blocks, messages)
            except RuntimeError as error:
                return f"API 失败于第 {round_index} 轮：{error}"
            self._last_latency = latency

            parsed = parse_segment_text(model_text)
            # 光标状态跨段延续：GUI 不关，光标就停在上段末尾的位置。
            compiled = compile_parsed_segment(parsed, cursor_start=self._cursor_position)
            self._cursor_position = compiled.cursor_end
            requested_yaw = sum(item.x for item in parsed.aim_items)
            requested_pitch = sum(item.y for item in parsed.aim_items)

            executed, tripped, observation_frames, interrupt_frame = self.execute_segment(
                parsed, compiled,
            )

            after_full = self._current_observation["full"]
            _, counts_after = summarise_inventory(after_full)
            outcome = RoundOutcome(
                round_index=round_index,
                raw_model_text=model_text,
                canonical_text=canonical_segment_text(parsed),
                parse_warnings=list(parsed.warnings),
                planned_ticks=parsed.duration_ticks,
                executed_ticks=executed,
                tripped_guard=tripped,
                latency_seconds=latency,
                usage=dict(usage),
                image_count=image_count,
                inventory_before=counts_before,
                inventory_after=counts_after,
                position_before=position_before,
                position_after=(float(after_full.x), float(after_full.y), float(after_full.z)),
                yaw_before=yaw_before,
                yaw_after=float(after_full.yaw),
                pitch_before=pitch_before,
                pitch_after=float(after_full.pitch),
                requested_yaw_delta=requested_yaw,
                requested_pitch_delta=requested_pitch,
                prompt_preview=preview,
                raycast_before=raycast_before,
                raycast_after=probe_raycast(self._current_observation),
            )
            outcome.truncation_notes = self._record_facts(parsed, compiled, outcome)

            # 落盘帧证据。
            frame_paths: List[str] = []
            for tick, frame in observation_frames:
                path = self.frame_output_directory / f"r{round_index:02d}_t{tick:03d}.png"
                Image.fromarray(frame).save(path)
                frame_paths.append(path.name)
            if interrupt_frame is not None:
                tick, frame = interrupt_frame
                path = self.frame_output_directory / f"r{round_index:02d}_t{tick:03d}_guard.png"
                Image.fromarray(frame).save(path)
                frame_paths.append(path.name)
            current_path = self.frame_output_directory / f"r{round_index:02d}_end.png"
            Image.fromarray(self._frame_of(self._current_observation)).save(current_path)
            outcome.observation_frame_paths = frame_paths
            outcome.current_frame_path = current_path.name
            self.rounds.append(outcome)

            state_note = ""
            gained = {
                name: counts_after.get(name, 0) - counts_before.get(name, 0)
                for name in set(counts_after) | set(counts_before)
                if counts_after.get(name, 0) != counts_before.get(name, 0)
            }
            if gained:
                state_note = "背包变化：" + ", ".join(
                    f"{name} {delta:+d}" for name, delta in sorted(gained.items())
                )
            record = SegmentRecord(
                index=round_index,
                canonical_text=outcome.canonical_text,
                why_text=parsed.why_text,
                executed_ticks=executed,
                planned_ticks=parsed.duration_ticks,
                tripped_guard=tripped,
                parse_warnings=list(parsed.warnings),
                truncation_notes=outcome.truncation_notes,
                observation_frames=observation_frames,
                interrupt_frame=interrupt_frame,
                state_note=state_note,
            )
            self.prompt_builder.append_record(record)

            # 退化循环保护：同一守卫连续触发上限次则强制压缩，逼模型换招。
            if tripped is not None and tripped == self._consecutive_guard[0]:
                count = self._consecutive_guard[1] + 1
                self._consecutive_guard = (tripped, count)
                if count >= self.limits.max_consecutive_same_guard:
                    self.prompt_builder.force_compress_on_repeat(tripped, count)
                    self._consecutive_guard = (None, 0)
            elif tripped is not None:
                self._consecutive_guard = (tripped, 1)
            else:
                self._consecutive_guard = (None, 0)
