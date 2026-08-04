"""标准输入动作序列的有状态编译与逐 tick 调度。"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

_ACTION_SEQUENCE = re.compile(
    r"Device (?P<device>[^\r\n]+)\r?\n"
    r"Tick (?P<tick>[^\r\n]+)\r?\n"
    r"<action>(?P<actions>.*?)</action>",
    re.DOTALL,
)
_STREAM_HEADER = re.compile(
    r"Device (?P<device>[^\r\n]+)\r?\n"
    r"Tick (?P<tick>[^\r\n]+)\r?\n"
    r"<action>",
)
_REPEAT = re.compile(r"x([1-9]\d*)$")
_DEVICES = {"KeyboardMouse", "Gamepad", "Touch"}
_STREAM_BUFFER_LIMIT = 1_000_000
_KEYBOARD_KEYS = (
    {chr(code) for code in range(ord("A"), ord("Z") + 1)}
    | {str(number) for number in range(10)}
    | {"Up", "Down", "Left", "Right", "Shift", "Ctrl", "Alt"}
    | {"Space", "Enter", "Escape", "Tab", "Backspace", "Delete"}
    | {f"F{number}" for number in range(1, 13)}
)
_MOUSE_BUTTONS = {
    "MouseLeft",
    "MouseRight",
    "MouseMiddle",
    "MouseButton4",
    "MouseButton5",
}
_GAMEPAD_BUTTONS = {
    "A",
    "B",
    "X",
    "Y",
    "LeftBumper",
    "RightBumper",
    "LeftStickButton",
    "RightStickButton",
    "DPadUp",
    "DPadDown",
    "DPadLeft",
    "DPadRight",
    "Menu",
    "View",
}
_RESERVED_ACTION_TOKENS = {"Device", "Tick", "<action>", "</action>"}


class UnderflowPolicy(Enum):
    NOOP = "noop"
    REPEAT_LAST = "repeat_last"
    WAIT = "wait"


class DecisionKind(Enum):
    ACTION = "action"
    OBSERVE = "observe"
    WAIT = "wait"


class GenerationStatus(Enum):
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ActionTick:
    inputs: tuple[str, ...] = ()
    observe: bool = False


@dataclass(frozen=True)
class ActionSequence:
    device: str
    offset: int
    ticks: tuple[ActionTick, ...]


@dataclass(frozen=True)
class TickDecision:
    kind: DecisionKind
    tick: int
    action: ActionTick | None = None
    source: str | None = None
    revision: int = 0
    device: str | None = None


@dataclass(frozen=True)
class Submission:
    start_tick: int
    accepted_ticks: int
    expired_ticks: int
    overwritten_ticks: int
    cold_start: bool


@dataclass(frozen=True)
class GenerationTelemetry:
    request_id: str | None = None
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    time_to_first_token_ms: float | None = None
    total_generation_ms: float | None = None
    raw_provider_metrics: Mapping[str, object] | None = None


@dataclass(frozen=True)
class GenerationRecord:
    sequence_number: int
    generation_id: str
    status: GenerationStatus
    telemetry: GenerationTelemetry
    input_chunks: tuple[str, ...]
    started_at_ns: int
    first_content_at_ns: int | None
    first_action_at_ns: int | None
    completed_at_ns: int | None
    waiting_ticks: int
    waiting_before_first_content_ticks: int
    waiting_before_first_action_ticks: int
    noop_waiting_ticks: int
    repeated_waiting_ticks: int
    max_consecutive_waiting_ticks: int
    accepted_ticks: int
    expired_ticks: int
    overwritten_ticks: int
    submissions: tuple[Submission, ...]
    error: str | None = None

    @property
    def complete_input(self) -> str:
        return "".join(self.input_chunks)

    @property
    def time_to_first_content_ms(self) -> float | None:
        if self.first_content_at_ns is None:
            return None
        return (self.first_content_at_ns - self.started_at_ns) / 1_000_000

    @property
    def time_to_first_action_ms(self) -> float | None:
        if self.first_action_at_ns is None:
            return None
        return (self.first_action_at_ns - self.started_at_ns) / 1_000_000

    @property
    def first_content_to_complete_ms(self) -> float | None:
        if self.first_content_at_ns is None or self.completed_at_ns is None:
            return None
        return (self.completed_at_ns - self.first_content_at_ns) / 1_000_000

    @property
    def total_observed_generation_ms(self) -> float | None:
        if self.completed_at_ns is None:
            return None
        return (self.completed_at_ns - self.started_at_ns) / 1_000_000


@dataclass
class _GenerationRecord:
    sequence_number: int
    generation_id: str
    status: GenerationStatus
    telemetry: GenerationTelemetry
    started_at_ns: int
    input_chunks: list[str]
    submissions: list[Submission]
    first_content_at_ns: int | None = None
    first_action_at_ns: int | None = None
    completed_at_ns: int | None = None
    waiting_ticks: int = 0
    waiting_before_first_content_ticks: int = 0
    waiting_before_first_action_ticks: int = 0
    noop_waiting_ticks: int = 0
    repeated_waiting_ticks: int = 0
    current_waiting_ticks: int = 0
    max_consecutive_waiting_ticks: int = 0
    accepted_ticks: int = 0
    expired_ticks: int = 0
    overwritten_ticks: int = 0
    error: str | None = None

    def snapshot(self) -> GenerationRecord:
        return GenerationRecord(
            sequence_number=self.sequence_number,
            generation_id=self.generation_id,
            status=self.status,
            telemetry=self.telemetry,
            input_chunks=tuple(self.input_chunks),
            started_at_ns=self.started_at_ns,
            first_content_at_ns=self.first_content_at_ns,
            first_action_at_ns=self.first_action_at_ns,
            completed_at_ns=self.completed_at_ns,
            waiting_ticks=self.waiting_ticks,
            waiting_before_first_content_ticks=self.waiting_before_first_content_ticks,
            waiting_before_first_action_ticks=self.waiting_before_first_action_ticks,
            noop_waiting_ticks=self.noop_waiting_ticks,
            repeated_waiting_ticks=self.repeated_waiting_ticks,
            max_consecutive_waiting_ticks=self.max_consecutive_waiting_ticks,
            accepted_ticks=self.accepted_ticks,
            expired_ticks=self.expired_ticks,
            overwritten_ticks=self.overwritten_ticks,
            submissions=tuple(self.submissions),
            error=self.error,
        )


@dataclass
class _StreamingSequence:
    device: str
    next_tick: int
    first_segment: bool = True
    last_tick_observe: bool = False


@dataclass(frozen=True)
class _ScheduledTick:
    device: str
    action: ActionTick


def _merge_telemetry(
    original: GenerationTelemetry,
    update: GenerationTelemetry,
) -> GenerationTelemetry:
    values = {
        field: value
        for field in GenerationTelemetry.__dataclass_fields__
        if (value := getattr(update, field)) is not None
    }
    return replace(original, **values)


def _warn_invalid_tick(message: str) -> tuple[ActionTick, ...]:
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    return (ActionTick(),)


def _consume_arguments(
    tokens: list[str],
    start: int,
    count: int,
    converter: type[int | float],
) -> tuple[list[int | float], int] | None:
    end = start + count
    if end > len(tokens):
        return None
    try:
        values = [converter(token) for token in tokens[start:end]]
    except ValueError:
        return None
    return values, end


def _validate_action_tokens(device: str, tokens: list[str]) -> str | None:
    index = 0
    while index < len(tokens):
        command = tokens[index]
        if command in _RESERVED_ACTION_TOKENS or "<action" in command or "</action" in command:
            return "动作标签内不能出现设备声明、Tick 元数据或嵌套动作标签"
        if command == "Observe":
            return "Observe 只能出现在 tick 开头"

        if device == "KeyboardMouse":
            if command in _KEYBOARD_KEYS or command in _MOUSE_BUTTONS:
                index += 1
                continue
            if command == "MouseMove":
                parsed = _consume_arguments(tokens, index + 1, 2, int)
                if parsed is not None:
                    _, index = parsed
                    continue
                if index + 1 == len(tokens) or (
                    tokens[index + 1] in (_KEYBOARD_KEYS | _MOUSE_BUTTONS | {"MouseMove", "Scroll"})
                    and not tokens[index + 1].isdigit()
                ):
                    index += 1
                    continue
                return "MouseMove 需要两个整数参数，或省略参数表示 0 0"
            if command == "Scroll":
                parsed = _consume_arguments(tokens, index + 1, 1, int)
                if parsed is None:
                    return "Scroll 需要一个整数参数"
                _, index = parsed
                continue
            return f"KeyboardMouse 不支持输入 {command!r}"

        if device == "Gamepad":
            if command in _GAMEPAD_BUTTONS:
                index += 1
                continue
            if command in {"LeftStick", "RightStick"}:
                if index + 1 == len(tokens) or tokens[index + 1] in (
                    _GAMEPAD_BUTTONS | {"LeftStick", "RightStick", "LeftTrigger", "RightTrigger"}
                ):
                    index += 1
                    continue
                parsed = _consume_arguments(tokens, index + 1, 2, float)
                if parsed is None:
                    return f"{command} 需要两个浮点数参数，或省略参数表示 0.0 0.0"
                values, index = parsed
                if any(value < -1.0 or value > 1.0 for value in values):
                    warnings.warn(
                        f"{command} 参数超出 [-1.0, 1.0]，保留原始值",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                continue
            if command in {"LeftTrigger", "RightTrigger"}:
                parsed = _consume_arguments(tokens, index + 1, 1, float)
                if parsed is None:
                    return f"{command} 需要一个浮点数参数"
                values, index = parsed
                if values[0] < 0.0 or values[0] > 1.0:
                    warnings.warn(
                        f"{command} 参数超出 [0.0, 1.0]，保留原始值",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                continue
            return f"Gamepad 不支持输入 {command!r}"

        argument_counts = {"Tap": 2, "LongPress": 3, "Swipe": 5}
        if command == "Pinch":
            coordinates = _consume_arguments(tokens, index + 1, 2, int)
            if coordinates is None:
                return "Pinch 中心坐标必须是两个整数"
            coordinate_values, scale_index = coordinates
            scale = _consume_arguments(tokens, scale_index, 1, float)
            if scale is None:
                return "Pinch scale 必须是浮点数"
            scale_values, index = scale
            if any(value < 0 for value in coordinate_values):
                warnings.warn(
                    "Pinch 包含负坐标，保留原始值",
                    RuntimeWarning,
                    stacklevel=3,
                )
            if scale_values[0] <= 0:
                warnings.warn(
                    "Pinch scale 必须为正数，保留原始值",
                    RuntimeWarning,
                    stacklevel=3,
                )
            continue
        specification = argument_counts.get(command)
        if specification is None:
            return f"Touch 不支持输入 {command!r}"
        parsed = _consume_arguments(tokens, index + 1, specification, int)
        if parsed is None:
            return f"{command} 参数数量或类型错误"
        values, index = parsed
        if command in {"Tap", "LongPress", "Swipe"} and any(value < 0 for value in values):
            warnings.warn(
                f"{command} 包含负坐标或时长，保留原始值",
                RuntimeWarning,
                stacklevel=3,
            )
    return None


def _parse_action_ticks(actions: str, device: str) -> tuple[ActionTick, ...]:
    segments = actions.split(";")
    ticks: list[ActionTick] = []
    for segment in segments:
        if not segment.strip():
            ticks.extend(_warn_invalid_tick("空分号段按一个 NoOp tick 处理"))
            continue
        tokens = segment.split()
        observe = tokens[:1] == ["Observe"]
        if observe:
            tokens.pop(0)
        repeat = 1
        if tokens and (repeat_match := _REPEAT.fullmatch(tokens[-1])):
            repeat = int(repeat_match.group(1))
            tokens.pop()
        if tokens == ["NoOp"]:
            tokens.clear()
        elif not tokens:
            ticks.extend(_warn_invalid_tick("Observe 后必须包含动作或 NoOp；按 NoOp 处理"))
            continue
        elif "NoOp" in tokens:
            ticks.extend(_warn_invalid_tick("NoOp 不能与其他输入同时出现；按 NoOp 处理"))
            continue
        validation_error = _validate_action_tokens(device, tokens)
        if validation_error is not None:
            ticks.extend(_warn_invalid_tick(f"{validation_error}；按 NoOp 处理"))
            continue
        tick = ActionTick(tuple(tokens), observe)
        ticks.extend(replace(tick, observe=observe and index == 0) for index in range(repeat))
    return tuple(ticks)


def parse_action_sequence(text: str) -> ActionSequence:
    """把标准输入动作文本编译为逐 tick 序列。"""
    match = _ACTION_SEQUENCE.fullmatch(text)
    if match is None:
        raise ValueError("动作序列必须严格匹配 Device、Tick 和 <action> 结构")

    device = match.group("device").strip()
    if device not in _DEVICES:
        raise ValueError(f"不支持的设备类型：{device}")
    try:
        offset = int(match.group("tick").strip())
    except ValueError as error:
        raise ValueError("Tick 偏移必须是非负整数") from error
    if offset < 0:
        raise ValueError("Tick 偏移必须是非负整数")

    return ActionSequence(device, offset, _parse_action_ticks(match.group("actions"), device))


def extract_action_sequence_text(text: str) -> str:
    """从模型响应中提取唯一的 standard-input-action/v1 控制块。"""
    matches = tuple(_ACTION_SEQUENCE.finditer(text))
    if not matches:
        raise ValueError("教师响应中没有 standard-input-action/v1 动作序列")
    if len(matches) != 1:
        raise ValueError("教师响应必须且只能包含一个动作序列")
    return matches[0].group(0)


def parse_action_sequence_strict(text: str) -> ActionSequence:
    """严格解析动作序列，拒绝通常会降级为 NoOp 的非法 tick。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return parse_action_sequence(text)
    except RuntimeWarning as error:
        raise ValueError(str(error)) from error


def format_action_sequence(sequence: ActionSequence) -> str:
    """将动作序列序列化为 standard-input-action/v1 文本。"""
    if sequence.device not in _DEVICES:
        raise ValueError(f"不支持的设备类型：{sequence.device}")
    if sequence.offset < 0:
        raise ValueError("Tick 偏移必须是非负整数")
    segments = []
    for tick in sequence.ticks:
        tokens = (["Observe"] if tick.observe else []) + list(tick.inputs or ("NoOp",))
        validation_error = _validate_action_tokens(sequence.device, list(tick.inputs))
        if validation_error is not None:
            raise ValueError(validation_error)
        segments.append(" ".join(tokens))
    if not segments:
        raise ValueError("动作序列不能为空")
    return (
        f"Device {sequence.device}\nTick {sequence.offset}\n<action>{' ; '.join(segments)}</action>"
    )


class ActionSequenceCompiler:
    """维护无固定周期的逻辑环境 tick 和动作缓存，供 CraftGround 最大吞吐拉取。"""

    def __init__(
        self,
        underflow: UnderflowPolicy = UnderflowPolicy.NOOP,
        *,
        record_generations: bool = False,
        auto_observe: bool = False,
    ) -> None:
        self.underflow = underflow
        self.record_generations = record_generations
        self.auto_observe = auto_observe
        self._generation_records: list[_GenerationRecord] = []
        self._active_generation: _GenerationRecord | None = None
        self._next_generation_number = 0
        self.current_waiting_ticks = 0
        self.max_waiting_ticks = 0
        self.total_waiting_ticks = 0
        self.reset()

    def reset(self) -> None:
        if self._active_generation is not None:
            self._finish_generation(GenerationStatus.CANCELLED)
        self.current_waiting_ticks = 0
        self.current_tick = 0
        self._queue: dict[int, _ScheduledTick] = {}
        self._last_action: _ScheduledTick | None = None
        self._observed_tick: int | None = None
        self._revision = 0
        self._input_buffer = ""
        self._retained_text = ""
        self._streaming_sequence: _StreamingSequence | None = None

    def begin_generation(
        self,
        *,
        request_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        telemetry: GenerationTelemetry | None = None,
    ) -> str | None:
        if not self.record_generations:
            return None
        if self._active_generation is not None:
            raise RuntimeError("a generation is already active")
        supplied = telemetry or GenerationTelemetry()
        supplied = replace(
            supplied,
            request_id=request_id if request_id is not None else supplied.request_id,
            provider=provider if provider is not None else supplied.provider,
            model=model if model is not None else supplied.model,
        )
        sequence_number = self._next_generation_number
        self._next_generation_number += 1
        generation_id = f"generation-{sequence_number}"
        record = _GenerationRecord(
            sequence_number=sequence_number,
            generation_id=generation_id,
            status=GenerationStatus.STREAMING,
            telemetry=supplied,
            started_at_ns=time.monotonic_ns(),
            input_chunks=[],
            submissions=[],
        )
        self._generation_records.append(record)
        self._active_generation = record
        return generation_id

    def end_generation(
        self,
        *,
        telemetry: GenerationTelemetry | None = None,
        error: str | None = None,
    ) -> GenerationRecord | None:
        if not self.record_generations:
            return None
        record = self._require_active_generation()
        if telemetry is not None:
            record.telemetry = _merge_telemetry(record.telemetry, telemetry)
        record.error = error
        status = GenerationStatus.FAILED if error is not None else GenerationStatus.COMPLETED
        return self._finish_generation(status)

    def _finish_generation(self, status: GenerationStatus) -> GenerationRecord:
        record = self._require_active_generation()
        record.status = status
        record.completed_at_ns = time.monotonic_ns()
        self._active_generation = None
        return record.snapshot()

    def _require_active_generation(self) -> _GenerationRecord:
        if self._active_generation is None:
            raise RuntimeError("no generation is active")
        return self._active_generation

    def submit(self, text: str) -> Submission:
        self._record_input_chunk(text)
        submission = self._submit(parse_action_sequence(text))
        self._record_submissions((submission,))
        return submission

    def feed(self, chunk: str) -> tuple[Submission, ...]:
        """接收模型输出分片，并在每个完整 tick 到达后立即提交。"""
        self._record_input_chunk(chunk)
        self._input_buffer += chunk
        submissions: list[Submission] = []
        while True:
            if self._streaming_sequence is None:
                header = _STREAM_HEADER.search(self._input_buffer)
                if header is None:
                    break
                self._retained_text += self._input_buffer[: header.start()]
                self._input_buffer = self._input_buffer[header.end() :]
                stream = self._start_stream(header)
                if stream is None:
                    continue
                self._streaming_sequence = stream

            nested_header = _STREAM_HEADER.search(self._input_buffer)
            separator = self._input_buffer.find(";")
            closing_tag = self._input_buffer.find("</action>")
            event_positions = [position for position in (separator, closing_tag) if position >= 0]
            next_event = min(event_positions, default=-1)
            if nested_header is not None and (next_event < 0 or nested_header.start() < next_event):
                warnings.warn(
                    "未闭合的动作序列已被新的 Device/Tick 序列替代",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._retained_text += self._input_buffer[: nested_header.start()]
                self._input_buffer = self._input_buffer[nested_header.start() :]
                self._streaming_sequence = None
                continue
            if next_event < 0:
                break

            segment = self._input_buffer[:next_event]
            is_closing = next_event == closing_tag
            consumed = len("</action>") if is_closing else 1
            self._input_buffer = self._input_buffer[next_event + consumed :]
            submission = self._submit_stream_segment(segment)
            if submission is not None:
                submissions.append(submission)
            if is_closing:
                automatic = self._submit_automatic_observe()
                if automatic is not None:
                    submissions.append(automatic)
                self._streaming_sequence = None

        self._limit_stream_buffer()
        self._record_submissions(submissions)
        return tuple(submissions)

    def _record_input_chunk(self, chunk: str) -> None:
        record = self._active_generation
        if record is None:
            return
        record.input_chunks.append(chunk)
        if record.first_content_at_ns is None and chunk:
            record.first_content_at_ns = time.monotonic_ns()

    def _record_submissions(self, submissions: list[Submission] | tuple[Submission, ...]) -> None:
        record = self._active_generation
        if record is None:
            return
        record.submissions.extend(submissions)
        record.accepted_ticks += sum(item.accepted_ticks for item in submissions)
        record.expired_ticks += sum(item.expired_ticks for item in submissions)
        record.overwritten_ticks += sum(item.overwritten_ticks for item in submissions)
        if record.first_action_at_ns is None and any(item.accepted_ticks for item in submissions):
            record.first_action_at_ns = time.monotonic_ns()

    def _start_stream(self, header: re.Match[str]) -> _StreamingSequence | None:
        device = header.group("device").strip()
        tick_text = header.group("tick").strip()
        try:
            offset = int(tick_text)
        except ValueError:
            offset = -1
        if device not in _DEVICES or offset < 0:
            warnings.warn(
                f"跳过无效动作序列头：Device={device!r}, Tick={tick_text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            closing_tag = self._input_buffer.find("</action>")
            if closing_tag >= 0:
                self._input_buffer = self._input_buffer[closing_tag + len("</action>") :]
            return None
        return _StreamingSequence(device, self.current_tick + offset)

    def _submit_stream_segment(self, segment: str) -> Submission | None:
        stream = self._streaming_sequence
        if stream is None:
            raise RuntimeError("流式动作状态缺失")
        try:
            ticks = _parse_action_ticks(segment, stream.device)
        except ValueError as error:
            warnings.warn(f"跳过无效流式 tick：{error}", RuntimeWarning, stacklevel=2)
            return None

        submission = self._submit_at(
            stream.next_tick,
            stream.device,
            ticks,
            overwrite_future=stream.first_segment,
        )
        stream.next_tick += len(ticks)
        stream.first_segment = False
        if ticks:
            stream.last_tick_observe = ticks[-1].observe
        return submission

    def _submit_automatic_observe(self) -> Submission | None:
        stream = self._streaming_sequence
        if stream is None or not self.auto_observe or stream.last_tick_observe:
            return None
        submission = self._submit_at(
            stream.next_tick,
            stream.device,
            (ActionTick(observe=True),),
            overwrite_future=stream.first_segment,
        )
        stream.next_tick += 1
        stream.first_segment = False
        stream.last_tick_observe = True
        return submission

    def _limit_stream_buffer(self) -> None:
        if len(self._input_buffer) <= _STREAM_BUFFER_LIMIT:
            return
        overflow = len(self._input_buffer) - _STREAM_BUFFER_LIMIT
        self._retained_text += self._input_buffer[:overflow]
        self._input_buffer = self._input_buffer[overflow:]
        self._streaming_sequence = None
        warnings.warn("流式输入超过缓冲上限，已放弃未完成序列", RuntimeWarning, stacklevel=2)

    def _submit(self, sequence: ActionSequence) -> Submission:
        ticks = sequence.ticks
        if self.auto_observe and (not ticks or not ticks[-1].observe):
            ticks = (*ticks, ActionTick(observe=True))
        return self._submit_at(
            self.current_tick + sequence.offset,
            sequence.device,
            ticks,
            overwrite_future=True,
        )

    def _submit_at(
        self,
        requested_start: int,
        device: str,
        ticks: tuple[ActionTick, ...],
        *,
        overwrite_future: bool,
    ) -> Submission:
        cold_start = self._last_action is None and not self._queue
        expired = max(0, min(len(ticks), self.current_tick - requested_start))
        start = max(requested_start, self.current_tick)
        accepted = ticks[expired:]
        if overwrite_future:
            overwritten = sum(tick >= start for tick in self._queue)
            self._queue = {tick: action for tick, action in self._queue.items() if tick < start}
        else:
            overwritten = sum(start <= tick < start + len(accepted) for tick in self._queue)
        self._queue.update(
            (start + index, _ScheduledTick(device, action)) for index, action in enumerate(accepted)
        )
        self._revision += 1
        return Submission(start, len(accepted), expired, overwritten, cold_start)

    def pull(self) -> TickDecision:
        """读取当前环境 tick；只有 WAIT 下溢不产生可提交动作。"""
        scheduled = self._queue.get(self.current_tick)
        if scheduled is not None:
            if scheduled.action.observe and self._observed_tick != self.current_tick:
                return TickDecision(
                    DecisionKind.OBSERVE,
                    self.current_tick,
                    revision=self._revision,
                    device=scheduled.device,
                )
            return TickDecision(
                DecisionKind.ACTION,
                self.current_tick,
                scheduled.action,
                "sequence",
                self._revision,
                scheduled.device,
            )
        if self.underflow is UnderflowPolicy.WAIT:
            return TickDecision(
                DecisionKind.WAIT,
                self.current_tick,
                revision=self._revision,
            )
        if self.underflow is UnderflowPolicy.REPEAT_LAST and self._last_action is not None:
            return TickDecision(
                DecisionKind.ACTION,
                self.current_tick,
                replace(self._last_action.action, observe=False),
                "repeat_last",
                self._revision,
                self._last_action.device,
            )
        return TickDecision(
            DecisionKind.ACTION,
            self.current_tick,
            ActionTick(),
            "noop",
            self._revision,
        )

    def observed(self) -> None:
        scheduled = self._queue.get(self.current_tick)
        if scheduled is None or not scheduled.action.observe:
            raise RuntimeError("当前 tick 没有待确认的观察请求")
        self._observed_tick = self.current_tick

    def record_wait(self, decision: TickDecision) -> None:
        """记录一次调度等待；WAIT 不提交动作，也不推进环境逻辑 tick。"""
        if decision.kind is not DecisionKind.WAIT or decision.tick != self.current_tick:
            raise RuntimeError("只能记录当前 tick 的 WAIT 决策")
        if decision.revision != self._revision:
            raise RuntimeError("WAIT 决策已被新序列覆盖")
        self.current_waiting_ticks += 1
        self.total_waiting_ticks += 1
        self.max_waiting_ticks = max(self.max_waiting_ticks, self.current_waiting_ticks)
        record = self._active_generation
        if record is None:
            return
        record.waiting_ticks += 1
        record.current_waiting_ticks += 1
        record.max_consecutive_waiting_ticks = max(
            record.max_consecutive_waiting_ticks,
            record.current_waiting_ticks,
        )
        if record.first_content_at_ns is None:
            record.waiting_before_first_content_ticks += 1
        if record.first_action_at_ns is None:
            record.waiting_before_first_action_ticks += 1
        record.noop_waiting_ticks += 1

    def commit(self, decision: TickDecision) -> None:
        """确认环境已完成当前动作；调用后下一逻辑 tick 可立即执行。"""
        if decision.kind is not DecisionKind.ACTION or decision.tick != self.current_tick:
            raise RuntimeError("只能提交当前 tick 的动作决策")
        if decision.revision != self._revision:
            raise RuntimeError("动作决策已被新序列覆盖")
        if decision.source == "sequence":
            expected = self._queue.get(self.current_tick)
            if (
                expected is None
                or expected.action != decision.action
                or expected.device != decision.device
            ):
                raise RuntimeError("动作决策已被新序列覆盖")
            del self._queue[self.current_tick]
            self.current_waiting_ticks = 0
            if self._active_generation is not None:
                self._active_generation.current_waiting_ticks = 0
        elif decision.source in {"noop", "repeat_last"}:
            self.current_waiting_ticks += 1
            self.total_waiting_ticks += 1
            self.max_waiting_ticks = max(
                self.max_waiting_ticks,
                self.current_waiting_ticks,
            )
            record = self._active_generation
            if record is not None:
                record.waiting_ticks += 1
                record.current_waiting_ticks += 1
                record.max_consecutive_waiting_ticks = max(
                    record.max_consecutive_waiting_ticks,
                    record.current_waiting_ticks,
                )
                if record.first_content_at_ns is None:
                    record.waiting_before_first_content_ticks += 1
                if record.first_action_at_ns is None:
                    record.waiting_before_first_action_ticks += 1
                if decision.source == "noop":
                    record.noop_waiting_ticks += 1
                else:
                    record.repeated_waiting_ticks += 1
        self._last_action = (
            _ScheduledTick(
                decision.device,
                replace(decision.action, observe=False),
            )
            if decision.action is not None and decision.device is not None
            else None
        )
        self.current_tick += 1
        self._observed_tick = None

    @property
    def buffered_ticks(self) -> int:
        return len(self._queue)

    @property
    def pending_input(self) -> str:
        return self._input_buffer

    @property
    def retained_text(self) -> str:
        return self._retained_text

    def drain_retained_text(self) -> str:
        """取出已经确认位于动作序列之外的文本，并清空内部副本。"""
        text = self._retained_text
        self._retained_text = ""
        return text

    @property
    def generation_records(self) -> tuple[GenerationRecord, ...]:
        return tuple(record.snapshot() for record in self._generation_records)

    @property
    def latest_generation_record(self) -> GenerationRecord | None:
        if not self._generation_records:
            return None
        return self._generation_records[-1].snapshot()

    def get_generation_record(self, generation_id: str) -> GenerationRecord:
        for record in self._generation_records:
            if record.generation_id == generation_id:
                return record.snapshot()
        raise KeyError(generation_id)

    def drain_generation_records(self) -> tuple[GenerationRecord, ...]:
        if self._active_generation is not None:
            raise RuntimeError("cannot drain records while a generation is active")
        records = self.generation_records
        self._generation_records.clear()
        return records

    def discard_buffered_from_current_tick(self) -> int:
        """丢弃当前及未来未执行动作，供失败生成事务回滚。"""
        discarded = sum(tick >= self.current_tick for tick in self._queue)
        self._queue = {
            tick: action for tick, action in self._queue.items() if tick < self.current_tick
        }
        self._input_buffer = ""
        self._retained_text = ""
        self._streaming_sequence = None
        self._observed_tick = None
        self._revision += 1
        return discarded
