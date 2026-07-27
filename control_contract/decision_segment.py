# -*- coding: utf-8 -*-
"""一次大模型推理的输出单位：时间延展的决策段（Segment）及其守卫与租约。

对外接口：
    Aim / Move / Point / Press / Hold / Release / Select / TextInput — 原语。
    Step — 一组并发原语 + 毫秒时长，段内串行。
    Guard / GuardComparison — 每 tick 求值的中断条件（帧级反应性来源）。
    TailPolicy — 段执行完、下一段尚未到达期间的兜底行为。
    Segment — 完整决策段：steps + guards + tail + lease_ms。
    ControlState — 跨段延续的 latch / 光标 / 槽位状态。
    ExecutionReport — 段实际执行结果，回灌给下一轮推理。

时间模型（针对纯大模型控制的核心设计）：
  一次推理 = 一个 Segment = 数百毫秒到数秒的行为程序，绝不是一帧。段内每 tick 的设备
  输入由编译器确定性展开，运行时无需推理即可执行；同时每 tick 求值 guards，任一命中就
  立即截断本段、抓取观测并请求下一轮推理——**反应性由守卫提供，语义由大模型提供**。
  推理期间运行时执行 tail 策略，`lease_ms` 是死人开关：超时未收到新段即强制安全态。

单位约定：角度用度，时长用毫秒，力度与屏幕坐标归一化到 0..1。大模型永远看不到 tick，
故同一段文本可跨 20Hz / 30Hz / 60Hz 环境复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# 单步时长的合法区间（毫秒）。下限防止大模型写出短于一个 tick 的空步；上限约束单步盲执行。
MIN_STEP_DURATION_MS = 20
MAX_STEP_DURATION_MS = 20_000

# 段租约的默认值与上限（毫秒）。租约 = 运行时在没有新段时愿意继续执行 tail 的时间。
DEFAULT_LEASE_MS = 2_000
MAX_LEASE_MS = 30_000


@dataclass(frozen=True)
class Aim:
    """相对视角增量（度）。正 yaw 右转，正 pitch 上抬。"""

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass(frozen=True)
class Move:
    """极坐标位移意图。

    Attributes
    ----------
    direction_deg : float
        相对角色朝向的方向：0=正前，90=正右，180=后，270=左。任意实数，编译器归一化到 [0,360)。
    power : float
        力度 0..1。连续力度的摇杆按原值生效；只有单档力度的摇杆（键盘）仅用它判定是否移动。
    """

    direction_deg: float = 0.0
    power: float = 1.0


@dataclass(frozen=True)
class Point:
    """界面指向：归一化屏幕坐标，(0,0) 左上，(1,1) 右下。"""

    x: float = 0.5
    y: float = 0.5


@dataclass(frozen=True)
class Press:
    """点按一个角色：本步开始时按下，本步内松开（不跨段）。"""

    role: str
    repeat: int = 1


@dataclass(frozen=True)
class Hold:
    """按住一个角色：跨段 latch，直到 Release、租约到期或 tail 策略释放。"""

    role: str


@dataclass(frozen=True)
class Release:
    """松开一个被 latch 的角色。"""

    role: str


@dataclass(frozen=True)
class Select:
    """选择离散槽位（1 起）。CYCLE_ONLY 设备上编译为最短方向的循环切换。"""

    slot: int


@dataclass(frozen=True)
class TextInput:
    """文本输入（仅 supports_text 的 profile 可用）。"""

    text: str


# 内建观测通道：无需任何游戏集成即可计算，保证守卫机制对"大多数游戏"可用。
# 相邻观测帧的平均绝对像素差，归一化到 0..1。
PIXEL_CHANGE_CHANNEL = "pixel.change"
# 自本段开始以来累计的像素变化量（同样 0..1），用于"画面已经翻页/场景已切换"判定。
PIXEL_DRIFT_CHANNEL = "pixel.drift"
BUILTIN_CHANNELS: Tuple[str, ...] = (PIXEL_CHANGE_CHANNEL, PIXEL_DRIFT_CHANNEL)


class GuardComparison(str, Enum):
    """守卫的比较方式。"""

    BELOW = "below"                  # 通道值 < threshold
    ABOVE = "above"                  # 通道值 > threshold
    DELTA_ABOVE = "delta_above"      # |通道值 - 段起始值| > threshold


@dataclass(frozen=True)
class Guard:
    """每 tick 求值的中断条件：命中即截断本段、抓观测、请求下一轮推理。

    守卫是本控制契约获得帧级反应性而不做帧级推理的唯一手段：条件在编译期固化为纯数值
    判定，运行时零推理成本。通道既可是内建像素通道（任何游戏可用），也可是环境声明的
    数值信号（血量、速度、进度等，由环境侧 signal profile 提供）。

    Attributes
    ----------
    channel : str
        观测通道名（BUILTIN_CHANNELS 之一，或环境声明的信号名）。
    comparison : GuardComparison
        比较方式。
    threshold : float
        阈值，单位随通道。
    sustain_ms : int
        条件需连续成立的时长（毫秒）；0 表示瞬时命中即触发。用于抑制抖动误触。
    label : str
        人类可读标签，触发后写进 ExecutionReport 回灌给大模型。
    """

    channel: str
    comparison: GuardComparison
    threshold: float
    sustain_ms: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.channel.strip():
            raise ValueError("Guard.channel 不能为空")
        if self.sustain_ms < 0:
            raise ValueError(f"Guard.sustain_ms 不能为负，收到 {self.sustain_ms}")


@dataclass
class Step:
    """一个决策步：一组并发原语 + 毫秒时长。段内各步按顺序串行执行。

    同一步内的原语并发生效（可同时 move + aim + hold），互不排斥。位移用极坐标表达，
    结构上不可能出现"前后同按"这类冲突——这是取代旧互斥组校验的做法。

    Attributes
    ----------
    duration_ms : int
        本步时长，[MIN_STEP_DURATION_MS, MAX_STEP_DURATION_MS]。编译器按 tick_hz 取整为
        至少 1 tick。
    aim : Optional[Aim]
        视角增量。
    move : Optional[Move]
        位移意图；不给则本步不主动位移（但已 latch 的移动角色仍然保持）。
    point : Optional[Point]
        界面指向。
    presses : List[Press]
        本步点按的角色。
    holds : List[Hold]
        本步开始 latch 的角色。
    releases : List[Release]
        本步松开的角色。
    select : Optional[Select]
        槽位选择。
    text : Optional[TextInput]
        文本输入。
    """

    duration_ms: int = MIN_STEP_DURATION_MS
    aim: Optional[Aim] = None
    move: Optional[Move] = None
    point: Optional[Point] = None
    presses: List[Press] = field(default_factory=list)
    holds: List[Hold] = field(default_factory=list)
    releases: List[Release] = field(default_factory=list)
    select: Optional[Select] = None
    text: Optional[TextInput] = None

    def __post_init__(self) -> None:
        if not isinstance(self.duration_ms, int):
            self.duration_ms = int(round(float(self.duration_ms)))
        if not MIN_STEP_DURATION_MS <= self.duration_ms <= MAX_STEP_DURATION_MS:
            raise ValueError(
                f"Step.duration_ms 须在 [{MIN_STEP_DURATION_MS}, {MAX_STEP_DURATION_MS}]，"
                f"收到 {self.duration_ms}"
            )
        held = {item.role for item in self.holds}
        released = {item.role for item in self.releases}
        conflict = held & released
        if conflict:
            raise ValueError(f"同一步内既 hold 又 release 同一角色：{sorted(conflict)}")


class TailPolicy(str, Enum):
    """段执行完毕、下一段尚未到达期间（即推理延迟窗口）运行时的行为。

    - ``HOLD``：保持全部 latch（继续跑 / 继续按住），适合"沿路前进"这类连续行为。
    - ``RELEASE_MOVE``：松开位移类 latch，保留其他按住（原地停下但仍举盾/瞄准）。
    - ``RELEASE_ALL``：松开全部 latch，回到静止安全态。默认值。
    """

    HOLD = "hold"
    RELEASE_MOVE = "release_move"
    RELEASE_ALL = "release_all"


@dataclass
class Segment:
    """一次推理产出的完整决策段。

    Attributes
    ----------
    steps : List[Step]
        串行执行的决策步；至少一个。
    guards : List[Guard]
        每 tick 求值的中断条件；任一命中即截断本段并请求下一轮推理。
    tail : TailPolicy
        本段结束后、新段到达前的兜底行为。
    lease_ms : int
        死人开关：从本段结束起，最多按 tail 策略维持这么久；超时强制 RELEASE_ALL。
    intent : str
        大模型对本段意图的一句话说明，仅用于日志与训练样本可读性，不参与执行。
    """

    steps: List[Step] = field(default_factory=list)
    guards: List[Guard] = field(default_factory=list)
    tail: TailPolicy = TailPolicy.RELEASE_ALL
    lease_ms: int = DEFAULT_LEASE_MS
    intent: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("Segment 至少需要一个 Step")
        if not isinstance(self.lease_ms, int):
            self.lease_ms = int(round(float(self.lease_ms)))
        if not 0 <= self.lease_ms <= MAX_LEASE_MS:
            raise ValueError(f"Segment.lease_ms 须在 [0, {MAX_LEASE_MS}]，收到 {self.lease_ms}")

    def total_duration_ms(self) -> int:
        """本段全部步的名义总时长（毫秒，未计守卫提前截断）。"""
        return sum(step.duration_ms for step in self.steps)


@dataclass
class ControlState:
    """跨段延续的控制状态。运行时持有，逐段传入编译器并取回更新值。

    Attributes
    ----------
    latched_roles : frozenset[str]
        当前被 Hold 按住的核心角色。
    cursor_x, cursor_y : float
        界面光标当前归一化位置。光标能单 tick 直达的设备上意义不大（下次 point 直接覆盖），
        限速逼近的设备上则是必要的延续状态。
    current_slot : int
        当前选中的槽位（1 起；0 表示未知 / 无槽位概念）。
    """

    latched_roles: frozenset = frozenset()
    cursor_x: float = 0.5
    cursor_y: float = 0.5
    current_slot: int = 0


@dataclass
class ExecutionReport:
    """一段实际执行的结果，回灌给下一轮推理，让大模型知道"我闭眼这段时间发生了什么"。

    Attributes
    ----------
    executed_ms : int
        本段实际执行的毫秒数（守卫截断时小于名义时长）。
    planned_ms : int
        本段名义总时长（毫秒）。
    completed_steps : int
        完整执行完的步数。
    tripped_guard : Optional[Guard]
        触发截断的守卫；None 表示本段自然跑完。
    tail_ms : int
        段末 tail 策略实际维持的毫秒数（即观测到下一段生效之间的推理延迟）。
    observation_lag_ms : int
        本轮推理所依据的观测帧距"新段真正开始执行"的时间差（毫秒）。大模型据此知道
        自己看到的画面有多旧。
    state : ControlState
        本段结束后的控制状态。
    """

    executed_ms: int = 0
    planned_ms: int = 0
    completed_steps: int = 0
    tripped_guard: Optional[Guard] = None
    tail_ms: int = 0
    observation_lag_ms: int = 0
    state: ControlState = field(default_factory=ControlState)
