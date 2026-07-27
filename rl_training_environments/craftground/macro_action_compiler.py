# -*- coding: utf-8 -*-
"""回合宏 → 逐 tick V2 动作序列 + 观察点的纯逻辑编译器。

手写轨迹录制器的核心：作者用"回合"（Turn）写轨迹——一个回合 = 大模型一次推理边界 =
一条训练样本的切点。一个回合内可并发多个动作（组合键），编译器把回合序列**确定性地**
展开成逐 tick 的 CraftGround V2 动作 dict 序列，并计算观察点落点。

回合模型（见记忆 craftground-recorder-capability-probe-2026-07-25 与用户定稿）：
  - **长按 latch 键**（HOLD_KEYS = wasd + shift + ctrl）：设定一个 tick 预算（时长），
    作为**后台倒计时**跨多个回合延续——"没设置就不覆盖"，某回合不再指定该键则沿用其
    剩余预算继续按住，预算归零自动松开；重新指定同一键则覆盖刷新预算。
  - **回合前台时长**：click（TAP_TICKS）/ 相机（CAM_TICKS）/ 非 latch 键的定时长按 /
    纯等待 wait_ticks 决定该回合推进多少 tick。latch 键只在后台倒计时，不单独推进时间轴。
  - **组合键**：一个回合可同时按多个离散键（jump+attack 等），互不排斥。录制器不做互斥
    组校验（跨互斥组的同按会在策略解码端 net/action_token_codec 被消解，仅 README 提示）。
  - **相机**两种结构：① delta（delta_yaw/delta_pitch，度）；② screen（绝对屏幕坐标，
    GUI 光标定位）。均固定 CAM_TICKS(=2) tick。delta 模式单回合旋转 ≤ CAM_TICKS×CAM_MAX_DEG
    (=36°)，超限报错（否则 BC 编码端 action_contract.deg_to_bins 会静默截断）。
  - 所有回合展开完后，仍有剩余预算的 latch 键以**尾段**实现（按 max_blind_ticks 自动切分）。
  - 原始 Minecraft 命令（mc）不进逐帧动作，单独归到该 tick 的 commands 列表，由录制器
    决定发送通道（fast_reset 的 extra_commands 或 env.add_command）。

对外接口：
    Turn / turn(...) — 回合宏及其构造器
    MinecraftCommand / minecraft_command — 原始命令宏
    HOLD_KEYS / TAP_TICKS / CAM_TICKS — 回合契约常量
    build_noop_action() — 全零 V2 动作 dict
    compile_macros(commands, max_blind_ticks, manual_observation_ticks) -> CompiledTrajectory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rl_training_environments.craftground.action_contract import CAM_MAX_DEG, V2_KEYS

# 二值键之外的两个连续相机字段（度），构成 CraftGround V2 完整动作 dict 的键集合。
CAMERA_KEYS = ("camera_yaw", "camera_pitch")

# 全部 V2 二值键集合（离散键 + 长按键）。
VALID_BINARY_KEYS = set(V2_KEYS)

# 长按 latch 键：wasd（移动）+ shift（潜行 sneak）+ ctrl（冲刺 sprint）。
# 只有这些键的时长会作为后台倒计时预算跨回合延续；其余键的定时长按仅限本回合内。
HOLD_KEYS = frozenset(("forward", "back", "left", "right", "sneak", "sprint"))

# 离散点按（click）固定 tick 数。
TAP_TICKS = 2
# 相机（delta / screen 两种结构）固定 tick 数。
CAM_TICKS = 2

# 默认最大盲执行 tick 数：任何连续无观察点的动作段超过此值就自动补插观察点。
# 20 tick ≈ 1 秒（Minecraft 20Hz）；15 秒是"模型最长可闭眼"的经验上限，可由调用方覆盖。
DEFAULT_MAX_BLIND_TICKS = 300

# 相机结构标签。
CAMERA_NONE = "none"
CAMERA_DELTA = "delta"
CAMERA_SCREEN = "screen"


def build_noop_action() -> Dict[str, object]:
    """构造一个全零的 V2 动作 dict（所有二值键 False，相机增量 0.0）。

    与 craftground.environment.action_space.no_op_v2() 的键集合一致，但本模块不依赖
    craftground 运行时（保持纯逻辑可单测）。录制器把本 dict 直接喂给 env.step。

    Returns
    -------
    Dict[str, object]
        键集 = V2_KEYS ∪ CAMERA_KEYS；二值键值 False，相机字段 float 0.0。
    """
    action: Dict[str, object] = {key: False for key in V2_KEYS}
    action["camera_yaw"] = 0.0
    action["camera_pitch"] = 0.0
    return action


@dataclass
class MacroCommand:
    """一条宏命令的基类。子类通过 expand() 产出自己的逐 tick 动作。

    Attributes
    ----------
    kind : str
        宏类型标签（用于导出与界面展示）。
    label : str
        可选的人类可读标签（界面显示，不影响编译）。
    """

    kind: str
    label: str = ""

    def minecraft_commands(self) -> List[str]:
        """本宏附带的原始 Minecraft 命令（仅 MinecraftCommand 非空）。"""
        return []


@dataclass
class Turn(MacroCommand):
    """一个回合 = 一次推理 = 一个观察段；回合内可并发多个动作（组合键）。

    Attributes
    ----------
    holds : Dict[str, int]
        本回合设定/刷新的长按键 → 时长（tick，≥1）。键 ∈ V2_KEYS。其中 ∈ HOLD_KEYS 的键
        作为后台倒计时预算跨回合延续；∉ HOLD_KEYS 的键（离散键的定时长按）仅限本回合内、
        按各自时长为真（短键提前松开），不跨回合。
    clicks : List[str]
        本回合点按的离散键（各 TAP_TICKS tick），键 ∈ V2_KEYS，仅限本回合。自由组合、不互斥。
    release : List[str]
        本回合要松开（停止倒计时延续）的 latch 键，键 ∈ HOLD_KEYS。
    camera_mode : str
        相机结构：CAMERA_NONE / CAMERA_DELTA / CAMERA_SCREEN。
    delta_yaw, delta_pitch : float
        camera_mode==CAMERA_DELTA 时的相机增量（度）；单回合 |Δ| ≤ CAM_TICKS×CAM_MAX_DEG。
    screen_x, screen_y : float
        camera_mode==CAMERA_SCREEN 时的绝对屏幕坐标驱动量（GUI 光标定位，屏幕空间，不做度校验）。
    gui_cursor : bool
        相机是否为 GUI 光标语义（导出标注；CraftGround 里 GUI 打开时 camera 字段驱动屏幕光标）。
    wait_ticks : int
        纯等待 tick 数（≥0）；无任何前台动作时用它推进时间轴（等熔炉烧/等岩浆流），
        期间后台 latch 键继续。
    """

    holds: Dict[str, int] = field(default_factory=dict)
    clicks: List[str] = field(default_factory=list)
    release: List[str] = field(default_factory=list)
    camera_mode: str = CAMERA_NONE
    delta_yaw: float = 0.0
    delta_pitch: float = 0.0
    screen_x: float = 0.0
    screen_y: float = 0.0
    gui_cursor: bool = False
    wait_ticks: int = 0

    def __post_init__(self) -> None:
        for key, ticks in self.holds.items():
            if key not in VALID_BINARY_KEYS:
                raise ValueError(f"Turn.holds: 未知的 V2 二值键 {key!r}")
            if not isinstance(ticks, int) or ticks < 1:
                raise ValueError(f"Turn.holds[{key!r}]: 时长必须是 >=1 的整数，收到 {ticks!r}")
        for key in self.clicks:
            if key not in VALID_BINARY_KEYS:
                raise ValueError(f"Turn.clicks: 未知的 V2 二值键 {key!r}")
        for key in self.release:
            if key not in HOLD_KEYS:
                raise ValueError(f"Turn.release: 只能松开 latch 键 {sorted(HOLD_KEYS)}，收到 {key!r}")
        if self.camera_mode not in (CAMERA_NONE, CAMERA_DELTA, CAMERA_SCREEN):
            raise ValueError(f"Turn.camera_mode 非法：{self.camera_mode!r}")
        if self.camera_mode == CAMERA_DELTA:
            limit = CAM_TICKS * CAM_MAX_DEG
            if abs(self.delta_yaw) > limit or abs(self.delta_pitch) > limit:
                raise ValueError(
                    f"Turn 相机增量超单回合上限 ±{limit}°（CAM_TICKS×CAM_MAX_DEG）："
                    f"Δyaw={self.delta_yaw}, Δpitch={self.delta_pitch}。请拆成多个回合。"
                )
        if self.wait_ticks < 0:
            raise ValueError(f"Turn.wait_ticks 必须 >=0，收到 {self.wait_ticks}")
        if not (self.holds or self.clicks or self.release
                or self.camera_mode != CAMERA_NONE or self.wait_ticks):
            raise ValueError("Turn 为空回合（无任何键/相机/等待）；纯等待请用 wait_ticks 表达")

    def latched_holds(self) -> Dict[str, int]:
        """本回合设定的 latch 键（∈ HOLD_KEYS）→ 时长预算。"""
        return {key: ticks for key, ticks in self.holds.items() if key in HOLD_KEYS}

    def within_turn_holds(self) -> Dict[str, int]:
        """本回合的非 latch 定时长按（∉ HOLD_KEYS）→ 时长，仅限本回合内。"""
        return {key: ticks for key, ticks in self.holds.items() if key not in HOLD_KEYS}

    def foreground_length(self) -> int:
        """本回合前台推进的 tick 数（latch 键在后台倒计时，不计入）。

        Returns
        -------
        int
            max(click 有则 TAP_TICKS, 相机有则 CAM_TICKS, 非 latch 长按各自时长, wait_ticks)；
            全为后台 latch/release 时返回 0（本回合不单独产生帧，预算留给后续回合/尾段消费）。
        """
        lengths: List[int] = []
        if self.clicks:
            lengths.append(TAP_TICKS)
        if self.camera_mode != CAMERA_NONE:
            lengths.append(CAM_TICKS)
        lengths.extend(self.within_turn_holds().values())
        if self.wait_ticks:
            lengths.append(self.wait_ticks)
        return max(lengths) if lengths else 0

    def _camera_per_tick(self) -> Tuple[float, float]:
        """相机每 tick 增量（delta 度均摊 / screen 屏幕坐标均摊到 CAM_TICKS）。"""
        if self.camera_mode == CAMERA_DELTA:
            return self.delta_yaw / CAM_TICKS, self.delta_pitch / CAM_TICKS
        if self.camera_mode == CAMERA_SCREEN:
            return self.screen_x / CAM_TICKS, self.screen_y / CAM_TICKS
        return 0.0, 0.0

    def expand(self, carry: Dict[str, int]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
        """展开本回合的逐 tick 动作，并返回更新后的后台 latch 预算。

        Parameters
        ----------
        carry : Dict[str, int]
            进入本回合时仍延续的 latch 键 → 剩余 tick 预算（来自之前回合）。

        Returns
        -------
        frames : List[Dict[str, object]]
            本回合前台推进产生的逐 tick 动作（长度 == foreground_length()；后台-only 回合为空）。
        new_carry : Dict[str, int]
            本回合后仍延续的 latch 键 → 剩余预算（已应用 release、刷新新预算、扣减本回合前台长度）。
        """
        carry = dict(carry)
        for key in self.release:
            carry.pop(key, None)
        for key, ticks in self.latched_holds().items():
            carry[key] = ticks

        foreground = self.foreground_length()
        within = self.within_turn_holds()
        frames: List[Dict[str, object]] = []
        if foreground > 0:
            per_yaw, per_pitch = self._camera_per_tick()
            for tick in range(foreground):
                action = build_noop_action()
                for key, remaining in carry.items():
                    if tick < min(remaining, foreground):
                        action[key] = True
                for key, ticks in within.items():
                    if tick < ticks:
                        action[key] = True
                if self.clicks and tick < TAP_TICKS:
                    for key in self.clicks:
                        action[key] = True
                if self.camera_mode != CAMERA_NONE and tick < CAM_TICKS:
                    action["camera_yaw"] = per_yaw
                    action["camera_pitch"] = per_pitch
                frames.append(action)
            for key in list(carry):
                carry[key] -= foreground
                if carry[key] <= 0:
                    del carry[key]
        return frames, carry


@dataclass
class MinecraftCommand(MacroCommand):
    """一条原始 Minecraft 命令（如 setblock / give / time set day）。

    不推进游戏 tick、不产生动作帧——归到"下一条推进 tick 的回合之前的挂起命令"，由录制器
    在合适通道发送（fast_reset 的 extra_commands 或 env.add_command）。多条连续 mc 命令
    会累积到同一个挂起列表。
    """

    command: str = ""

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("MinecraftCommand: command 不能为空")

    def minecraft_commands(self) -> List[str]:
        return [self.command]


# ── 构造器（界面/脚本用的简短入口，带默认 kind 标签）────────────────────────────
def turn(
    holds: Optional[Dict[str, int]] = None,
    clicks: Optional[Sequence[str]] = None,
    release: Optional[Sequence[str]] = None,
    camera_mode: str = CAMERA_NONE,
    delta_yaw: float = 0.0,
    delta_pitch: float = 0.0,
    screen_x: float = 0.0,
    screen_y: float = 0.0,
    gui_cursor: bool = False,
    wait_ticks: int = 0,
    label: str = "",
) -> Turn:
    """构造一个回合宏（字段说明见 Turn）。"""
    return Turn(
        kind="turn", label=label,
        holds=dict(holds or {}), clicks=list(clicks or []), release=list(release or []),
        camera_mode=camera_mode, delta_yaw=delta_yaw, delta_pitch=delta_pitch,
        screen_x=screen_x, screen_y=screen_y, gui_cursor=gui_cursor, wait_ticks=wait_ticks,
    )


def minecraft_command(command: str, label: str = "") -> MinecraftCommand:
    return MinecraftCommand(kind="mc", label=label, command=command)


@dataclass
class CompiledTrajectory:
    """编译产物：逐 tick 动作序列 + 观察点索引 + 每 tick 挂起命令 + 每 tick 溯源。

    Attributes
    ----------
    tick_actions : List[Dict[str, object]]
        逐 tick 的完整 V2 动作 dict 列表（长度 = 总 tick 数）。
    observation_tick_indices : List[int]
        观察点落在哪些 tick 边界（升序、去重）。索引 i 表示"第 i 个 tick 执行前是一个观察点"，
        i 取值 [0, len(tick_actions)]；索引 0 是轨迹起始观察点，末尾索引是结束观察点。
    tick_commands : Dict[int, List[str]]
        tick_commands[i] = 在第 i 个 tick 执行"之前"要发送的原始 mc 命令列表。
    tick_sources : List[int]
        tick_sources[i] = 第 i 个 tick 来自第几条宏命令（-1 表示尾段自动延续的 latch 预算）。
    pending_tail_commands : List[str]
        排在所有 tick 之后（末尾）的挂起 mc 命令。
    """

    tick_actions: List[Dict[str, object]] = field(default_factory=list)
    observation_tick_indices: List[int] = field(default_factory=list)
    tick_commands: Dict[int, List[str]] = field(default_factory=dict)
    tick_sources: List[int] = field(default_factory=list)
    pending_tail_commands: List[str] = field(default_factory=list)

    @property
    def total_ticks(self) -> int:
        return len(self.tick_actions)

    @property
    def num_observation_points(self) -> int:
        return len(self.observation_tick_indices)

    def segment_bounds(self) -> List[tuple]:
        """返回相邻观察点之间的 (start_tick, end_tick) 段列表（模型一次推理的动作段）。"""
        idx = self.observation_tick_indices
        return [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]


# 尾段（消费剩余 latch 预算）的观察点切分粒度：贴合"离散动作 2 tick 一次推理"的基础节律。
_TRAILING_SEGMENT_TICKS = TAP_TICKS


def _emit_carry_frames(
    carry: Dict[str, int], length: int,
) -> List[Dict[str, object]]:
    """产出 length 帧、后台 latch 键在其剩余预算内为真的动作帧（供尾段消费）。"""
    frames: List[Dict[str, object]] = []
    for tick in range(length):
        action = build_noop_action()
        for key, remaining in carry.items():
            if tick < remaining:
                action[key] = True
        frames.append(action)
    return frames


def compile_macros(
    commands: Sequence[MacroCommand],
    max_blind_ticks: int = DEFAULT_MAX_BLIND_TICKS,
    manual_observation_ticks: Optional[Sequence[int]] = None,
) -> CompiledTrajectory:
    """把回合宏序列编译成逐 tick 动作 + 观察点。

    latch 语义：HOLD_KEYS 的时长作为后台倒计时预算跨回合延续；回合前台（click/相机/
    非 latch 长按/wait）推进时间轴并扣减预算；所有回合展开后仍有剩余预算的 latch 键以
    尾段实现（按基础节律 _TRAILING_SEGMENT_TICKS 切分观察点）。

    观察点落点规则：
      1. 轨迹起点（tick 0）与终点（总 tick 数）恒为观察点。
      2. 每个推进 tick 的回合的**结束边界**是一个观察点。
      3. 尾段每 _TRAILING_SEGMENT_TICKS 一个观察点（长按持续时的基础推理节律）。
      4. 任意相邻观察点之间若超过 max_blind_ticks，按该上限等距补插。
      5. manual_observation_ticks 强制成为观察点（界面手插），与上述合并。

    Parameters
    ----------
    commands : Sequence[MacroCommand]
        回合宏（Turn）与原始命令宏（MinecraftCommand）的序列。
    max_blind_ticks : int
        单个盲执行段的最大 tick 数（超过则自动补插观察点），须 >= 1。
    manual_observation_ticks : Optional[Sequence[int]]
        额外手动插入的观察点 tick 索引（可选）。

    Returns
    -------
    CompiledTrajectory

    Raises
    ------
    ValueError
        max_blind_ticks < 1，或手动观察点索引越界。
    """
    if max_blind_ticks < 1:
        raise ValueError(f"max_blind_ticks 必须 >= 1，收到 {max_blind_ticks}")

    tick_actions: List[Dict[str, object]] = []
    tick_sources: List[int] = []
    tick_commands: Dict[int, List[str]] = {}
    command_boundary_ticks: set = {0}     # 命令/尾段边界观察点（含起点 0）
    pending_commands: List[str] = []       # 尚未附着到某个 tick 的挂起 mc 命令
    carry: Dict[str, int] = {}             # 后台 latch 键 → 剩余 tick 预算

    for command_index, command in enumerate(commands):
        mc_cmds = command.minecraft_commands()
        if mc_cmds:
            pending_commands.extend(mc_cmds)
            continue

        # 仅 Turn 会推进/延续时间轴。
        frames, carry = command.expand(carry)
        if not frames:
            # 后台-only 回合（仅设定/松开 latch，无前台推进）：不产生帧，预算留待后续消费。
            continue

        if pending_commands:
            attach_index = len(tick_actions)
            tick_commands.setdefault(attach_index, []).extend(pending_commands)
            pending_commands = []

        for frame in frames:
            tick_actions.append(frame)
            tick_sources.append(command_index)
        command_boundary_ticks.add(len(tick_actions))

    # 尾段：消费所有回合展开后仍剩余的 latch 预算，按基础节律切分观察点。
    while carry:
        remaining_max = max(carry.values())
        segment_length = min(_TRAILING_SEGMENT_TICKS, remaining_max)
        frames = _emit_carry_frames(carry, segment_length)
        for frame in frames:
            tick_actions.append(frame)
            tick_sources.append(-1)
        for key in list(carry):
            carry[key] -= segment_length
            if carry[key] <= 0:
                del carry[key]
        command_boundary_ticks.add(len(tick_actions))

    total_ticks = len(tick_actions)
    pending_tail_commands = pending_commands

    if manual_observation_ticks:
        for manual in manual_observation_ticks:
            if manual < 0 or manual > total_ticks:
                raise ValueError(f"手动观察点 tick {manual} 越界 [0, {total_ticks}]")
            command_boundary_ticks.add(manual)

    command_boundary_ticks.add(total_ticks)

    # 按 max_blind_ticks 在过长盲段中补插观察点。
    sorted_boundaries = sorted(command_boundary_ticks)
    observation_indices: List[int] = []
    for i in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[i]
        end = sorted_boundaries[i + 1]
        observation_indices.append(start)
        gap = end - start
        if gap > max_blind_ticks:
            num_segments = (gap + max_blind_ticks - 1) // max_blind_ticks
            step = gap / num_segments
            for seg in range(1, num_segments):
                observation_indices.append(start + round(seg * step))
    observation_indices.append(sorted_boundaries[-1])

    observation_tick_indices = sorted(set(observation_indices))

    return CompiledTrajectory(
        tick_actions=tick_actions,
        observation_tick_indices=observation_tick_indices,
        tick_commands=tick_commands,
        tick_sources=tick_sources,
        pending_tail_commands=pending_tail_commands,
    )
