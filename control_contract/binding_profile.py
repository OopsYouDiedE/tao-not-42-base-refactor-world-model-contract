# -*- coding: utf-8 -*-
"""每游戏 / 每设备一份的绑定配置：控制契约中唯一允许携带具体绑定知识的地方。

对外接口：
    AxisSpec — 一个抽象摇杆的能力（方向 / 力度量化档数 + 单 tick 推进上限）。
    AbilityBinding — 游戏专有能力名 → 核心角色的别名声明。
    BindingProfile — 两个摇杆 + 光标上限 + 能力别名 + 槽位数的完整声明。
    load_binding_profile / parse_binding_profile — 从 JSON 文件 / dict 构造。
    describe_capabilities — 生成给大模型的能力说明（只列本 profile 真正支持的原语）。
    SCREEN_DIAGONAL — 光标上限达到它即为"单 tick 跳到任意位置"。

设计要点：大模型永远不读这个文件，它只按 describe_capabilities 的自然语言说明写语义指令。

**键鼠与手柄不是两个设备族**：两者都是"位移摇杆 + 瞄准摇杆 + 按钮"，差别只是
``AxisSpec`` 里的量化档数与单 tick 上限取值不同——键盘位移是 8 向单档，鼠标瞄准的光标上限
大到可单 tick 直达（即"跳转位置"）。因此编译器只有一条代码路径，换游戏、换设备只改数据。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import math

from control_contract.role_vocabulary import CORE_ROLES, PrimitiveName, is_core_role

# 归一化屏幕对角线长度：光标单 tick 上限达到它就意味着"任意位置一 tick 直达"（鼠标跳转）。
SCREEN_DIAGONAL = math.sqrt(2.0)


@dataclass(frozen=True)
class AxisSpec:
    """一个抽象摇杆的能力声明。位移摇杆与瞄准摇杆共用这一个结构。

    键鼠与手柄的差别全部落在这几个数值上，不需要按设备族分支：

    ==================  ====================  =========================
    字段                 手柄                   键鼠
    ==================  ====================  =========================
    位移 direction_count 0（连续）              8（WASD）
    位移 magnitude_levels 0（连续）              1（开关）
    瞄准 cap_per_tick    7.3 度（220 度/秒）    18 度（鼠标一帧能甩多少）
    ==================  ====================  =========================

    Attributes
    ----------
    direction_count : int
        方向量化档数；0 表示连续任意方向。键盘 WASD = 8。
    magnitude_levels : int
        力度量化档数；0 表示连续，1 表示只有"满力度"一档（开关式按键）。
    dead_zone : float
        力度低于此值视为摇杆归中（0..1）。
    cap_per_tick : float
        单 tick 最多能推进多少"被控量"。瞄准摇杆的被控量是度；光标语境下是归一化屏幕
        单位。上限越大越接近"跳转"，达到 SCREEN_DIAGONAL 即为单 tick 任意直达。
    """

    direction_count: int = 0
    magnitude_levels: int = 0
    dead_zone: float = 0.12
    cap_per_tick: float = 1.0

    def __post_init__(self) -> None:
        if self.direction_count < 0:
            raise ValueError(f"direction_count 不能为负，收到 {self.direction_count}")
        if self.magnitude_levels < 0:
            raise ValueError(f"magnitude_levels 不能为负，收到 {self.magnitude_levels}")
        if not 0.0 <= self.dead_zone < 1.0:
            raise ValueError(f"dead_zone 须在 [0, 1)，收到 {self.dead_zone}")
        if self.cap_per_tick <= 0:
            raise ValueError(f"cap_per_tick 必须为正，收到 {self.cap_per_tick}")

    @property
    def is_continuous_direction(self) -> bool:
        """方向是否连续（手柄摇杆 True，键盘 WASD False）。"""
        return self.direction_count == 0

    @property
    def is_continuous_magnitude(self) -> bool:
        """力度是否连续（手柄 True，键盘开关键 False）。"""
        return self.magnitude_levels == 0

    def quantise(self, direction_deg: float, magnitude: float) -> tuple[float, float]:
        """按本摇杆的量化能力把 (方向, 力度) 规整到可表达的最近取值。

        Parameters
        ----------
        direction_deg : float
            请求方向（度，任意实数）。
        magnitude : float
            请求力度（0..1，超界自动截断）。

        Returns
        -------
        tuple[float, float]
            ``(量化后方向, 量化后力度)``；力度低于死区时返回 ``(0.0, 0.0)``。
        """
        clamped = max(0.0, min(1.0, magnitude))
        if clamped < self.dead_zone:
            return 0.0, 0.0
        direction = direction_deg % 360.0
        if not self.is_continuous_direction:
            grid = 360.0 / self.direction_count
            direction = (round(direction / grid) % self.direction_count) * grid
        if not self.is_continuous_magnitude:
            level = math.ceil(clamped * self.magnitude_levels)
            clamped = min(1.0, level / self.magnitude_levels)
        return direction, clamped


@dataclass(frozen=True)
class AbilityBinding:
    """游戏专有能力名到核心角色的别名。

    Attributes
    ----------
    name : str
        大模型可直接书写的能力名（如 ``grapple``），小写、无空格。
    role : str
        该能力实际占用的核心角色（须 ∈ CORE_ROLES），编译器只认角色。
    description : str
        给大模型的一句话说明，进入 prompt 的能力清单。
    """

    name: str
    role: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AbilityBinding.name 不能为空")
        if not is_core_role(self.role):
            raise ValueError(f"AbilityBinding.role 必须属于 CORE_ROLES，收到 {self.role!r}")


@dataclass(frozen=True)
class BindingProfile:
    """一个游戏 × 一套输入设备的控制能力声明。

    Attributes
    ----------
    profile_name : str
        标识（如 ``minecraft_mouse_keyboard`` / ``generic_gamepad``）。
    tick_hz : float
        目标环境的控制步频（Hz）。编译器用它把毫秒换成 tick，大模型永远看不到 tick。
    locomotion_axis : AxisSpec
        位移摇杆。手柄为连续方向 + 连续力度；键盘 WASD 为 ``direction_count=8,
        magnitude_levels=1``。其 ``cap_per_tick`` 无实义（偏转即刻生效），保持默认 1。
    aim_axis : AxisSpec
        瞄准摇杆，被控量为**度**。``cap_per_tick`` 即单 tick 最大转角：手柄 = 角速度 ÷ 步频，
        鼠标 = 单帧最大甩动角度。方向与力度在此摇杆上一律连续。
    cursor_cap_per_tick : float
        同一个瞄准摇杆在**界面光标**语境下的单 tick 上限（归一化屏幕单位）。手柄约 0.04；
        CraftGround 的鼠标受相机上限约束约 0.2；真实操作系统鼠标 ≥ SCREEN_DIAGONAL，
        即单 tick 跳到任意位置。这是"连续控制"与"跳转位置"的唯一区别。
    sprint_threshold : float
        位移摇杆无法表达高力度时（``magnitude_levels==1``），力度高于此值自动附加 sprint
        角色作为近似（0..1；>1 表示从不自动）。
    menu_cursor : bool
        界面里是否存在光标。False 表示只能用 nav_* 移动焦点 + confirm（主机原生 UI），
        此时 point 原语不可用。这是**游戏**属性，与用什么设备无关。
    direct_slot_buttons : bool
        是否有独立按键可直达任意槽位（键盘数字键 True，手柄肩键 False → 展开为 next/prev
        循环）。
    slot_count : int
        可选槽位数（0 表示本游戏无槽位概念）。
    supports_text : bool
        是否支持文本输入原语。
    abilities : tuple[AbilityBinding, ...]
        游戏专有能力别名。
    unavailable_roles : frozenset[str]
        本游戏 / 设备上确实不存在的核心角色，编译器遇到即报错而非静默丢弃。
    """

    profile_name: str
    tick_hz: float = 20.0
    locomotion_axis: AxisSpec = AxisSpec(direction_count=8, magnitude_levels=1, dead_zone=0.15)
    aim_axis: AxisSpec = AxisSpec(cap_per_tick=18.0)
    cursor_cap_per_tick: float = SCREEN_DIAGONAL
    sprint_threshold: float = 2.0
    menu_cursor: bool = True
    direct_slot_buttons: bool = True
    slot_count: int = 0
    supports_text: bool = False
    abilities: tuple[AbilityBinding, ...] = ()
    unavailable_roles: frozenset = frozenset()

    def __post_init__(self) -> None:
        if self.tick_hz <= 0:
            raise ValueError(f"tick_hz 必须为正，收到 {self.tick_hz}")
        if self.cursor_cap_per_tick <= 0:
            raise ValueError(f"cursor_cap_per_tick 必须为正，收到 {self.cursor_cap_per_tick}")
        if self.slot_count > 0 and not self.direct_slot_buttons:
            missing = {"next", "prev"} & set(self.unavailable_roles)
            if missing:
                raise ValueError(
                    f"无直达槽位按键的 profile 必须保留 next/prev 角色，但 {sorted(missing)} "
                    "被列入 unavailable_roles")
        if self.slot_count < 0:
            raise ValueError("slot_count 不能为负")
        for role in self.unavailable_roles:
            if not is_core_role(role):
                raise ValueError(f"unavailable_roles 含未知角色 {role!r}")
        seen: set = set()
        for ability in self.abilities:
            if ability.name in seen:
                raise ValueError(f"能力别名重复：{ability.name!r}")
            seen.add(ability.name)

    def seconds_per_tick(self) -> float:
        """单 tick 的秒数。"""
        return 1.0 / self.tick_hz

    @property
    def cursor_jumps_in_one_tick(self) -> bool:
        """光标能否单 tick 跳到任意位置（真实鼠标 True，手柄与受相机约束的鼠标 False）。"""
        return self.cursor_cap_per_tick >= SCREEN_DIAGONAL

    def resolve_role(self, name: str) -> str:
        """把大模型写的角色名或能力别名解析为核心角色。

        Parameters
        ----------
        name : str
            核心角色名或本 profile 声明的能力别名（大小写不敏感）。

        Returns
        -------
        str
            核心角色名。

        Raises
        ------
        ValueError
            名字既非核心角色也非本 profile 能力别名，或该角色在本 profile 上不存在。
        """
        lowered = name.strip().lower()
        resolved = lowered
        if not is_core_role(lowered):
            for ability in self.abilities:
                if ability.name.lower() == lowered:
                    resolved = ability.role
                    break
            else:
                raise ValueError(
                    f"未知角色 / 能力 {name!r}；可用角色 {CORE_ROLES}，"
                    f"可用能力 {[item.name for item in self.abilities]}"
                )
        if resolved in self.unavailable_roles:
            raise ValueError(f"角色 {resolved!r} 在 profile {self.profile_name!r} 上不存在")
        return resolved

    def available_primitives(self) -> tuple[PrimitiveName, ...]:
        """本 profile 真正支持的原语通道（无界面光标则无 point，无键盘则无 text）。"""
        names = [
            PrimitiveName.AIM, PrimitiveName.MOVE, PrimitiveName.PRESS,
            PrimitiveName.HOLD, PrimitiveName.RELEASE,
        ]
        if self.menu_cursor:
            names.append(PrimitiveName.POINT)
        if self.slot_count > 0:
            names.append(PrimitiveName.SELECT)
        if self.supports_text:
            names.append(PrimitiveName.TEXT)
        return tuple(names)

    def usable_roles(self) -> tuple[str, ...]:
        """本 profile 上可用的核心角色（已剔除 unavailable_roles）。"""
        return tuple(role for role in CORE_ROLES if role not in self.unavailable_roles)


def _resolve_cap(payload: Mapping[str, Any], default: float, tick_hz: float) -> float:
    """取单 tick 上限：JSON 可写 ``cap_per_tick``，也可写更好标定的 ``cap_per_second``。

    手柄厂商参数天然是"度/秒"，因此允许按秒声明再由步频换算，避免手工除法写错。
    """
    if "cap_per_second" in payload:
        return float(payload["cap_per_second"]) / tick_hz
    return float(payload.get("cap_per_tick", default))


def _parse_axis(
    payload: Any, default: AxisSpec, tick_hz: float,
) -> AxisSpec:
    """从 dict 构造 AxisSpec，缺字段用 default 补齐。"""
    if payload is None:
        return default
    if not isinstance(payload, Mapping):
        raise ValueError(f"摇杆声明必须是对象，收到 {type(payload).__name__}")
    return AxisSpec(
        direction_count=int(payload.get("direction_count", default.direction_count)),
        magnitude_levels=int(payload.get("magnitude_levels", default.magnitude_levels)),
        dead_zone=float(payload.get("dead_zone", default.dead_zone)),
        cap_per_tick=_resolve_cap(payload, default.cap_per_tick, tick_hz),
    )


def parse_binding_profile(payload: Mapping[str, Any]) -> BindingProfile:
    """从 dict 构造 BindingProfile。

    Parameters
    ----------
    payload : Mapping[str, Any]
        字段名与 BindingProfile 一致；``abilities`` 为对象数组，枚举字段用字符串值。

    Returns
    -------
    BindingProfile

    Raises
    ------
    ValueError
        缺少 ``profile_name``、枚举取值非法或字段互相矛盾。
    """
    if "profile_name" not in payload:
        raise ValueError("binding profile 缺少 profile_name")
    abilities = tuple(
        AbilityBinding(
            name=str(item["name"]).strip().lower(),
            role=str(item["role"]).strip().lower(),
            description=str(item.get("description", "")),
        )
        for item in payload.get("abilities", ())
    )
    defaults = BindingProfile(profile_name=str(payload["profile_name"]))
    tick_hz = float(payload.get("tick_hz", defaults.tick_hz))
    return BindingProfile(
        profile_name=str(payload["profile_name"]),
        tick_hz=tick_hz,
        locomotion_axis=_parse_axis(
            payload.get("locomotion_axis"), defaults.locomotion_axis, tick_hz),
        aim_axis=_parse_axis(payload.get("aim_axis"), defaults.aim_axis, tick_hz),
        cursor_cap_per_tick=(
            float(payload["cursor_cap_per_second"]) / tick_hz
            if "cursor_cap_per_second" in payload
            else float(payload.get("cursor_cap_per_tick", defaults.cursor_cap_per_tick))),
        sprint_threshold=float(payload.get("sprint_threshold", defaults.sprint_threshold)),
        menu_cursor=bool(payload.get("menu_cursor", defaults.menu_cursor)),
        direct_slot_buttons=bool(
            payload.get("direct_slot_buttons", defaults.direct_slot_buttons)),
        slot_count=int(payload.get("slot_count", defaults.slot_count)),
        supports_text=bool(payload.get("supports_text", defaults.supports_text)),
        abilities=abilities,
        unavailable_roles=frozenset(
            str(role).strip().lower() for role in payload.get("unavailable_roles", ())),
    )


def load_binding_profile(path: Path) -> BindingProfile:
    """从 JSON 文件读取 BindingProfile。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_binding_profile(json.load(handle))


def _describe_locomotion(profile: BindingProfile) -> str:
    """描述位移摇杆的可表达能力（由量化档数推导，不按设备族查表）。"""
    axis = profile.locomotion_axis
    if axis.is_continuous_direction:
        direction = "move.dir accepts any direction"
    else:
        direction = (
            f"move.dir is snapped to the nearest of {axis.direction_count} directions "
            f"({360.0 / axis.direction_count:.0f} degrees apart)"
        )
    if axis.is_continuous_magnitude:
        power = "move.power is a true magnitude: 0.3 walks slowly, 1.0 runs at full speed"
    elif axis.magnitude_levels == 1:
        power = (
            "move.power only decides whether you move at all, not how fast; "
            "use the sprint role explicitly to go faster"
        )
    else:
        power = f"move.power is rounded up to one of {axis.magnitude_levels} speed levels"
    return f"{direction}; {power}."


def _describe_aim(profile: BindingProfile) -> str:
    """描述瞄准摇杆的单 tick 转角预算，让大模型自己算得出步长够不够。"""
    per_second = profile.aim_axis.cap_per_tick * profile.tick_hz
    return (
        f"aim turns at most {per_second:.0f} degrees per second, so a large turn needs a "
        f"proportionally longer step or it will be truncated."
    )


def _describe_cursor(profile: BindingProfile) -> str:
    """描述界面光标能力：跳转 / 限速逼近 / 无光标，三者由数值判定而非枚举。"""
    if not profile.menu_cursor:
        return (
            "there is no cursor in menus: point is unavailable, move the UI focus with the "
            "nav_up / nav_down / nav_left / nav_right roles and press confirm."
        )
    if profile.cursor_jumps_in_one_tick:
        return "point puts the cursor straight onto the requested screen position."
    per_second = profile.cursor_cap_per_tick * profile.tick_hz
    return (
        f"point steers the cursor towards the requested position, crossing at most "
        f"{per_second:.2f} of the screen per second, so long cursor travel needs a longer "
        f"step and may not finish in one step."
    )


def _describe_select(profile: BindingProfile) -> str:
    """描述槽位选择代价。"""
    if profile.direct_slot_buttons:
        return "select jumps directly to any slot."
    return (
        "select has no direct buttons: it is expanded into repeated next/prev cycling, so "
        "switching to a distant slot costs more time."
    )


def describe_capabilities(profile: BindingProfile) -> str:
    """生成给大模型的 profile 能力说明（进入 system prompt 的设备段）。

    只描述本 profile 真正支持的原语与其代价，不泄露任何物理键位——这样同一个模型端
    prompt 构造代码可以服务任意游戏与任意设备。

    Parameters
    ----------
    profile : BindingProfile

    Returns
    -------
    str
        多行英文说明。
    """
    primitives = ", ".join(item.value for item in profile.available_primitives())
    lines = [
        f"Control profile: {profile.profile_name}.",
        f"Available primitives: {primitives}.",
        f"Available roles: {', '.join(profile.usable_roles())}.",
        f"Device notes: {_describe_locomotion(profile)}",
        f"  {_describe_aim(profile)}",
        f"  {_describe_cursor(profile)}",
    ]
    if profile.slot_count > 0:
        lines.append(f"  Slots 1..{profile.slot_count} exist and {_describe_select(profile)}")
    if profile.abilities:
        lines.append("Game abilities you may name directly instead of a raw role:")
        for ability in profile.abilities:
            suffix = f" — {ability.description}" if ability.description else ""
            lines.append(f"  {ability.name} (acts as {ability.role}){suffix}")
    if profile.unavailable_roles:
        lines.append(
            "Unavailable on this device / game: "
            f"{', '.join(sorted(profile.unavailable_roles))}."
        )
    return "\n".join(lines)
