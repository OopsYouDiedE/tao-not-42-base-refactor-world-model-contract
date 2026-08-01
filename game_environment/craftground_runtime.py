"""当前 CraftGround 闭环管线共用的运行时适配。"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from tao.protocols.action.codec import DEGREES_PER_PIXEL, MINECRAFT_KEYMAP

SCENE_COMMANDS = (
    "gamemode survival @s",
    "gamerule doDaylightCycle false",
    "gamerule doWeatherCycle false",
    "gamerule doMobSpawning false",
    "time set 6000",
    "weather clear",
    "fill 0 62 0 8 69 10 minecraft:air",
    "fill 0 62 0 8 62 10 minecraft:smooth_stone",
    "fill 0 63 0 8 63 10 minecraft:oak_planks",
    "fill 0 64 0 8 67 0 minecraft:stone_bricks",
    "fill 0 64 10 8 67 10 minecraft:stone_bricks",
    "fill 0 64 0 0 67 10 minecraft:stone_bricks",
    "fill 8 64 0 8 67 10 minecraft:stone_bricks",
    "setblock 3 64 3 minecraft:chest[facing=south]",
    "item replace block 3 64 3 container.0 with minecraft:iron_ingot 3",
    "setblock 4 64 3 minecraft:crafting_table",
    "setblock 5 64 3 minecraft:furnace[facing=south]",
    "item replace block 5 64 3 container.0 with minecraft:raw_iron 2",
    "item replace block 5 64 3 container.1 with minecraft:coal 2",
    "setblock 2 64 3 minecraft:gold_block",
    "setblock 6 64 3 minecraft:diamond_block",
    "setblock 1 65 1 minecraft:torch",
    "setblock 7 65 1 minecraft:torch",
    "clear @s",
    "give @s minecraft:stick 2",
    "tp @s 4.5 64 8.5 180 12",
)

RESET_PLAYER_COMMANDS = (
    "clear @s",
    "give @s minecraft:stick 2",
    "tp @s 4.5 64 8.5 180 12",
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_INVERSE_KEYMAP = {token: field for field, token in MINECRAFT_KEYMAP.items()}
_HOTBAR_KEYS = {str(slot) for slot in range(1, 10)}
HOTBAR_SLOT_COUNT = 9


def validate_identifier(value: str, field_name: str) -> str:
    """校验会进入文件路径或 CraftGround 命令的稳定标识符。"""
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} 只能包含字母、数字、点、下划线和连字符")
    return value


def build_environment(
    runtime: Path,
    *,
    image_width: int = 640,
    image_height: int = 360,
    port: int = 18300,
) -> Any:
    """创建正式闭环所用的 CraftGround V2 环境。"""
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.screen_encoding_modes import ScreenEncodingMode

    config = InitialEnvironmentConfig(
        image_width=image_width,
        image_height=image_height,
        seed="424242",
        render_distance=3,
        simulation_distance=5,
        screen_encoding_mode=ScreenEncodingMode.RAW,
    )
    return CraftGroundEnvironment(
        config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        env_path=str(runtime),
        port=port,
        find_free_port=True,
        cleanup_world=False,
        verbose=False,
    )


def build_v2_action(
    overrides: dict[str, bool | float],
    *,
    action_factory: Callable[[], dict[str, bool | float]] | None = None,
) -> dict[str, bool | float]:
    """基于 CraftGround 的完整 no-op 字典构造单 tick 动作。"""
    if action_factory is None:
        from craftground.environment.action_space import no_op_v2

        action_factory = no_op_v2
    action = action_factory()
    unknown = set(overrides).difference(action)
    if unknown:
        raise ValueError(f"未知 CraftGround V2 动作字段: {sorted(unknown)}")
    action.update(overrides)
    return action


def scroll_hotbar_slot(selected_hotbar: int, scroll: int) -> int:
    """把滚轮相对位移转换成 1–9 范围内的快捷栏槽位。"""
    if not 1 <= selected_hotbar <= HOTBAR_SLOT_COUNT:
        raise ValueError("selected_hotbar 必须位于 1 到 9")
    return (selected_hotbar - int(scroll) - 1) % HOTBAR_SLOT_COUNT + 1


def action_tick_to_v2_action(
    keys: tuple[str, ...],
    mouse: tuple[int, int],
    scroll: int = 0,
    *,
    selected_hotbar: int | None = None,
    action_factory: Callable[[], dict[str, bool | float]] | None = None,
) -> dict[str, bool | float]:
    """把一个 TAP tick 转为完整 CraftGround V2 动作。"""
    if scroll:
        if selected_hotbar is None:
            raise ValueError("转换 Scroll 动作时必须提供当前快捷栏槽位")
        if any(key in _HOTBAR_KEYS for key in keys):
            raise ValueError("同一 tick 不能同时使用 Scroll 和快捷栏数字键")
    overrides: dict[str, bool | float] = {
        _INVERSE_KEYMAP[key]: True for key in keys if key in _INVERSE_KEYMAP
    }
    if scroll:
        target_slot = scroll_hotbar_slot(selected_hotbar, scroll)
        overrides[f"hotbar.{target_slot}"] = True
    overrides.update(
        camera_yaw=mouse[0] * DEGREES_PER_PIXEL,
        camera_pitch=mouse[1] * DEGREES_PER_PIXEL,
    )
    return build_v2_action(overrides, action_factory=action_factory)


@dataclass
class CraftGroundActionAdapter:
    """维护快捷栏位置，并把连续 TAP tick 转成 CraftGround V2 动作。"""

    selected_hotbar: int = 1
    action_factory: Callable[[], dict[str, bool | float]] | None = None

    def reset(self, selected_hotbar: int = 1) -> dict[str, bool | float]:
        """重置快捷栏状态，并返回用于同步游戏状态的绝对选择动作。"""
        if not 1 <= selected_hotbar <= HOTBAR_SLOT_COUNT:
            raise ValueError("selected_hotbar 必须位于 1 到 9")
        self.selected_hotbar = selected_hotbar
        return build_v2_action(
            {f"hotbar.{selected_hotbar}": True},
            action_factory=self.action_factory,
        )

    def convert(
        self,
        keys: tuple[str, ...],
        mouse: tuple[int, int],
        scroll: int = 0,
    ) -> dict[str, bool | float]:
        """转换一个 tick，并同步显式数字键或滚轮产生的快捷栏状态。"""
        hotbar_keys = [key for key in keys if key in _HOTBAR_KEYS]
        if len(hotbar_keys) > 1:
            raise ValueError("同一 tick 只能选择一个快捷栏槽位")
        action = action_tick_to_v2_action(
            keys,
            mouse,
            scroll,
            selected_hotbar=self.selected_hotbar,
            action_factory=self.action_factory,
        )
        if scroll:
            self.selected_hotbar = scroll_hotbar_slot(self.selected_hotbar, scroll)
        elif hotbar_keys:
            self.selected_hotbar = int(hotbar_keys[0])
        return action


def step_commands(environment: Any, commands: Iterable[str], ticks: int = 5) -> Any:
    """提交命令后执行指定数量的同步 no-op tick，并返回末帧观测。"""
    if ticks < 1:
        raise ValueError("ticks 必须大于零")
    from craftground.environment.action_space import no_op_v2

    command_list = list(commands)
    if command_list:
        environment.add_commands(command_list)
    observation = None
    for _ in range(ticks):
        observation = environment.step(no_op_v2())[0]
    return observation


def save_rgb(observation: Any, path: Path) -> None:
    """把 CraftGround 观测中的 RGB 数组写入图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observation["rgb"]).save(path)
