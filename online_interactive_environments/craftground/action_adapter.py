"""标准键鼠动作到 CraftGround V2 动作字典的适配器。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from online_interactive_environments import ActionTick

DEGREES_PER_MOUSE_UNIT = 0.15
HOTBAR_SLOT_COUNT = 9
_KEY_FIELDS = {
    "W": "forward",
    "S": "back",
    "A": "left",
    "D": "right",
    "Space": "jump",
    "Shift": "sneak",
    "Ctrl": "sprint",
    "MouseLeft": "attack",
    "MouseRight": "use",
    "Q": "drop",
    "E": "inventory",
    **{str(slot): f"hotbar.{slot}" for slot in range(1, 10)},
}


def scroll_hotbar_slot(selected_hotbar: int, scroll: int) -> int:
    if not 1 <= selected_hotbar <= HOTBAR_SLOT_COUNT:
        raise ValueError("selected_hotbar 必须位于 1 到 9")
    return (selected_hotbar - int(scroll) - 1) % HOTBAR_SLOT_COUNT + 1


def _default_action_factory() -> dict[str, bool | float]:
    from craftground.environment.action_space import no_op_v2

    return no_op_v2()


@dataclass
class CraftGroundKeyboardMouseAdapter:
    """维护快捷栏状态并逐 tick 转换标准键鼠输入。"""

    selected_hotbar: int = 1
    action_factory: Callable[[], dict[str, bool | float]] = _default_action_factory

    def reset(self, selected_hotbar: int = 1) -> None:
        if not 1 <= selected_hotbar <= HOTBAR_SLOT_COUNT:
            raise ValueError("selected_hotbar 必须位于 1 到 9")
        self.selected_hotbar = selected_hotbar

    def convert(self, tick: ActionTick) -> dict[str, bool | float]:
        action = self.action_factory()
        tokens = tick.inputs
        mouse_x = 0
        mouse_y = 0
        scroll = 0
        keys: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "MouseMove":
                mouse_x = int(tokens[index + 1])
                mouse_y = int(tokens[index + 2])
                index += 3
            elif token == "Scroll":
                scroll = int(tokens[index + 1])
                index += 2
            else:
                keys.append(token)
                index += 1

        hotbar_keys = [key for key in keys if key.isdigit() and key != "0"]
        if len(hotbar_keys) > 1:
            raise ValueError("同一 tick 只能选择一个快捷栏槽位")
        if scroll and hotbar_keys:
            raise ValueError("同一 tick 不能同时使用 Scroll 和快捷栏数字键")
        overrides: dict[str, bool | float] = {
            field: True for key in keys if (field := _KEY_FIELDS.get(key)) is not None
        }
        if scroll:
            self.selected_hotbar = scroll_hotbar_slot(self.selected_hotbar, scroll)
            overrides[f"hotbar.{self.selected_hotbar}"] = True
        elif hotbar_keys:
            self.selected_hotbar = int(hotbar_keys[0])
        overrides["camera_yaw"] = mouse_x * DEGREES_PER_MOUSE_UNIT
        overrides["camera_pitch"] = mouse_y * DEGREES_PER_MOUSE_UNIT
        unknown = set(overrides).difference(action)
        if unknown:
            raise ValueError(f"CraftGround V2 动作字典缺少字段：{sorted(unknown)}")
        action.update(overrides)
        return action
