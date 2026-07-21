"""结构化 Minecraft 动作与视觉大模型文本 token 之间的可逆编解码。

对外接口：
    ActionTokenFormat — 候选文本格式枚举（供动作 token 表示实验对比）。
    StructuredAction — 单帧结构化动作（相机 bin + 20 个 V2 键，构造上互斥）。
    encode_actions / decode_actions — 结构化动作序列 ↔ 文本。
    describe_format — 返回某格式给大模型的自然语言说明与样例。
    ACTION_KEY_GROUPS — 互斥组定义，解码端据此强制结构有界（AGENTS §5）。

设计要点：解码端永远只产生结构合法的动作——前后 / 左右 / 姿态 / hotbar 各组至多一个
被激活，非法组合按组内优先级或直接置空消解，绝不把冲突键同时置真。相机沿用
``rl_training_environments/craftground/action_contract`` 的 mu-law 分箱，单位与部署端一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from rl_training_environments.craftground.action_contract import CAM_BINS, V2_KEYS

# 相机中性 bin（(CAM_BINS-1)/2），对应零角度增量。
CAMERA_NEUTRAL_BIN = (CAM_BINS - 1) // 2

# 互斥组：组内至多一个键为真。列表按组内优先级排序，解码遇冲突时保留靠前者。
ACTION_KEY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("forward", "back"),
    ("left", "right"),
    ("sneak", "sprint"),
    tuple(f"hotbar.{index}" for index in range(1, 10)),
)
# 不属于任何互斥组的独立二值键。
INDEPENDENT_KEYS: tuple[str, ...] = ("jump", "attack", "use", "drop", "inventory")

_GROUPED_KEYS = frozenset(key for group in ACTION_KEY_GROUPS for key in group)
if _GROUPED_KEYS | set(INDEPENDENT_KEYS) != set(V2_KEYS):
    raise RuntimeError("互斥组与独立键并集必须恰好覆盖全部 V2_KEYS")
if len(_GROUPED_KEYS) + len(INDEPENDENT_KEYS) != len(V2_KEYS):
    raise RuntimeError("互斥组与独立键之间存在重复键")


class ActionTokenFormat(str, Enum):
    """动作 token 的候选文本表示，供实验比较大模型最擅长哪种。

    - ``COMPACT_TAG``：单行紧凑标签串，如 ``F R sprint attack h3 cam=+3,-1``。
    - ``KEY_VALUE``：显式键值行，如 ``move=forward turn=right cam_yaw=3 ...``。
    - ``JSON_LINE``：每帧一行 JSON 对象，字段名与 V2 语义对齐。
    """

    COMPACT_TAG = "compact_tag"
    KEY_VALUE = "key_value"
    JSON_LINE = "json_line"


# 移动方向与转向的可读标签，用于紧凑格式与键值格式。
_MOVE_LABEL = {("forward", "back"): {"forward": "forward", "back": "back"}}
_COMPACT_MOVE = {"forward": "F", "back": "B"}
_COMPACT_TURN = {"left": "L", "right": "R"}


@dataclass
class StructuredAction:
    """单帧结构化动作。

    Attributes
    ----------
    camera_yaw_bin : int
        水平相机 mu-law 分箱，取值 ``[0, CAM_BINS-1]``，中性为 ``CAMERA_NEUTRAL_BIN``。
    camera_pitch_bin : int
        垂直相机 mu-law 分箱，取值同上。
    keys : dict[str, bool]
        20 个 V2 键的布尔状态；构造后满足全部互斥约束。
    """

    camera_yaw_bin: int = CAMERA_NEUTRAL_BIN
    camera_pitch_bin: int = CAMERA_NEUTRAL_BIN
    keys: dict[str, bool] = field(default_factory=lambda: {key: False for key in V2_KEYS})

    def __post_init__(self) -> None:
        self.camera_yaw_bin = _clamp_bin(self.camera_yaw_bin)
        self.camera_pitch_bin = _clamp_bin(self.camera_pitch_bin)
        normalized = {key: bool(self.keys.get(key, False)) for key in V2_KEYS}
        self.keys = _enforce_exclusivity(normalized)

    def active_keys(self) -> list[str]:
        """返回当前为真的键，按 V2_KEYS 顺序。"""
        return [key for key in V2_KEYS if self.keys[key]]


def _clamp_bin(value: int) -> int:
    """把相机 bin 截断到合法范围。"""
    return max(0, min(CAM_BINS - 1, int(value)))


def _enforce_exclusivity(keys: dict[str, bool]) -> dict[str, bool]:
    """按组内优先级消解互斥冲突，保证结构有界（AGENTS §5）。"""
    result = dict(keys)
    for group in ACTION_KEY_GROUPS:
        seen_active = False
        for key in group:
            if result[key] and not seen_active:
                seen_active = True
            else:
                result[key] = False
    return result


def _signed_offset(bin_value: int) -> int:
    """把相机 bin 转成相对中性 bin 的带符号偏移。"""
    return bin_value - CAMERA_NEUTRAL_BIN


def _hotbar_index(action: StructuredAction) -> int | None:
    """返回激活的 hotbar 槽位 ``1..9``，未激活返回 None。"""
    for index in range(1, 10):
        if action.keys[f"hotbar.{index}"]:
            return index
    return None


def _encode_compact(action: StructuredAction) -> str:
    """紧凑标签串，例如 ``F R sprint jump attack h3 cam=+3,-1``。"""
    tokens: list[str] = []
    for forward_key, symbol in _COMPACT_MOVE.items():
        if action.keys[forward_key]:
            tokens.append(symbol)
    for turn_key, symbol in _COMPACT_TURN.items():
        if action.keys[turn_key]:
            tokens.append(symbol)
    for key in ("jump", "sneak", "sprint", "attack", "use", "drop", "inventory"):
        if action.keys[key]:
            tokens.append(key)
    hotbar = _hotbar_index(action)
    if hotbar is not None:
        tokens.append(f"h{hotbar}")
    yaw = _signed_offset(action.camera_yaw_bin)
    pitch = _signed_offset(action.camera_pitch_bin)
    tokens.append(f"cam={yaw:+d},{pitch:+d}")
    return " ".join(tokens) if tokens else "noop"


def _encode_key_value(action: StructuredAction) -> str:
    """显式键值行，例如 ``move=forward turn=right cam_yaw=+3 cam_pitch=-1 buttons=attack,jump hotbar=3``。"""
    move = "forward" if action.keys["forward"] else "back" if action.keys["back"] else "none"
    turn = "left" if action.keys["left"] else "right" if action.keys["right"] else "none"
    stance = "sneak" if action.keys["sneak"] else "sprint" if action.keys["sprint"] else "none"
    buttons = [key for key in ("jump", "attack", "use", "drop", "inventory") if action.keys[key]]
    hotbar = _hotbar_index(action)
    parts = [
        f"move={move}",
        f"turn={turn}",
        f"stance={stance}",
        f"cam_yaw={_signed_offset(action.camera_yaw_bin):+d}",
        f"cam_pitch={_signed_offset(action.camera_pitch_bin):+d}",
        f"buttons={','.join(buttons) if buttons else 'none'}",
        f"hotbar={hotbar if hotbar is not None else 'none'}",
    ]
    return " ".join(parts)


def _encode_json(action: StructuredAction) -> str:
    """每帧一行 JSON。手写以保证字段顺序稳定、便于大模型模仿。"""
    move = "forward" if action.keys["forward"] else "back" if action.keys["back"] else "none"
    turn = "left" if action.keys["left"] else "right" if action.keys["right"] else "none"
    stance = "sneak" if action.keys["sneak"] else "sprint" if action.keys["sprint"] else "none"
    buttons = [key for key in ("jump", "attack", "use", "drop", "inventory") if action.keys[key]]
    hotbar = _hotbar_index(action)
    button_list = ", ".join(f'"{key}"' for key in buttons)
    return (
        f'{{"move": "{move}", "turn": "{turn}", "stance": "{stance}", '
        f'"cam_yaw": {_signed_offset(action.camera_yaw_bin)}, '
        f'"cam_pitch": {_signed_offset(action.camera_pitch_bin)}, '
        f'"buttons": [{button_list}], '
        f'"hotbar": {hotbar if hotbar is not None else "null"}}}'
    )


_ENCODERS = {
    ActionTokenFormat.COMPACT_TAG: _encode_compact,
    ActionTokenFormat.KEY_VALUE: _encode_key_value,
    ActionTokenFormat.JSON_LINE: _encode_json,
}


def encode_single_action(
    action: StructuredAction,
    action_format: ActionTokenFormat,
) -> str:
    """把单帧动作编码为一行文本（不含 ``t<下标>:`` 前缀）。

    交错布局的 prompt 里，历史动作紧跟其所属帧图像之后，故需要不带序号前缀的
    单帧编码。

    Parameters
    ----------
    action : StructuredAction
        待编码的单帧动作。
    action_format : ActionTokenFormat
        目标文本格式。

    Returns
    -------
    str
        该帧动作的单行文本。
    """
    return _ENCODERS[action_format](action)


def encode_actions(
    actions: list[StructuredAction],
    action_format: ActionTokenFormat,
) -> str:
    """把结构化动作序列编码为逐帧一行的文本。

    Parameters
    ----------
    actions : list[StructuredAction]
        待编码的动作序列，一元素一帧。
    action_format : ActionTokenFormat
        目标文本格式。

    Returns
    -------
    str
        每帧一行、以 ``\\n`` 连接的文本；空序列返回空串。
    """
    encoder = _ENCODERS[action_format]
    return "\n".join(f"t{index}: {encoder(action)}" for index, action in enumerate(actions))


_SIGNED_INTEGER = r"[-+]?\d+"


def _parse_camera_offset(text: str) -> int:
    """把带符号偏移文本转为合法相机 bin，越界截断。"""
    try:
        offset = int(text)
    except ValueError:
        return CAMERA_NEUTRAL_BIN
    return _clamp_bin(CAMERA_NEUTRAL_BIN + offset)


def _decode_line(line: str) -> StructuredAction | None:
    """从单行文本鲁棒解码一帧动作；无可识别内容返回 None。

    对三种格式统一用宽松扫描：不依赖精确分隔符，容忍大模型输出漂移。所有被识别
    的键先收集，再交给 ``StructuredAction`` 的构造器统一强制互斥。
    """
    import re

    body = line.split(":", 1)[1] if re.match(r"\s*t\d+\s*:", line) else line
    lowered = body.lower()
    if not lowered.strip():
        return None
    keys = {key: False for key in V2_KEYS}

    move_match = re.search(r'move"?\s*[=:]\s*"?(forward|back|none)', lowered)
    turn_match = re.search(r'turn"?\s*[=:]\s*"?(left|right|none)', lowered)
    stance_match = re.search(r'stance"?\s*[=:]\s*"?(sneak|sprint|none)', lowered)
    explicit = any((move_match, turn_match, stance_match))
    if explicit:
        for match, mapping in (
            (move_match, {"forward": "forward", "back": "back"}),
            (turn_match, {"left": "left", "right": "right"}),
            (stance_match, {"sneak": "sneak", "sprint": "sprint"}),
        ):
            if match and match.group(1) in mapping:
                keys[mapping[match.group(1)]] = True
        for key in ("jump", "attack", "use", "drop", "inventory"):
            if re.search(rf"\b{key}\b", lowered):
                keys[key] = True
        hotbar_match = re.search(r'hotbar"?\s*[=:]\s*"?([1-9])', lowered)
        if hotbar_match:
            keys[f"hotbar.{hotbar_match.group(1)}"] = True
    recognized = explicit
    if not explicit:
        if re.search(r"\bnoop\b", lowered):
            recognized = True
        tokens = re.findall(r"[a-z]+\d*|\bF\b|\bB\b|\bL\b|\bR\b", body)
        for token in tokens:
            upper = token.upper()
            if upper == "F":
                keys["forward"] = True
            elif upper == "B":
                keys["back"] = True
            elif upper == "L":
                keys["left"] = True
            elif upper == "R":
                keys["right"] = True
            elif token.lower() in ("jump", "sneak", "sprint", "attack", "use", "drop", "inventory"):
                keys[token.lower()] = True
            else:
                hotbar = re.fullmatch(r"h([1-9])", token.lower())
                if hotbar:
                    keys[f"hotbar.{hotbar.group(1)}"] = True
                    recognized = True
                continue
            recognized = True

    yaw_match = re.search(rf'cam_yaw"?\s*[=:]\s*"?({_SIGNED_INTEGER})', lowered)
    pitch_match = re.search(rf'cam_pitch"?\s*[=:]\s*"?({_SIGNED_INTEGER})', lowered)
    if yaw_match is None and pitch_match is None:
        compact = re.search(
            rf"cam\s*[=:]\s*({_SIGNED_INTEGER})\s*,\s*({_SIGNED_INTEGER})", lowered,
        )
        yaw = _parse_camera_offset(compact.group(1)) if compact else CAMERA_NEUTRAL_BIN
        pitch = _parse_camera_offset(compact.group(2)) if compact else CAMERA_NEUTRAL_BIN
        recognized = recognized or compact is not None
    else:
        yaw = _parse_camera_offset(yaw_match.group(1)) if yaw_match else CAMERA_NEUTRAL_BIN
        pitch = _parse_camera_offset(pitch_match.group(1)) if pitch_match else CAMERA_NEUTRAL_BIN
        recognized = True
    if not recognized:
        return None
    return StructuredAction(camera_yaw_bin=yaw, camera_pitch_bin=pitch, keys=keys)


def decode_actions(text: str, horizon: int) -> list[StructuredAction]:
    """把大模型生成的多行文本解码为定长结构化动作序列。

    Parameters
    ----------
    text : str
        大模型自由文本；逐行扫描，无法识别的行跳过。
    horizon : int
        期望的动作帧数。识别到的动作不足时用 noop 补齐，超出则截断，
        保证下游拿到定长、结构合法的动作块（AGENTS §5）。

    Returns
    -------
    list[StructuredAction]
        长度恰为 ``horizon`` 的结构化动作序列。
    """
    if horizon < 1:
        raise ValueError("horizon 必须大于零")
    decoded: list[StructuredAction] = []
    for line in text.splitlines():
        action = _decode_line(line)
        if action is not None:
            decoded.append(action)
        if len(decoded) >= horizon:
            break
    while len(decoded) < horizon:
        decoded.append(StructuredAction())
    return decoded[:horizon]


_CAMERA_RANGE = f"整数偏移，范围 [{-CAMERA_NEUTRAL_BIN}, {CAM_BINS - 1 - CAMERA_NEUTRAL_BIN}]，0 表示不动"

_FORMAT_DESCRIPTIONS = {
    ActionTokenFormat.COMPACT_TAG: (
        "每一帧输出一行，格式为 `t<下标>: <标签...>`。标签含义：\n"
        "  F=前进 B=后退（二选一）；L=左移 R=右移（二选一）；\n"
        "  jump/sneak/sprint/attack/use/drop/inventory 为动作，写出即表示按下；\n"
        "  sneak 与 sprint 二选一；h1..h9 表示切换到对应快捷栏槽位；\n"
        f"  cam=<yaw>,<pitch> 为相机增量（{_CAMERA_RANGE}）。\n"
        "  什么都不做时写 noop。\n"
        "样例：\n  t0: F sprint attack cam=+2,-1\n  t1: F R h3 cam=0,0"
    ),
    ActionTokenFormat.KEY_VALUE: (
        "每一帧输出一行，格式为 `t<下标>: move=<forward|back|none> turn=<left|right|none> "
        "stance=<sneak|sprint|none> cam_yaw=<int> cam_pitch=<int> "
        "buttons=<逗号分隔的jump/attack/use/drop/inventory或none> hotbar=<1..9或none>`。\n"
        f"  cam_yaw / cam_pitch 为{_CAMERA_RANGE}。\n"
        "样例：\n  t0: move=forward turn=none stance=sprint cam_yaw=2 cam_pitch=-1 buttons=attack hotbar=none\n"
        "  t1: move=forward turn=right stance=none cam_yaw=0 cam_pitch=0 buttons=none hotbar=3"
    ),
    ActionTokenFormat.JSON_LINE: (
        "每一帧输出一行 JSON 对象，字段：move(forward|back|none)、turn(left|right|none)、"
        "stance(sneak|sprint|none)、cam_yaw(int)、cam_pitch(int)、"
        "buttons(数组，取自 jump/attack/use/drop/inventory)、hotbar(1..9 或 null)。\n"
        f"  cam_yaw / cam_pitch 为{_CAMERA_RANGE}。\n"
        '样例：\n  t0: {"move": "forward", "turn": "none", "stance": "sprint", '
        '"cam_yaw": 2, "cam_pitch": -1, "buttons": ["attack"], "hotbar": null}\n'
        '  t1: {"move": "forward", "turn": "right", "stance": "none", '
        '"cam_yaw": 0, "cam_pitch": 0, "buttons": [], "hotbar": 3}'
    ),
}


def describe_format(action_format: ActionTokenFormat) -> str:
    """返回给大模型的该格式自然语言说明与样例。"""
    return _FORMAT_DESCRIPTIONS[action_format]
