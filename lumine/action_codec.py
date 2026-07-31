"""开放帧率的命名 token 动作格式编解码。

对外接口：
    ACTION_START, ACTION_END, CHUNK_SEPARATOR — 动作串的分隔标记。
    MINECRAFT_KEYMAP — MineStudio 动作字段 → Lumine 键名。
    LumineActionChunk — 单个电机 chunk 的键集合。
    LumineWindowAction — 一个感知窗口的变长 chunk 序列。
    encode_lumine_action — MineStudio 窗口动作 → Lumine 动作串。
    decode_lumine_action — Lumine 动作串 → LumineWindowAction。
    press_release_events — 从相邻窗口的键集合推出按下 / 松开事件。

每个分号分隔一个执行 tick，chunk 数量不固定。按键在相邻 chunk 连续出现表示持续按住，
缺席表示松开。带值操作使用命名 token，并只作用于所在 tick。鼠标相对移动写成
``Mouse dx dy``，可以与同 tick 的键名混排；没有同时动作时单独写 Mouse 更清晰。

串形状::

    <|action_start|> ; Mouse 35 30 ; W D ; Mouse 4 -2 W D <|action_end|>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

ACTION_START = "<|action_start|>"
ACTION_END = "<|action_end|>"
CHUNK_SEPARATOR = ";"

# 鼠标像素增量的钳位范围（Lumine 原文为开区间 (-1000, 1000)，取整后按 ±999 钳）。
MOUSE_DELTA_LIMIT = 999
# 滚轮档位钳位范围。
SCROLL_LIMIT = 5
# VPT / MineStudio 的相机换算：每像素 0.15 度。
DEGREES_PER_PIXEL = 0.15

# MineStudio 动作字段 → Lumine 键名。Lumine 用 PC 实际按键，这里取 Minecraft 默认键位；
# 鼠标左右键给语义名，避免与键盘键混淆。
MINECRAFT_KEYMAP: dict[str, str] = {
    "forward": "W",
    "back": "S",
    "left": "A",
    "right": "D",
    "jump": "space",
    "sneak": "shift",
    "sprint": "ctrl",
    "attack": "MouseLeft",
    "use": "MouseRight",
    "drop": "Q",
    "inventory": "E",
    "hotbar.1": "1",
    "hotbar.2": "2",
    "hotbar.3": "3",
    "hotbar.4": "4",
    "hotbar.5": "5",
    "hotbar.6": "6",
    "hotbar.7": "7",
    "hotbar.8": "8",
    "hotbar.9": "9",
}


@dataclass(frozen=True)
class LumineActionChunk:
    """单个电机 chunk 的按键和相对鼠标移动。"""

    keys: tuple[str, ...]
    mouse: tuple[int, int] = (0, 0)
    scroll: int = 0


@dataclass(frozen=True)
class LumineWindowAction:
    """一个感知窗口内按时间排列的变长动作 chunk。"""

    chunks: tuple[LumineActionChunk, ...]

    def to_text(self) -> str:
        """序列化为命名 token 动作串。"""
        encoded: list[str] = []
        for chunk in self.chunks:
            tokens: list[str] = []
            if chunk.mouse != (0, 0):
                tokens.extend(("Mouse", str(chunk.mouse[0]), str(chunk.mouse[1])))
            if chunk.scroll:
                tokens.extend(("Scroll", str(chunk.scroll)))
            tokens.extend(chunk.keys)
            encoded.append(" ".join(tokens))
        body = f" {CHUNK_SEPARATOR} ".join(encoded)
        return f"{ACTION_START} {CHUNK_SEPARATOR} {body} {ACTION_END}"


def _clamp(value: float, limit: int) -> int:
    """四舍五入到整数并对称钳位到 ``±limit``。"""
    rounded = int(np.rint(value))
    return max(-limit, min(limit, rounded))


def encode_lumine_action(
    actions: dict[str, np.ndarray],
    frames_per_chunk: int = 1,
    keymap: dict[str, str] | None = None,
    degrees_per_pixel: float = DEGREES_PER_PIXEL,
) -> LumineWindowAction:
    """把一个感知窗口的 MineStudio 动作编码成 Lumine 窗口动作。

    Parameters
    ----------
    actions : dict of str to numpy.ndarray
        MineStudio ``action`` 模态的窗口切片：二值键为 shape (T,) 的 0/1 数组，
        ``camera`` 为 shape (T, 2) 的浮点数组，列序 ``[pitch, yaw]``，单位度。
    frames_per_chunk : int
        每个电机 chunk 覆盖的帧数。MineStudio 为 20Hz（50ms/帧），取 1 即
        50ms/chunk；取 2 则 100ms/chunk。窗口帧数需能被其整除。
    keymap : dict of str to str or None
        动作字段 → 键名映射，None 表示用 ``MINECRAFT_KEYMAP``。
    degrees_per_pixel : float
        相机度数 → 鼠标像素的换算系数。

    Returns
    -------
    LumineWindowAction
        鼠标增量位于对应 chunk 内，chunk 列表长度为 ``T // frames_per_chunk``。

    Raises
    ------
    ValueError
        窗口为空、帧数不能被 ``frames_per_chunk`` 整除，或缺少 ``camera`` 字段。

    Notes
    -----
    一个 chunk 覆盖多帧时，只要键在该 chunk 的**任一帧**按下就记为按住——这是把
    50ms 采样降到更粗电机步时唯一不丢短按的选择。
    """
    if frames_per_chunk < 1:
        raise ValueError("frames_per_chunk 必须 >= 1")
    if "camera" not in actions:
        raise ValueError("actions 缺少 camera 字段")
    mapping = MINECRAFT_KEYMAP if keymap is None else keymap
    camera = np.asarray(actions["camera"], dtype=np.float64)
    if camera.ndim != 2 or camera.shape[1] != 2:
        raise ValueError(f"camera 应为 shape (T, 2)，实际 {camera.shape}")
    num_frames = camera.shape[0]
    if num_frames == 0:
        raise ValueError("动作窗口不能为空")
    if num_frames % frames_per_chunk != 0:
        raise ValueError(
            f"窗口帧数 {num_frames} 不能被 frames_per_chunk {frames_per_chunk} 整除",
        )

    if degrees_per_pixel <= 0.0:
        raise ValueError("degrees_per_pixel 必须为正")

    # 一次性堆成 (键数, 帧数) 布尔矩阵再按 chunk 归约。逐键逐 chunk 调 np.any 的写法
    # 在全量构建里会产生百万级单元素 numpy 调用，纯属调用开销。
    present = [(field, name) for field, name in mapping.items() if field in actions]
    chunks: list[LumineActionChunk] = []
    if present:
        matrix = np.stack(
            [np.asarray(actions[field]).astype(bool) for field, _ in present],
        )
        if matrix.shape[1] != num_frames:
            raise ValueError(
                f"按键帧数 {matrix.shape[1]} 与 camera 帧数 {num_frames} 不一致",
            )
        held = matrix.reshape(len(present), -1, frames_per_chunk).any(axis=2)
        names = [name for _, name in present]
        chunk_camera = camera.reshape(-1, frames_per_chunk, 2).sum(axis=1)
        for column, (pitch, yaw) in zip(held.T, chunk_camera, strict=True):
            chunks.append(
                LumineActionChunk(
                    keys=tuple(name for name, flag in zip(names, column, strict=True) if flag),
                    mouse=(
                        _clamp(yaw / degrees_per_pixel, MOUSE_DELTA_LIMIT),
                        _clamp(pitch / degrees_per_pixel, MOUSE_DELTA_LIMIT),
                    ),
                ),
            )
    else:
        for pitch, yaw in camera.reshape(-1, frames_per_chunk, 2).sum(axis=1):
            chunks.append(
                LumineActionChunk(
                    keys=(),
                    mouse=(
                        _clamp(yaw / degrees_per_pixel, MOUSE_DELTA_LIMIT),
                        _clamp(pitch / degrees_per_pixel, MOUSE_DELTA_LIMIT),
                    ),
                )
            )

    # MineStudio / VPT 的动作空间没有滚轮，快捷栏走数字键，故 ΔZ 恒为 0。
    return LumineWindowAction(chunks=tuple(chunks))


_ACTION_PATTERN = re.compile(
    re.escape(ACTION_START) + r"(?P<body>.*?)" + re.escape(ACTION_END),
    re.DOTALL,
)


def decode_lumine_action(
    text: str,
    allowed_keys: frozenset[str] | None = None,
    expected_chunks: int | None = None,
) -> LumineWindowAction:
    """把 Lumine 动作串解析回结构化窗口动作。

    Parameters
    ----------
    text : str
        含 ``<|action_start|>…<|action_end|>`` 的文本；多段时只取第一段。
    allowed_keys : frozenset of str or None
        允许的键名白名单，None 表示接受 ``MINECRAFT_KEYMAP`` 的全部键名。
        白名单外的 token 被丢弃，而不是抛错。
    expected_chunks : int or None
        期望的 chunk 数；给定时不足补空 chunk、超出截断。

    Returns
    -------
    LumineWindowAction
        解析结果。鼠标增量与滚轮已钳位，键名已去重并按首次出现顺序保留。

    Raises
    ------
    ValueError
        文本里找不到成对的动作标记。

    Notes
    -----
    解码对脏输出是结构容错的：缺失或非法的数值按 0 处理，未知键名丢弃，chunk 数按
    ``expected_chunks`` 对齐。这样大模型无论吐出什么，都能落到定长、合法的动作块上。
    """
    matched = _ACTION_PATTERN.search(text)
    if matched is None:
        raise ValueError("文本中没有成对的 action 标记")
    valid = frozenset(MINECRAFT_KEYMAP.values()) if allowed_keys is None else allowed_keys
    segments = matched.group("body").split(CHUNK_SEPARATOR)

    chunks: list[LumineActionChunk] = []
    for segment in segments[1:] if not segments[0].strip() else segments:
        seen: dict[str, None] = {}
        mouse = (0, 0)
        scroll = 0
        tokens = segment.split()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in valid:
                seen.setdefault(token, None)
                index += 1
            elif token == "Mouse":
                try:
                    mouse = (
                        _clamp(float(tokens[index + 1]), MOUSE_DELTA_LIMIT),
                        _clamp(float(tokens[index + 2]), MOUSE_DELTA_LIMIT),
                    )
                except (IndexError, ValueError):
                    mouse = (0, 0)
                index += 3
            elif token == "Scroll":
                try:
                    scroll = _clamp(float(tokens[index + 1]), SCROLL_LIMIT)
                except (IndexError, ValueError):
                    scroll = 0
                index += 2
            else:
                index += 1
        chunks.append(LumineActionChunk(keys=tuple(seen), mouse=mouse, scroll=scroll))

    if expected_chunks is not None:
        chunks = chunks[:expected_chunks]
        while len(chunks) < expected_chunks:
            chunks.append(LumineActionChunk(keys=()))

    return LumineWindowAction(chunks=tuple(chunks))


def press_release_events(
    chunks: tuple[LumineActionChunk, ...],
    previously_held: frozenset[str] = frozenset(),
) -> list[tuple[frozenset[str], frozenset[str]]]:
    """由 run-length 键集合推出逐 chunk 的按下 / 松开事件。

    Parameters
    ----------
    chunks : tuple of LumineActionChunk
        窗口内各 chunk 的键集合。
    previously_held : frozenset of str
        进入本窗口前仍按住的键，用于跨窗口保持按下状态。

    Returns
    -------
    list of tuple
        每个 chunk 一项 ``(pressed, released)``：``pressed`` 是本 chunk 新按下的键，
        ``released`` 是本 chunk 松开的键。连续按住的键不会重复出现在 ``pressed`` 里。
    """
    events: list[tuple[frozenset[str], frozenset[str]]] = []
    held = previously_held
    for chunk in chunks:
        current = frozenset(chunk.keys)
        events.append((current - held, held - current))
        held = current
    return events
