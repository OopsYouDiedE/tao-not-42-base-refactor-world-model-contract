"""TAO Temporal Action Protocol（TAP v1）。"""

from tao.protocols.action.codec import (
    ACTION_END,
    ACTION_START,
    MINECRAFT_KEYMAP,
    MOUSE_DELTA_LIMIT,
    PROTOCOL_VERSION,
    SCROLL_LIMIT,
    TICK_SEPARATOR,
    ActionSequence,
    ActionTick,
    decode_action_sequence,
    encode_action_sequence,
    press_release_events,
)
from tao.protocols.action.timing import (
    DEFAULT_WINDOW_FRAMES,
    FRAMES_PER_SECOND,
    HISTORY_FRAME_INTERVAL,
)
from tao.protocols.action.validation import (
    ActionSegment,
    action_ticks,
    compress_action_ticks,
    expand_action_segments,
    validate_action_image_alignment,
)

__all__ = [
    "ACTION_END",
    "ACTION_START",
    "TICK_SEPARATOR",
    "MINECRAFT_KEYMAP",
    "MOUSE_DELTA_LIMIT",
    "PROTOCOL_VERSION",
    "SCROLL_LIMIT",
    "ActionSegment",
    "ActionSequence",
    "ActionTick",
    "DEFAULT_WINDOW_FRAMES",
    "FRAMES_PER_SECOND",
    "HISTORY_FRAME_INTERVAL",
    "action_ticks",
    "compress_action_ticks",
    "decode_action_sequence",
    "encode_action_sequence",
    "expand_action_segments",
    "press_release_events",
    "validate_action_image_alignment",
]
