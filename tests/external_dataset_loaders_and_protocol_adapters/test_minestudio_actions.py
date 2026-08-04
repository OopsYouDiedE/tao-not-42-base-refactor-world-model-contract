import numpy as np

import external_dataset_loaders_and_protocol_adapters as adapters
from online_interactive_environments import (
    ActionSequence,
    ActionTick,
    format_action_sequence,
)


def test_minestudio_actions_use_standard_protocol_directly() -> None:
    actions = {
        "camera": np.array([[0.3, 0.6], [0.0, 0.0]]),
        "forward": np.array([1, 0]),
        "jump": np.array([1, 0]),
        "sneak": np.array([0, 1]),
        "sprint": np.array([0, 1]),
    }

    sequence = adapters.encode_minestudio_actions(actions, offset=7)

    assert sequence == ActionSequence(
        "KeyboardMouse",
        7,
        (
            ActionTick(("MouseMove", "4", "2", "W", "Space")),
            ActionTick(("Shift", "Ctrl")),
        ),
    )
    assert format_action_sequence(sequence).startswith("Device KeyboardMouse\nTick 7\n")


def test_adapter_package_has_no_legacy_action_protocol_api() -> None:
    assert not any(name.lower().startswith("tap") for name in dir(adapters))
