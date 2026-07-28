import numpy as np

from bc_datasets.minestudio.mechanical_segmentation_probe import intent_preserving_segments


def make_actions(length: int = 20) -> dict[str, np.ndarray]:
    actions = {name: np.zeros(length, dtype=np.int64) for name in (
        "forward", "back", "left", "right", "jump", "attack", "use",
    )}
    actions["camera"] = np.zeros((length, 2), dtype=np.float64)
    return actions


def test_short_mouse_pause_does_not_split_intent() -> None:
    actions = make_actions()
    actions["camera"][:8, 1] = 1.0
    actions["camera"][10:18, 1] = 1.0
    segments = intent_preserving_segments(actions, minimum_length=4, bridge_frames=2)
    assert (0, 18) in segments


def test_different_key_intents_remain_separate() -> None:
    actions = make_actions()
    actions["forward"][:8] = 1
    actions["right"][8:16] = 1
    segments = intent_preserving_segments(actions, minimum_length=4, bridge_frames=2)
    assert (0, 8) in segments
    assert (8, 16) in segments
