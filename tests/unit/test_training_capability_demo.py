"""八方面训练能力 demo 的确定性标签与数据契约测试。"""

from __future__ import annotations

import numpy as np

from bc_datasets.minestudio.lmdb_modal_reader import ModalKernelReader
from bc_datasets.training_capabillity_demo import (
    CAPABILITY_ASPECTS,
    action_contract_text,
    action_ticks,
    categorical_transition,
    coarse_inverse_dynamics,
    meaningful_events,
    state_transition,
)


def _actions() -> dict[str, np.ndarray]:
    return {
        "camera": np.asarray(
            [[0.0, 0.3], [-0.15, 0.0], [0.0, 0.0], [0.0, -0.3]],
            dtype=np.float64,
        ),
        "forward": np.asarray([1, 1, 1, 0], dtype=np.int64),
        "attack": np.asarray([0, 0, 1, 1], dtype=np.int64),
    }


def test_demo_declares_eight_distinct_capability_aspects() -> None:
    assert len(CAPABILITY_ASPECTS) == 8
    assert len(set(CAPABILITY_ASPECTS)) == 8


def test_action_ticks_use_tick_local_mouse_and_final_mouse_key_names() -> None:
    ticks = action_ticks(_actions())
    assert ticks[0] == {"keys": ["W"], "mouse": [2, 0]}
    assert ticks[2]["keys"] == ["W", "MouseLeft"]
    assert ticks[3] == {"keys": ["MouseLeft"], "mouse": [-2, 0]}


def test_action_contract_text_uses_named_mouse_tokens_per_tick() -> None:
    text = action_contract_text(_actions())
    assert text == (
        "<|action_start|> ; W Mouse 2 0 ; W Mouse 0 -1 ; "
        "W MouseLeft ; MouseLeft Mouse -2 0 <|action_end|>"
    )
    assert "mouse_left" not in text


def test_meaningful_events_drop_only_timer_events() -> None:
    metadata = [
        {"events": {
            "minecraft.custom:minecraft.play_one_minute": 1,
            "minecraft.custom:minecraft.walk_one_cm": 20,
            "mine_block:stone": 1,
        }},
        {"events": {"mine_block:stone": 2, "pickup:cobblestone": 1}},
    ]
    assert meaningful_events(metadata) == {
        "mine_block:stone": 3.0,
        "pickup:cobblestone": 1.0,
    }


def test_state_transition_separates_before_and_after_fields() -> None:
    before = {
        "xpos": 1.0, "ypos": 2.0, "zpos": 3.0, "yaw": 10.0, "pitch": 5.0,
        "isGuiOpen": False, "isGuiInventory": False, "hotbar": 0,
    }
    after = {
        "xpos": 1.5, "ypos": 2.0, "zpos": 2.0, "yaw": 12.0, "pitch": 4.0,
        "isGuiOpen": True, "isGuiInventory": True, "hotbar": 2,
    }
    transition = state_transition(before, after)
    assert transition["position_delta"] == {"xpos": 0.5, "ypos": 0.0, "zpos": -1.0}
    assert transition["gui_before"] is False
    assert transition["gui_after"] is True
    assert transition["hotbar_after"] == 3


def test_categorical_transition_avoids_brittle_exact_position_target() -> None:
    before = {
        "xpos": 1.0, "ypos": 2.0, "zpos": 3.0, "yaw": 10.0, "pitch": 5.0,
        "isGuiOpen": False, "isGuiInventory": False, "hotbar": 0,
    }
    after = {
        "xpos": 1.5, "ypos": 2.001, "zpos": 2.0, "yaw": 12.0, "pitch": 4.0,
        "isGuiOpen": False, "isGuiInventory": False, "hotbar": 0,
    }
    assert categorical_transition(before, after) == {
        "position_direction": {"xpos": "positive", "ypos": "stable", "zpos": "negative"},
        "view_direction": {"yaw": "positive", "pitch": "negative"},
        "gui_change": "unchanged",
        "hotbar_changed": False,
    }


def test_coarse_inverse_dynamics_avoids_unobservable_tick_sequence() -> None:
    assert coarse_inverse_dynamics(_actions()) == {
        "movement_keys": ["W"],
        "interaction_keys": ["MouseLeft"],
        "camera": {
            "pitch_direction": "negative",
            "yaw_direction": "stable",
            "magnitude": "small",
        },
    }


def test_meta_info_chunks_merge_as_frame_lists() -> None:
    reader = object.__new__(ModalKernelReader)
    reader.modal = "meta_info"
    assert reader._merge_chunks([[{"frame": 0}], [{"frame": 1}]]) == [
        {"frame": 0}, {"frame": 1},
    ]
