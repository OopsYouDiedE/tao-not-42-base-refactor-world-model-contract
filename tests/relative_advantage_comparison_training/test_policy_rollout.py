import pytest

from relative_advantage_comparison_training.policy_rollout import (
    _allowed_next_tokens,
    policy_prompt,
)


def test_policy_prompt_places_protocol_headers_outside_action_block() -> None:
    prompt = policy_prompt("Move forward")

    assert "Device KeyboardMouse\nTick 0\n<action>NoOp</action>" in prompt
    assert "Replace NoOp" in prompt
    assert prompt.endswith("Intent: Move forward")


def test_protocol_candidate_trie_returns_only_valid_next_tokens() -> None:
    candidates = ((10, 20, 30), (10, 21), (10, 20, 31))

    assert _allowed_next_tokens((), candidates, 99) == [10]
    assert _allowed_next_tokens((10,), candidates, 99) == [20, 21]
    assert _allowed_next_tokens((10, 20), candidates, 99) == [30, 31]
    assert _allowed_next_tokens((10, 21), candidates, 99) == [99]


def test_protocol_candidate_trie_rejects_out_of_domain_prefix() -> None:
    with pytest.raises(RuntimeError, match="left the configured protocol candidate set"):
        _allowed_next_tokens((12,), ((10, 20),), 99)
