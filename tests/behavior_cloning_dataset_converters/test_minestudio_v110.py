import json
from pathlib import Path

from behavior_cloning_dataset_converters.minestudio_v110 import (
    build_split,
    format_assistant_response,
    format_question_prompt,
    parse_episode_identity,
    sanitize_intent,
)


def test_v110_split_holds_out_complete_prefix(tmp_path: Path) -> None:
    first = "lovely-persimmon-angora-02e496ce4abb-20220421-092639"
    second = "lovely-persimmon-angora-02e496ce4abb-20220421-092640"
    third = "other-prefix-f153ac423f61-20220422-102639"

    output_path = tmp_path / "split.json"
    result = build_split(
        episode_frames={first: 30, second: 30, third: 60},
        validation_ratio=0.5,
        output_path=output_path,
    )

    assert result.holdout_level == "prefix"
    assert result.validation_episodes in ([first, second], [third])
    assert result.validation_groups in (["lovely-persimmon-angora"], ["other-prefix"])
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "validation_prefixes" in payload
    assert "validation_groups" not in payload


def test_v110_episode_identity_contract() -> None:
    identity = parse_episode_identity("lovely-persimmon-angora-02e496ce4abb-20220421-092639")

    assert identity.prefix == "lovely-persimmon-angora"
    assert identity.session == "02e496ce4abb"
    assert identity.date == "20220421"
    assert identity.time == "092639"


def test_v110_question_and_answer_use_standard_action_protocol() -> None:
    question = {
        "task_type": "image_sequence_to_action",
        "prompt": "obsolete",
        "inputs": {"action_block_ticks": [8]},
    }
    answer = {"reference_action_sequence": ["Device KeyboardMouse\nTick 0\n<action>\n"]}

    prompt = format_question_prompt(question)
    response = format_assistant_response(question, answer)

    assert "Minecraft image transition" in prompt
    assert "standard-input-action/v1" in prompt
    assert "Required action-block tick counts: [8]" in prompt
    assert response.endswith(
        "Reason: The action sequence follows the visible transition, intent, and required duration."
    )


def test_v110_intent_cleanup_preserves_historical_parenthesis_contract() -> None:
    assert sanitize_intent("向前移动（8 ticks， 然后跳跃") == "向前移动（然后跳跃"
