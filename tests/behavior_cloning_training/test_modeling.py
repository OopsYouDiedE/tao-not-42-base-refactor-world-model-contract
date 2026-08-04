import pytest

from behavior_cloning_training.modeling import _chat_template


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit", None),
        ("unsloth/gemma-4-4B-it", "gemma-4"),
        ("unsloth/gemma-4-26B-it", "gemma-4-thinking"),
    ],
)
def test_chat_template_matches_supported_unsloth_name(model: str, expected: str) -> None:
    assert _chat_template(model) == expected


def test_chat_template_rejects_unsupported_model_family() -> None:
    with pytest.raises(ValueError, match="unsupported vision model family"):
        _chat_template("owner/text-only-model")
