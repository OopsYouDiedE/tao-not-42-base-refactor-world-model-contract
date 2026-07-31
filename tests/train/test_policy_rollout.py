"""Policy rollout token 边界回归测试。"""

from __future__ import annotations

import torch

from train.policy_rollout import _truncate_at_action_end


class BoundaryMergingTokenizer:
    """模拟 Gemma 将标记前空白与左尖括号合并的编码。"""

    markers = {
        "<|action_end|>": [10, 11],
        " <|action_end|>": [20, 11],
        "\n<|action_end|>": [30, 11],
    }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self.markers[text]

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        return str(token_ids.tolist())


def test_truncate_accepts_marker_whose_leading_space_is_merged() -> None:
    tokenizer = BoundaryMergingTokenizer()
    generated = torch.tensor([1, 2, 20, 11, 99, 100])

    truncated = _truncate_at_action_end(generated, tokenizer)

    assert truncated.tolist() == [1, 2, 20, 11]


def test_truncate_uses_first_complete_marker() -> None:
    tokenizer = BoundaryMergingTokenizer()
    generated = torch.tensor([1, 10, 11, 2, 20, 11])

    truncated = _truncate_at_action_end(generated, tokenizer)

    assert truncated.tolist() == [1, 10, 11]
