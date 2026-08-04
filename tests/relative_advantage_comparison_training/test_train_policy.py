import torch

from relative_advantage_comparison_training.train_policy import normalized_candidate_logprob


def test_candidate_logprob_normalizes_only_over_allowed_tokens() -> None:
    logits = torch.zeros(128, dtype=torch.float32)
    logits[ord("A")] = 1.0
    logits[ord("B")] = 2.0

    value = normalized_candidate_logprob(logits, ord("A"), [ord("A"), ord("B")])

    assert torch.allclose(value, torch.tensor(-1.3132616))
