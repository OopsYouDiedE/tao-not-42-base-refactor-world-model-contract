import json

from train.review_rlhf_contract import (
    make_review_candidate,
    parse_review,
    reference_review,
    relative_advantages,
    score_review,
)

ANSWER = {"reference_action_sequence": ["<|action_start|> ; W ; W <|action_end|>"]}


def test_synthetic_candidate_is_explicit_and_does_not_modify_source() -> None:
    candidate = make_review_candidate(ANSWER, mutate=True)
    assert candidate.candidate_origin == "synthetic_mutation"
    assert candidate.mutation_type == "unsupported_key"
    assert "Drop" in candidate.answer["reference_action_sequence"][0]
    assert "Drop" not in ANSWER["reference_action_sequence"][0]


def test_reference_reviews_obey_schema_and_receive_maximum_reward() -> None:
    for mutate in (False, True):
        candidate = make_review_candidate(ANSWER, mutate=mutate)
        review = reference_review(candidate)
        assert parse_review(review) is not None
        reward, metrics = score_review(review, candidate)
        assert reward == (125.0 if mutate else 115.0)
        assert metrics["decision_correct"] is True


def test_false_approve_is_strongly_penalized() -> None:
    candidate = make_review_candidate(ANSWER, mutate=True)
    payload = json.loads(reference_review(make_review_candidate(ANSWER, mutate=False)))
    reward, metrics = score_review(json.dumps(payload), candidate)
    assert reward < 0
    assert metrics["false_approve"] is True


def test_invalid_json_and_group_size_are_rejected() -> None:
    assert score_review("approve", make_review_candidate(ANSWER, mutate=False))[0] == -40
    assert sum(relative_advantages(list(range(8)))) == 0
