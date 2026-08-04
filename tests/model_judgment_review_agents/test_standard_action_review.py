from model_judgment_review_agents import make_review_candidate


def test_review_mutation_uses_standard_action_syntax() -> None:
    answer = {"reference_action_sequence": ["Device KeyboardMouse\nTick 0\n<action>W</action>"]}

    candidate = make_review_candidate(answer, mutate=True)

    assert candidate.answer["reference_action_sequence"][0].endswith("W ; Q</action>")
    assert answer["reference_action_sequence"][0].endswith("W</action>")
