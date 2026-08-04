from online_environment_interaction_agents import (
    parse_teacher_decision,
)


def test_decision_envelope_preserves_pure_control() -> None:
    decision = parse_teacher_decision(
        "\nDevice KeyboardMouse\nTick 0\n<action>MouseMove 30 0 ; NoOp</action>\n"
    )

    assert decision.control.startswith("Device KeyboardMouse")
