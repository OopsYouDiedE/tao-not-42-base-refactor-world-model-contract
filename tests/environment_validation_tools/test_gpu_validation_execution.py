from environment_validation_tools.generate_gpu_validation_execution import (
    build_execution_payload,
)
from relative_advantage_comparison_training import PolicyGeneration


def test_build_execution_payload_preserves_six_on_policy_generations() -> None:
    generations = [
        PolicyGeneration(
            "Device KeyboardMouse\nTick 0\n<action>W</action>",
            (10, 11),
            (-0.1, -0.2),
            "policy-v1",
            {"temperature": 0.8},
        )
        for _ in range(6)
    ]

    payload = build_execution_payload(
        generations,
        frame_name="observation.png",
        reference_action_text="Device KeyboardMouse\nTick 0\n<action>W</action>",
    )

    trajectories = payload["trajectories"]
    assert len(trajectories) == 8
    assert [item["source_role"] for item in trajectories].count("reference_expert") == 2
    assert [item["source_role"] for item in trajectories].count("policy_sample") == 6
    assert trajectories[-1]["response_token_ids"] == [10, 11]
    assert trajectories[-1]["old_logprobs"] == [-0.1, -0.2]
