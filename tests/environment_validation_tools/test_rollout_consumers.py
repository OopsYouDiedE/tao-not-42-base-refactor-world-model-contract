from interaction_trajectory_review_agents import review_trajectory
from model_judgment_review_agents import review_comparison
from relative_advantage_comparison_training import build_comparison_group


def test_review_comparison_pipeline_centers_relative_advantages() -> None:
    trajectory = {
        "trajectory_id": "T01",
        "action_protocol": "standard-input-action/v1",
        "action_backend": "keyboard_and_mouse_only",
        "execution_ticks": [{"tick": 0}],
        "generation_records": [{"status": "completed"}],
    }
    successful = review_trajectory(
        trajectory,
        {
            "trajectory_id": "T01",
            "executed_ticks": 1,
            "trajectory_success": True,
            "trajectory_error": None,
        },
        action_budget_ticks=4,
    )
    unsuccessful = review_trajectory(
        {**trajectory, "trajectory_id": "T02"},
        {
            "trajectory_id": "T02",
            "executed_ticks": 1,
            "trajectory_success": False,
            "trajectory_error": None,
        },
        action_budget_ticks=4,
    )

    samples = build_comparison_group((successful, unsuccessful))
    judgment = review_comparison(samples)

    assert round(sum(item.relative_advantage for item in samples), 6) == 0
    assert samples[0].selected is True
    assert samples[0].rank == 1
    assert judgment.valid is True
    assert judgment.selected_trajectory_ids == ("T01",)
