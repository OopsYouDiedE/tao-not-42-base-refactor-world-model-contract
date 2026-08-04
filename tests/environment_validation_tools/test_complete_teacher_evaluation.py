import json
from pathlib import Path

from environment_validation_tools import run_complete_teacher_evaluation as complete


def test_single_command_entry_creates_baseline_runs_trajectories_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_create(output: Path, *, port: int, warmup_ticks: int) -> Path:
        calls["baseline_port"] = port
        calls["baseline_warmup"] = warmup_ticks
        world = output / "baseline-world"
        world.mkdir(parents=True)
        (world / "level.dat").write_bytes(b"fixed")
        return output / "baseline-world-manifest.json"

    def fake_run(output: Path, **kwargs) -> Path:
        calls["trajectory_output"] = output
        calls["trajectory_kwargs"] = kwargs
        output.mkdir(parents=True)
        (output / "shared-start-state.json").write_text("{}", encoding="utf-8")
        (output / "shared-start.png").write_bytes(b"png")
        (output / "README.md").write_text("# run", encoding="utf-8")
        trajectories = []
        comparisons = []
        for index in range(4):
            trajectory_id = f"T{index + 1:02d}"
            directory = output / trajectory_id
            directory.mkdir()
            (directory / "trajectory.md").write_text("# trajectory", encoding="utf-8")
            trajectories.append(
                {
                    "trajectory_id": trajectory_id,
                    "environment_slot": index,
                    "generation_count": 1,
                    "executed_ticks": 1,
                    "wall_clock_duration_seconds": 1.0,
                    "trajectory_success": index == 3,
                    "trajectory_error": None,
                }
            )
            comparisons.append(
                {
                    "trajectory_id": trajectory_id,
                    "rank": 4 - index,
                    "relative_advantage": float(index),
                }
            )
        payload = {
            "backend": "codex-cli",
            "model": "gpt-5.6-sol",
            "action_protocol": "standard-input-action/v1",
            "action_budget_ticks_per_arm": 8,
            "shared_start": {
                "environment_transport_backend": "socket",
                "restore_probe_passed": True,
            },
            "trajectories": trajectories,
            "comparison_samples": comparisons,
            "comparison_review": {
                "valid": True,
                "selected_trajectory_ids": ["T04"],
            },
        }
        (output / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        return output / "README.md"

    monkeypatch.setattr(complete, "create_baseline_world", fake_create)
    monkeypatch.setattr(complete, "run_four", fake_run)

    report = complete.run_complete(
        tmp_path / "run",
        port_base=21000,
        action_budget_ticks=8,
        max_generations=2,
        warmup_ticks=3,
    )
    report_text = report.read_text(encoding="utf-8")

    assert calls["baseline_port"] == 20999
    assert calls["baseline_warmup"] == 3
    assert calls["trajectory_output"] == (tmp_path / "run" / "trajectories").resolve()
    assert calls["trajectory_kwargs"]["use_shared_memory"] is False
    assert calls["trajectory_kwargs"]["action_budget_ticks"] == 8
    assert "CraftGround 四轨迹教师测评报告" in report_text
    assert "| 最佳轨迹 | T04 |" in report_text
    assert "[T04 轨迹报告](trajectories/T04/trajectory.md)" in report_text
