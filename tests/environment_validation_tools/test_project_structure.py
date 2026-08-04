from pathlib import Path

from environment_validation_tools import validate_project_structure
from environment_validation_tools.validate_project_structure import find_artifact_issues
from online_interactive_environments.craftground import (
    ACTION_BACKEND,
    CRAFTGROUND_ACTION_SPACE,
)


def test_current_project_structure_matches_readme_contract() -> None:
    root = Path(__file__).resolve().parents[2]

    assert validate_project_structure(root) == ()


def test_runtime_artifacts_are_rejected_inside_source_packages(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "environment_validation_tools" / "test_runs"
    artifact_directory.mkdir(parents=True)

    assert find_artifact_issues(tmp_path) == (
        "运行产物不得放在源码树中：environment_validation_tools/test_runs",
    )


def test_craftground_backend_has_project_and_upstream_names() -> None:
    assert ACTION_BACKEND == "keyboard_and_mouse_only"
    assert CRAFTGROUND_ACTION_SPACE == "V2_MINERL_HUMAN"
