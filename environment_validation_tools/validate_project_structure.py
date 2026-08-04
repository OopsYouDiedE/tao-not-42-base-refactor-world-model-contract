"""验证重组期间的顶层目录和当前文档入口。"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_DIRECTORIES = (
    "external_dataset_loaders_and_protocol_adapters",
    "behavior_cloning_dataset_converters",
    "behavior_cloning_training",
    "online_interactive_environments",
    "interaction_trajectory_review_agents",
    "online_environment_interaction_agents",
    "model_judgment_review_agents",
    "relative_advantage_comparison_training",
    "environment_validation_tools",
    "shared_tools",
    "tests",
)

REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    "docs/INSTALLATION.md",
    "scripts/bootstrap.sh",
    "scripts/bootstrap_gpu_craftground.sh",
    "online_interactive_environments/STANDARD_INPUT_ACTION_PROTOCOL.md",
    "online_interactive_environments/CRAFTGROUND_KEYBOARD_AND_MOUSE_ONLY_BACKEND.md",
    "online_environment_interaction_agents/TRAJECTORY_GENERATION_PROMPT.md",
    "online_interactive_environments/action_sequence_compiler.py",
    "online_interactive_environments/craftground/runtime.py",
)

LEGACY_README_PATHS = (
    "`tao/`",
    "`dataset/`",
    "`train/`",
    "`game_environment/`",
    "`tools/`",
    "`scripts/`",
)

FORBIDDEN_ARTIFACT_DIRECTORIES = (
    "artifacts",
    "environment_validation_tools/test_runs",
    "environment_validation_tools/runs",
    "online_environment_interaction_agents/test_runs",
    "online_environment_interaction_agents/runs",
    "online_interactive_environments/test_runs",
    "online_interactive_environments/runs",
)


def find_artifact_issues(project_root: Path) -> tuple[str, ...]:
    return tuple(
        f"运行产物不得放在源码树中：{relative_path}"
        for relative_path in FORBIDDEN_ARTIFACT_DIRECTORIES
        if (project_root / relative_path).exists()
    )


def validate_project_structure(root: Path | None = None) -> tuple[str, ...]:
    project_root = root or Path(__file__).resolve().parents[1]
    issues: list[str] = []

    for relative_path in REQUIRED_DIRECTORIES:
        if not (project_root / relative_path).is_dir():
            issues.append(f"缺少目录：{relative_path}")

    for relative_path in REQUIRED_FILES:
        if not (project_root / relative_path).is_file():
            issues.append(f"缺少文件：{relative_path}")

    issues.extend(find_artifact_issues(project_root))

    readme_path = project_root / "README.md"
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        for legacy_path in LEGACY_README_PATHS:
            declaration = f"| {legacy_path}"
            if declaration in readme:
                issues.append(f"README 仍把旧路径声明为当前结构：{legacy_path}")
        if "标准输入动作协议 v1" not in readme:
            issues.append("README 未声明标准输入动作协议 v1")
        if "`keyboard_and_mouse_only`" not in readme:
            issues.append("README 未声明 keyboard_and_mouse_only 后端")

    return tuple(issues)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    issues = validate_project_structure()
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print("项目结构检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
