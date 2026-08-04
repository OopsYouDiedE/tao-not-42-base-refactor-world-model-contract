"""环境与项目结构验证工具。"""

from pathlib import Path


def validate_project_structure(root: Path | None = None) -> tuple[str, ...]:
    from .validate_project_structure import validate_project_structure as validate

    return validate(root)


__all__ = ["validate_project_structure"]
