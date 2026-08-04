from pathlib import Path

import pytest

from shared_tools.configuration import EnvironmentConfigurationError, load_env_file


def test_env_file_does_not_override_existing_values(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("EXISTING=new\nQUOTED='value'\n", encoding="utf-8")
    target = {"EXISTING": "old"}

    loaded = load_env_file(path, environ=target)

    assert loaded == ("QUOTED",)
    assert target == {"EXISTING": "old", "QUOTED": "value"}


def test_env_file_rejects_invalid_names(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("INVALID-NAME=value\n", encoding="utf-8")

    with pytest.raises(EnvironmentConfigurationError, match="无效环境变量名"):
        load_env_file(path, environ={})
