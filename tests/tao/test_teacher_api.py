import json
from pathlib import Path

import pytest

from tao.baselines.teacher_api import TeacherAPIConfig
from tools.export_codex_api_env import load_exports, shell_exports


def test_teacher_api_requires_exported_model_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("API_KEY", "API_MODEL", "API_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="必须 export API_KEY, API_MODEL, API_URL"):
        TeacherAPIConfig.from_environment()


def test_teacher_api_loads_exported_model_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("API_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("API_URL", "https://example.test/v1")

    config = TeacherAPIConfig.from_environment()

    assert config.model == "gpt-5.6-sol"
    assert config.audit_dict()["api_key"] == "<redacted>"


def test_teacher_api_rejects_invalid_url() -> None:
    with pytest.raises(ValueError, match="完整的 HTTP"):
        TeacherAPIConfig("secret", "model", "example.test/v1")


def test_load_exports_reads_selected_global_provider(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '\n'.join(
            (
                'model = "gpt-5.6-sol"',
                'model_provider = "proxy"',
                '[model_providers.proxy]',
                'base_url = "https://example.test/v1"',
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "secret value"}), encoding="utf-8"
    )

    values = load_exports(tmp_path)

    assert values == {
        "API_KEY": "secret value",
        "API_MODEL": "gpt-5.6-sol",
        "API_URL": "https://example.test/v1",
    }
    assert "export API_KEY='secret value'" in shell_exports(values)
