import json

from environment_validation_tools.artifacts import atomic_write_json


def test_atomic_write_json_uses_utf8(tmp_path) -> None:
    document = tmp_path / "result.json"

    atomic_write_json(document, {"结论": "通过"})

    assert json.loads(document.read_text(encoding="utf-8")) == {"结论": "通过"}
