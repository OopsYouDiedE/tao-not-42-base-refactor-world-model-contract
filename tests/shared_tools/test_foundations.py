import json
from pathlib import Path

import pytest

from shared_tools.artifacts import append_jsonl, atomic_write_json
from shared_tools.authentication import (
    SecretValue,
    check_github_authentication,
    check_huggingface_authentication,
)
from shared_tools.datasets import dataset_id_from_repo_id, dataset_path, publish_dataset


def test_secret_value_never_displays_plaintext() -> None:
    secret = SecretValue("private-value")

    assert str(secret) == "***"
    assert repr(secret) == "SecretValue('***')"
    assert secret.reveal() == "private-value"


def test_skipped_cli_authentication_does_not_invoke_external_service() -> None:
    assert check_github_authentication(skip=True).status == "skipped"
    assert check_huggingface_authentication(skip=True).status == "skipped"


def test_dataset_paths_follow_project_contract() -> None:
    assert dataset_id_from_repo_id("owner/dataset") == "owner_dataset"
    assert dataset_path("owner/dataset", "external_dataset") == Path(
        "runs/external_dataset/owner_dataset"
    )


def test_dataset_publish_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="confirm_publish=True"):
        publish_dataset(
            tmp_path,
            "owner/dataset",
            private=True,
            commit_message="test",
        )


def test_artifact_writers_use_utf8_and_complete_lines(tmp_path: Path) -> None:
    document = tmp_path / "result.json"
    events = tmp_path / "events.jsonl"

    atomic_write_json(document, {"结论": "通过"})
    append_jsonl(events, {"event": "completed"})

    assert json.loads(document.read_text(encoding="utf-8")) == {"结论": "通过"}
    assert events.read_text(encoding="utf-8").endswith("\n")
