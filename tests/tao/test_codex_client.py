from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tao.baselines.codex.client import (
    CodexClient,
    CodexClientConfig,
    CodexInvocationError,
)

FAKE_CODEX = r'''from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

arguments = sys.argv[1:]
prompt = sys.stdin.read()
mode = os.environ.get("TAO_FAKE_CODEX_MODE", "success")
if mode == "timeout":
    time.sleep(2)
if mode == "nonzero":
    print("controlled failure", file=sys.stderr)
    raise SystemExit(7)
output = Path(arguments[arguments.index("--output-last-message") + 1])
if mode == "invalid_json":
    output.write_text("not-json", encoding="utf-8")
else:
    output.write_text(json.dumps({"ok": True, "prompt": prompt}), encoding="utf-8")
log_path = os.environ.get("TAO_FAKE_CODEX_LOG")
if log_path:
    Path(log_path).write_text(json.dumps(arguments), encoding="utf-8")
'''


def _fake_client(tmp_path: Path, *, timeout: float = 2.0) -> CodexClient:
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX, encoding="utf-8")
    return CodexClient(
        CodexClientConfig(
            model="test-codex-model",
            executable=sys.executable,
            executable_args=(str(script),),
            timeout_seconds=timeout,
            max_attempts=1,
            retry_delay_seconds=0,
            temporary_root=tmp_path,
        )
    )


def test_codex_client_uses_isolated_structured_noninteractive_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "observation.png"
    image.write_bytes(b"image")
    log = tmp_path / "arguments.json"
    monkeypatch.setenv("TAO_FAKE_CODEX_LOG", str(log))

    invocation = _fake_client(tmp_path).run_structured(
        "structured prompt",
        {"type": "object"},
        images=(image,),
    )

    assert invocation.result == {"ok": True, "prompt": "structured prompt"}
    arguments = json.loads(log.read_text(encoding="utf-8"))
    assert arguments[:4] == ["--ask-for-approval", "never", "exec", "-"]
    for flag in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--output-schema",
        "--output-last-message",
        "--json",
        "--image",
    ):
        assert flag in arguments
    assert "resume" not in arguments
    assert arguments[arguments.index("--sandbox") + 1] == "read-only"
    assert arguments[arguments.index("--model") + 1] == "test-codex-model"


@pytest.mark.parametrize("mode", ["nonzero", "invalid_json"])
def test_codex_client_rejects_failed_or_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("TAO_FAKE_CODEX_MODE", mode)
    with pytest.raises(CodexInvocationError):
        _fake_client(tmp_path).run_structured("prompt", {"type": "object"})


def test_codex_client_terminates_timed_out_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAO_FAKE_CODEX_MODE", "timeout")
    with pytest.raises(CodexInvocationError, match="timed out"):
        _fake_client(tmp_path, timeout=0.05).run_structured("prompt", {"type": "object"})


def test_codex_client_passes_explicit_provider_without_exposing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "arguments.json"
    monkeypatch.setenv("TAO_FAKE_CODEX_LOG", str(log))
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX, encoding="utf-8")
    client = CodexClient(
        CodexClientConfig(
            model="teacher-model",
            executable=sys.executable,
            executable_args=(str(script),),
            api_url="https://proxy.example/v1",
            api_key="secret-value",
            temporary_root=tmp_path,
        )
    )

    client.run_structured("prompt", {"type": "object"})

    arguments = json.loads(log.read_text(encoding="utf-8"))
    assert 'model_provider="tao_teacher"' in arguments
    assert 'model_providers.tao_teacher.base_url="https://proxy.example/v1"' in arguments
    assert "secret-value" not in arguments
