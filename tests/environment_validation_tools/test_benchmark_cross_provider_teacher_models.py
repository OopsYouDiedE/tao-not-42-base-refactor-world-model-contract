from pathlib import Path

from environment_validation_tools import benchmark_cross_provider_teacher_models as benchmark
from online_environment_interaction_agents import TeacherResponse


class FakeBackend:
    def __init__(self, config) -> None:
        self.model = config.model

    def stream(self, request, on_chunk):
        text = "Device KeyboardMouse\nTick 0\n<action>W x2 ; Observe W ; W x4</action>"
        on_chunk(text)
        return TeacherResponse(text, "fake", self.model, "id", 1, 1, 12.0)


def test_cross_provider_benchmark_writes_reports(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "start.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(benchmark, "ClaudeCLIBackend", FakeBackend)
    monkeypatch.setattr(benchmark, "CodexCLIBackend", FakeBackend)

    report = benchmark.run(tmp_path / "output", image=image)

    text = report.read_text(encoding="utf-8")
    assert "claude-opus-5" in text
    assert "claude-sonnet-5" in text
    assert "gpt-5.6-terra" in text
    assert (tmp_path / "output" / "result.json").is_file()
