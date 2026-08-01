from __future__ import annotations

from pathlib import Path

from tools.prepare_craftground_runtime import DISPATCH, prepare


class FakeDistribution:
    def __init__(self, root: Path):
        self.root = root

    def locate_file(self, path: str) -> Path:
        return self.root / path


def test_prepare_runtime_copies_and_patches_idempotently(
    tmp_path: Path, monkeypatch
) -> None:
    site_packages = tmp_path / "site-packages"
    package = site_packages / "craftground_runtime_mc121"
    java = package / "src/main/java/com/kyhsgeekcode/minecraftenv"
    java.mkdir(parents=True)
    (java / "MinecraftEnv.kt").write_text(
        "class MinecraftEnv {\n"
        "    private fun handleCommand(\n"
        "        command: String,\n"
        "    ): Boolean {\n"
        "        return false\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    patch = tmp_path / "MemorySnapshotStore.kt"
    patch.write_text("object MemorySnapshotStore\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.prepare_craftground_runtime.importlib.metadata.distribution",
        lambda name: FakeDistribution(site_packages),
    )
    target = tmp_path / "runtime"

    prepare(patch, target, build=False)
    prepare(patch, target, build=False)

    content = (target / java.relative_to(package) / "MinecraftEnv.kt").read_text(
        encoding="utf-8"
    )
    assert content.count(DISPATCH) == 1
    assert (target / java.relative_to(package) / patch.name).is_file()
