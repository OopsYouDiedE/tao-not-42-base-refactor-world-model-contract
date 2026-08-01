from pathlib import Path

from dataset.extraction.minestudio import download_and_read_minestudio_lmdb_dataset as minestudio


def test_load_keeps_only_requested_dataset_and_modalities(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "minestudio-data-7xx-v110"
    (target / "segmentation").mkdir(parents=True)
    (tmp_path / "minestudio-data-8xx-v110").mkdir()
    captured = {}
    monkeypatch.setattr(minestudio, "snapshot_download", lambda **kwargs: captured.update(kwargs))

    dataset = minestudio.load(
        ["7xx"],
        ["action", "image"],
        force_remove_other_dataset_in_this_group=True,
        output=tmp_path,
    )

    assert dataset.root == target
    assert captured["allow_patterns"] == ["action/*", "image/*"]
    assert not (target / "segmentation").exists()
    assert not (tmp_path / "minestudio-data-8xx-v110").exists()
