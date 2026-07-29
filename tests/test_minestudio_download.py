"""部分图像编码下载不会带入 segmentation。"""

from pathlib import Path

from datasets.minestudio_data import download


def test_partial_image_download_patterns(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(download, "list_repo_files", lambda *args, **kwargs: [
        "image/part-20/data.mdb", "image/part-10/data.mdb", "segmentation/part-10/data.mdb",
    ])
    monkeypatch.setattr(
        download, "snapshot_download", lambda **kwargs: captured.update(kwargs),
    )
    download.download_datasets(
        ["7xx"], ["action", "meta_info", "image"], tmp_path, maximum_image_parts=1,
    )
    assert captured["allow_patterns"] == [
        "action/*", "meta_info/*", "image/part-10/*",
    ]
