"""验证 VPT 滚动下载以完整文件对原子发布。"""

import json
from pathlib import Path
import time

from datasets.vpt.rolling_download import RollingVPTDownloadManager


def test_rolling_downloader_publishes_local_file_urls(tmp_path: Path):
    """本地 file URL 模拟远端，训练目录最终只看到完整 mp4/jsonl 对。"""
    source = tmp_path / "source"
    source.mkdir()
    video = source / "sample.mp4"
    actions = source / "sample.jsonl"
    video.write_bytes(b"video")
    actions.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{
        "name": "clip_a",
        "video_url": video.as_uri(),
        "action_url": actions.as_uri(),
    }]), encoding="utf-8")
    destination = tmp_path / "stream"
    manager = RollingVPTDownloadManager(
        str(manifest), destination, maximum_cache_bytes=1024,
        minimum_ready_pairs=1,
    ).start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            manager.check()
            if (destination / "clip_a.mp4").exists() and (
                destination / "clip_a.jsonl"
            ).exists():
                break
            time.sleep(0.01)
        assert (destination / "clip_a.mp4").read_bytes() == b"video"
        assert (destination / "clip_a.jsonl").read_text(encoding="utf-8") == "{}\n"
        assert not list(destination.glob("*.part"))
    finally:
        manager.stop()
