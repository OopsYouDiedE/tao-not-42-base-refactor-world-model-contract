"""从 HTTP 清单持续下载 VPT 视频对并维持有界磁盘缓存。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class VPTDownloadEntry:
    """一个可原子发布的 ``mp4 + jsonl`` 下载项。"""

    name: str
    video_url: str
    action_url: str
    video_sha256: str | None = None
    action_sha256: str | None = None


def load_download_manifest(location: str) -> list[VPTDownloadEntry]:
    """从本地路径或 HTTP(S) URL 读取 JSON/JSONL 清单。"""
    if location.startswith(("http://", "https://")):
        request = Request(location, headers=_authorization_headers())
        with urlopen(request, timeout=60) as response:
            content = response.read().decode("utf-8")
    else:
        content = Path(location).read_text(encoding="utf-8")
    stripped = content.lstrip()
    records = json.loads(content) if stripped.startswith("[") else [
        json.loads(line) for line in content.splitlines() if line.strip()
    ]
    entries = []
    for index, record in enumerate(records):
        video_url = record["video_url"]
        action_url = record["action_url"]
        inferred = Path(urlparse(video_url).path).stem or f"clip_{index:08d}"
        name = str(record.get("name") or inferred)
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"非法清单 name: {name!r}")
        entries.append(VPTDownloadEntry(
            name=name,
            video_url=video_url,
            action_url=action_url,
            video_sha256=record.get("video_sha256"),
            action_sha256=record.get("action_sha256"),
        ))
    if not entries:
        raise ValueError("下载清单不能为空")
    return entries


def _authorization_headers() -> dict[str, str]:
    """从环境读取 Hugging Face token，但不记录或持久化它。"""
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


class RollingVPTDownloadManager:
    """后台轮换下载清单项，并把训练目录限制在指定容量内。"""

    def __init__(
        self,
        manifest: str,
        data_directory: str | Path,
        maximum_cache_bytes: int,
        minimum_ready_pairs: int = 4,
        seed: int = 0,
    ):
        if maximum_cache_bytes <= 0:
            raise ValueError("maximum_cache_bytes 必须大于零")
        if minimum_ready_pairs < 1:
            raise ValueError("minimum_ready_pairs 必须大于零")
        self.entries = load_download_manifest(manifest)
        self.data_directory = Path(data_directory).resolve()
        self.maximum_cache_bytes = maximum_cache_bytes
        self.minimum_ready_pairs = min(minimum_ready_pairs, len(self.entries))
        self.random = random.Random(seed)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.failure: BaseException | None = None

    def start(self) -> "RollingVPTDownloadManager":
        """启动单一后台下载线程。"""
        if self.thread is not None:
            raise RuntimeError("下载线程已经启动")
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        """请求停止；已经完整发布的数据对继续保留以便续训。"""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)

    def check(self) -> None:
        """在训练主线程中传播不可恢复的下载错误。"""
        if self.failure is not None:
            raise RuntimeError("VPT 滚动下载线程失败") from self.failure

    def __enter__(self) -> "RollingVPTDownloadManager":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _ready_pairs(self) -> list[tuple[Path, Path]]:
        pairs = []
        for video in self.data_directory.glob("*.mp4"):
            actions = video.with_suffix(".jsonl")
            if actions.exists():
                pairs.append((video, actions))
        return pairs

    def _cache_bytes(self, pairs: list[tuple[Path, Path]]) -> int:
        return sum(path.stat().st_size for pair in pairs for path in pair if path.exists())

    def _evict(self) -> None:
        pairs = sorted(
            self._ready_pairs(),
            key=lambda pair: min(path.stat().st_mtime for path in pair),
        )
        while (
            len(pairs) > self.minimum_ready_pairs
            and self._cache_bytes(pairs) > self.maximum_cache_bytes
        ):
            video, actions = pairs.pop(0)
            try:
                video.unlink()
                actions.unlink()
            except FileNotFoundError:
                continue

    def _download(self, url: str, destination: Path, expected_sha256: str | None) -> None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        request = Request(url, headers=_authorization_headers())
        try:
            with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while not self.stop_event.is_set():
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
            if self.stop_event.is_set():
                temporary.unlink(missing_ok=True)
                return
            if expected_sha256 and digest.hexdigest().lower() != expected_sha256.lower():
                raise ValueError(f"{destination.name} SHA256 不匹配")
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _publish(self, entry: VPTDownloadEntry) -> bool:
        """发布一个文件对；实际发生下载时返回 True。"""
        video = self.data_directory / f"{entry.name}.mp4"
        actions = self.data_directory / f"{entry.name}.jsonl"
        if video.exists() and actions.exists():
            now = time.time()
            os.utime(video, (now, now))
            os.utime(actions, (now, now))
            return False
        video.unlink(missing_ok=True)
        actions.unlink(missing_ok=True)
        self._download(entry.video_url, video, entry.video_sha256)
        try:
            self._download(entry.action_url, actions, entry.action_sha256)
        except BaseException:
            video.unlink(missing_ok=True)
            raise
        return True

    def _run(self) -> None:
        entries = self.entries.copy()
        self.random.shuffle(entries)
        index = 0
        consecutive_failures = 0
        try:
            while not self.stop_event.is_set():
                entry = entries[index]
                index = (index + 1) % len(entries)
                try:
                    downloaded = self._publish(entry)
                    self._evict()
                    consecutive_failures = 0
                    if not downloaded and len(self._ready_pairs()) >= self.minimum_ready_pairs:
                        self.stop_event.wait(1.0)
                except (OSError, ValueError) as error:
                    consecutive_failures += 1
                    if consecutive_failures >= max(8, len(entries) * 2):
                        raise RuntimeError("清单中的数据连续下载失败") from error
                    delay = min(30.0, float(2 ** min(consecutive_failures, 4)))
                    self.stop_event.wait(delay)
        except BaseException as error:
            self.failure = error
