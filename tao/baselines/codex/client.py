"""可审计、非交互的 Codex CLI 结构化调用。"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CodexInvocationError(RuntimeError):
    """Codex CLI 在有限重试后仍未返回合法结构化结果。"""


@dataclass(frozen=True)
class CodexClientConfig:
    model: str
    executable: str = "codex"
    executable_args: tuple[str, ...] = ()
    api_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 240.0
    max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    temporary_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Codex model 必须显式指定")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于零")
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须至少为 1")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds 不能为负")
        if (self.api_url is None) != (self.api_key is None):
            raise ValueError("api_url 和 api_key 必须同时提供")


@dataclass(frozen=True)
class CodexInvocation:
    result: dict[str, Any]
    model: str
    attempts: int
    wall_ms: float
    image_count: int
    isolated_session: bool = True
    sandbox: str = "read-only"

    def audit_dict(self) -> dict[str, Any]:
        return {
            "provider": "codex-cli",
            "model": self.model,
            "attempts": self.attempts,
            "wall_ms": round(self.wall_ms, 3),
            "image_count": self.image_count,
            "isolated_session": self.isolated_session,
            "sandbox": self.sandbox,
        }


class CodexClient:
    """每次请求都创建独立 ephemeral 会话，并仅接收 JSON Schema 输出。"""

    def __init__(self, config: CodexClientConfig):
        self.config = config

    def run_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        images: tuple[Path, ...] = (),
    ) -> CodexInvocation:
        executable = shutil.which(self.config.executable) or (
            self.config.executable if Path(self.config.executable).is_file() else None
        )
        if executable is None:
            raise CodexInvocationError(f"找不到 Codex CLI：{self.config.executable}")
        resolved_images = tuple(path.resolve() for path in images)
        missing = [str(path) for path in resolved_images if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Codex 输入图片不存在：{missing}")

        failures: list[str] = []
        overall_start = time.perf_counter()
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = self._attempt(executable, prompt, schema, resolved_images)
                return CodexInvocation(
                    result=result,
                    model=self.config.model,
                    attempts=attempt,
                    wall_ms=(time.perf_counter() - overall_start) * 1000.0,
                    image_count=len(resolved_images),
                )
            except (CodexInvocationError, subprocess.TimeoutExpired) as error:
                failures.append(f"attempt {attempt}: {error}")
                if attempt < self.config.max_attempts and self.config.retry_delay_seconds:
                    time.sleep(self.config.retry_delay_seconds)
        raise CodexInvocationError("Codex CLI 调用失败；" + "；".join(failures))

    def _attempt(
        self,
        executable: str,
        prompt: str,
        schema: dict[str, Any],
        images: tuple[Path, ...],
    ) -> dict[str, Any]:
        root = None if self.config.temporary_root is None else str(self.config.temporary_root)
        with tempfile.TemporaryDirectory(prefix="tao-codex-", dir=root) as directory_name:
            directory = Path(directory_name)
            schema_path = directory / "schema.json"
            output_path = directory / "result.json"
            schema_path.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            command = [
                executable,
                *self.config.executable_args,
                *self._provider_arguments(),
                "--ask-for-approval",
                "never",
                "exec",
                "-",
                "--cd",
                str(directory),
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--model",
                self.config.model,
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "--color",
                "never",
            ]
            for image in images:
                command.extend(("--image", str(image)))

            popen_options: dict[str, Any] = {
                "cwd": directory,
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "shell": False,
            }
            if self.config.api_key is not None:
                popen_options["env"] = {
                    **os.environ,
                    "OPENAI_API_KEY": self.config.api_key,
                }
            if os.name == "nt":
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_options["start_new_session"] = True
            process = subprocess.Popen(command, **popen_options)
            try:
                stdout, stderr = process.communicate(prompt, timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                process.communicate()
                raise
            if process.returncode != 0:
                diagnostic = (stderr or stdout).strip().replace("\n", " ")[-1000:]
                raise CodexInvocationError(
                    f"退出码 {process.returncode}" + (f"：{diagnostic}" if diagnostic else "")
                )
            if not output_path.is_file():
                raise CodexInvocationError("未生成 --output-last-message 文件")
            try:
                value = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise CodexInvocationError("结构化输出不是合法 JSON") from error
            if not isinstance(value, dict):
                raise CodexInvocationError("结构化输出根节点必须是对象")
            return value

    def _provider_arguments(self) -> list[str]:
        if self.config.api_url is None:
            return []
        return [
            "-c",
            'model_provider="tao_teacher"',
            "-c",
            'model_providers.tao_teacher.name="tao_teacher"',
            "-c",
            f'model_providers.tao_teacher.base_url={json.dumps(self.config.api_url)}',
            "-c",
            'model_providers.tao_teacher.wire_api="responses"',
            "-c",
            "model_providers.tao_teacher.requires_openai_auth=true",
        ]


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.kill()
