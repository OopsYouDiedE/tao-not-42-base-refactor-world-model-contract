"""教师模型驱动的观察到动作轨迹生成。"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from online_interactive_environments import extract_action_sequence_text
from shared_tools.configuration import require_env

from .model_contracts import ModelResponse


class TeacherModelError(RuntimeError):
    """教师模型调用或输出校验失败。"""


@dataclass(frozen=True)
class TeacherDecisionEnvelope:
    control: str
    non_control_text: str


def parse_teacher_decision(text: str) -> TeacherDecisionEnvelope:
    """提取唯一动作控制块，并保留未执行的响应外壳用于审计。"""
    normalized = text.strip()
    if not normalized:
        raise ValueError("教师动作不能为空")
    control = extract_action_sequence_text(normalized)
    start = normalized.index(control)
    non_control = (normalized[:start] + normalized[start + len(control) :]).strip()
    return TeacherDecisionEnvelope(control, non_control)


@dataclass(frozen=True)
class TeacherRequest:
    system_prompt: str
    task_context: str
    step_context: str
    observation_paths: tuple[Path, ...] = ()


TeacherResponse = ModelResponse


class TeacherBackend(Protocol):
    provider: str
    model: str

    def generate(self, request: TeacherRequest) -> TeacherResponse: ...


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 180.0
    max_output_tokens: int = 1024
    wire_api: str = "chat_completions"

    @classmethod
    def from_environment(cls) -> OpenAICompatibleConfig:
        """从环境变量装配配置。

        除 `TEACHER_API_URL`、`TEACHER_API_KEY` 和 `TEACHER_MODEL` 外，同时读取可选的
        `TEACHER_WIRE_API` 与 `TEACHER_TIMEOUT_SECONDS`。缺少 wire 选择时保持
        `chat_completions` 默认值；只支持 `responses` 协议的模型必须显式设置
        `TEACHER_WIRE_API=responses`，否则服务端会以 `protocol_not_supported` 拒绝请求。
        """
        return cls(
            base_url=require_env("TEACHER_API_URL"),
            api_key=require_env("TEACHER_API_KEY"),
            model=require_env("TEACHER_MODEL"),
            timeout_seconds=float(os.getenv("TEACHER_TIMEOUT_SECONDS", "180")),
            wire_api=os.getenv("TEACHER_WIRE_API", "chat_completions"),
        )


class OpenAICompatibleBackend:
    """使用 OpenAI Chat Completions 兼容接口生成动作。"""

    provider = "openai-compatible"

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        if not config.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url 必须是 HTTP(S) URL")
        if not config.api_key or not config.model:
            raise ValueError("api_key 和 model 不能为空")
        self.config = config
        self.model = config.model

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        return self.stream(request, lambda chunk: None)

    def stream(
        self,
        request: TeacherRequest,
        on_chunk: Callable[[str], object],
    ) -> TeacherResponse:
        if self.config.wire_api == "responses":
            return self._stream_responses(request, on_chunk)
        content: list[dict[str, Any]] = [{"type": "text", "text": _combined_user_prompt(request)}]
        for path in _resolved_observations(request.observation_paths):
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": self.config.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        http_request = urllib.request.Request(
            _chat_completions_url(self.config.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "tao-teacher-trajectory/1.0",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.config.timeout_seconds
            ) as response:
                chunks: list[str] = []
                non_sse_lines: list[str] = []
                request_id = None
                usage: dict[str, Any] = {}
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        if line:
                            non_sse_lines.append(line)
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    request_id = request_id or event.get("id")
                    usage = event.get("usage") or usage
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}).get("content")
                        if isinstance(delta, str) and delta:
                            chunks.append(delta)
                            on_chunk(delta)
                if not chunks and non_sse_lines:
                    candidate = "".join(non_sse_lines)
                    if not candidate:
                        raise TeacherModelError(
                            "OpenAI Chat 流没有文本；非数据事件："
                            + ", ".join(line[:80] for line in non_sse_lines[:10])
                        )
                    try:
                        document = json.loads(candidate)
                    except json.JSONDecodeError as error:
                        raise TeacherModelError(
                            f"OpenAI Chat 非标准响应：{candidate[:500]!r}"
                        ) from error
                    if document.get("error"):
                        raise TeacherModelError(
                            f"OpenAI Chat 兼容接口返回错误：{document['error']}"
                        )
                    request_id = request_id or document.get("id")
                    usage = document.get("usage") or usage
                    choices = document.get("choices") or []
                    if choices:
                        content = choices[0].get("message", {}).get("content")
                        if isinstance(content, str) and content:
                            chunks.append(content)
                            on_chunk(content)
        except urllib.error.HTTPError as error:
            diagnostic = error.read().decode("utf-8", errors="replace")[-1000:]
            raise TeacherModelError(
                f"OpenAI Chat 兼容接口返回 HTTP {error.code}：{diagnostic}"
            ) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise TeacherModelError(f"OpenAI 兼容接口调用失败：{error}") from error
        elapsed_ms = (time.perf_counter() - started) * 1000
        text = "".join(chunks)
        if not text:
            raise TeacherModelError("OpenAI 兼容接口流没有消息文本")
        return TeacherResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            request_id=request_id,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            elapsed_ms=elapsed_ms,
        )

    def _stream_responses(
        self,
        request: TeacherRequest,
        on_chunk: Callable[[str], object],
    ) -> TeacherResponse:
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": _combined_user_prompt(request)}
        ]
        for path in _resolved_observations(request.observation_paths):
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {"type": "input_image", "image_url": f"data:{media_type};base64,{encoded}"}
            )
        payload = {
            "model": self.model,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self.config.max_output_tokens,
            "stream": True,
        }
        http_request = urllib.request.Request(
            _responses_url(self.config.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "codex-cli/0.146.0",
            },
            method="POST",
        )
        chunks: list[str] = []
        completed_response: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.config.timeout_seconds
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("type") == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            chunks.append(delta)
                            on_chunk(delta)
                    elif event.get("type") == "response.completed":
                        completed_response = event.get("response", {})
        except urllib.error.HTTPError as error:
            diagnostic = error.read().decode("utf-8", errors="replace")[-1000:]
            raise TeacherModelError(
                f"OpenAI Responses 兼容接口返回 HTTP {error.code}：{diagnostic}"
            ) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise TeacherModelError(f"OpenAI Responses 兼容接口调用失败：{error}") from error
        text = "".join(chunks)
        if not text:
            raise TeacherModelError("OpenAI Responses 流没有消息文本")
        usage = completed_response.get("usage", {})
        return TeacherResponse(
            text,
            self.provider,
            self.model,
            completed_response.get("id"),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            (time.perf_counter() - started) * 1000,
        )


@dataclass(frozen=True)
class AnthropicCompatibleConfig:
    base_url: str
    auth_token: str
    model: str
    timeout_seconds: float = 180.0
    max_output_tokens: int = 1024


class AnthropicCompatibleBackend:
    """使用 Anthropic Messages 兼容 SSE 接口生成动作。"""

    provider = "anthropic-compatible"

    def __init__(self, config: AnthropicCompatibleConfig) -> None:
        self.config = config
        self.model = config.model

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        return self.stream(request, lambda chunk: None)

    def stream(
        self,
        request: TeacherRequest,
        on_chunk: Callable[[str], object],
    ) -> TeacherResponse:
        content: list[dict[str, Any]] = []
        for path in _resolved_observations(request.observation_paths):
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": _combined_user_prompt(request)})
        payload = {
            "model": self.model,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_output_tokens,
            "temperature": 0,
            "stream": True,
        }
        http_request = urllib.request.Request(
            _anthropic_messages_url(self.config.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.auth_token}",
                "x-api-key": self.config.auth_token,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "tao-teacher-trajectory/1.0",
            },
            method="POST",
        )
        chunks: list[str] = []
        request_id = None
        input_tokens = None
        output_tokens = None
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.config.timeout_seconds
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    event_type = event.get("type")
                    if event_type == "message_start":
                        message = event.get("message", {})
                        request_id = message.get("id")
                        input_tokens = message.get("usage", {}).get("input_tokens")
                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            chunks.append(text)
                            on_chunk(text)
                    elif event_type == "message_delta":
                        output_tokens = event.get("usage", {}).get("output_tokens")
        except urllib.error.HTTPError as error:
            diagnostic = error.read().decode("utf-8", errors="replace")[-1000:]
            raise TeacherModelError(
                f"Anthropic 兼容接口返回 HTTP {error.code}：{diagnostic}"
            ) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise TeacherModelError(f"Anthropic 兼容接口调用失败：{error}") from error
        text = "".join(chunks)
        if not text:
            raise TeacherModelError("Anthropic 兼容接口流没有消息文本")
        return TeacherResponse(
            text,
            self.provider,
            self.model,
            request_id,
            input_tokens,
            output_tokens,
            (time.perf_counter() - started) * 1000,
        )


@dataclass(frozen=True)
class CLIConfig:
    model: str
    executable: str
    timeout_seconds: float = 180.0
    extra_arguments: tuple[str, ...] = ()
    command_arguments: tuple[str, ...] = ()


class CodexCLIBackend:
    provider = "codex-cli"

    def __init__(self, config: CLIConfig) -> None:
        self.config = config
        self.model = config.model

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        executable = _find_executable(self.config.executable)
        observations = _resolved_observations(request.observation_paths)
        with tempfile.TemporaryDirectory(prefix="teacher-codex-") as directory:
            output_path = Path(directory) / "last-message.txt"
            command = [
                executable,
                *self.config.extra_arguments,
                "exec",
                *self.config.command_arguments,
                "-",
                "--cd",
                directory,
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--model",
                self.model,
                "--output-last-message",
                str(output_path),
                "--color",
                "never",
            ]
            for path in observations:
                command.extend(("--image", str(path)))
            started = time.perf_counter()
            _run_cli(
                command,
                request.system_prompt + "\n\n" + _combined_user_prompt(request),
                self.config.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            if not output_path.is_file():
                raise TeacherModelError("Codex CLI 未生成最终消息文件")
            text = output_path.read_text(encoding="utf-8")
        return TeacherResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            request_id=None,
            input_tokens=None,
            output_tokens=None,
            elapsed_ms=elapsed_ms,
        )

    def stream(
        self,
        request: TeacherRequest,
        on_chunk: Callable[[str], object],
    ) -> TeacherResponse:
        executable = _find_executable(self.config.executable)
        observations = _resolved_observations(request.observation_paths)
        with tempfile.TemporaryDirectory(prefix="teacher-codex-stream-") as directory:
            command = [
                executable,
                *self.config.extra_arguments,
                "exec",
                *self.config.command_arguments,
                "-",
                "--cd",
                directory,
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--model",
                self.model,
                "--json",
                "--color",
                "never",
            ]
            for path in observations:
                command.extend(("--image", str(path)))
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            if process.stdin is None or process.stdout is None:
                raise TeacherModelError("Codex CLI 流管道创建失败")
            prompt = request.system_prompt + "\n\n" + _combined_user_prompt(request)
            started = time.perf_counter()
            process.stdin.write(prompt)
            process.stdin.close()
            text = ""
            request_id = None
            usage: dict[str, Any] = {}
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "thread.started":
                    request_id = event.get("thread_id")
                elif event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        chunk = item.get("text", "")
                        if chunk:
                            text += chunk
                            on_chunk(chunk)
                elif event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
            process.wait(timeout=self.config.timeout_seconds)
            if process.returncode != 0:
                diagnostic = "" if process.stderr is None else process.stderr.read().strip()[-1000:]
                raise TeacherModelError(
                    f"Codex CLI 退出码 {process.returncode}"
                    + (f"：{diagnostic}" if diagnostic else "")
                )
        if not text:
            raise TeacherModelError("Codex CLI JSON 流没有 agent_message")
        return TeacherResponse(
            text,
            self.provider,
            self.model,
            request_id,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            (time.perf_counter() - started) * 1000,
        )


class ClaudeCLIBackend:
    provider = "claude-cli"

    def __init__(self, config: CLIConfig) -> None:
        self.config = config
        self.model = config.model

    def generate(self, request: TeacherRequest) -> TeacherResponse:
        return self.stream(request, lambda chunk: None)

    def stream(
        self,
        request: TeacherRequest,
        on_chunk: Callable[[str], object],
    ) -> TeacherResponse:
        executable = _find_executable(self.config.executable)
        image_instruction = "\n".join(
            f"观察图片 {index} 的绝对路径：{_claude_path(path, executable)}"
            for index, path in enumerate(_resolved_observations(request.observation_paths), start=1)
        )
        prompt = _combined_user_prompt(request)
        if image_instruction:
            prompt += "\n\n" + image_instruction
        command = [
            executable,
            *self.config.extra_arguments,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model",
            self.model,
            "--system-prompt",
            request.system_prompt,
            "--tools",
            "Read",
            "--dangerously-skip-permissions",
        ]
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        if process.stdin is None or process.stdout is None:
            raise TeacherModelError("Claude CLI 流管道创建失败")
        process.stdin.write(prompt)
        process.stdin.close()
        chunks: list[str] = []
        result_event: dict[str, Any] = {}
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "stream_event":
                    nested = event.get("event", {})
                    delta = nested.get("delta", {})
                    text = delta.get("text") if delta.get("type") == "text_delta" else None
                    if isinstance(text, str) and text:
                        chunks.append(text)
                        on_chunk(text)
                elif event.get("type") == "result":
                    result_event = event
            process.wait(timeout=self.config.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            raise TeacherModelError(
                f"Claude CLI 调用超过 {self.config.timeout_seconds} 秒"
            ) from error
        if process.returncode != 0:
            diagnostic = "" if process.stderr is None else process.stderr.read().strip()[-1000:]
            raise TeacherModelError(
                f"Claude CLI 退出码 {process.returncode}"
                + (f"：{diagnostic}" if diagnostic else "")
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        text = "".join(chunks) or result_event.get("result", "")
        if not isinstance(text, str) or not text:
            raise TeacherModelError("Claude CLI 流没有消息文本")
        usage = result_event.get("usage", {})
        return TeacherResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            request_id=result_event.get("session_id"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            elapsed_ms=elapsed_ms,
        )


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _responses_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    return normalized + "/responses"


def _anthropic_messages_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/messages"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/messages"
    return normalized + "/v1/messages"


def _combined_user_prompt(request: TeacherRequest) -> str:
    return request.task_context.strip() + "\n\n" + request.step_context.strip()


def _resolved_observations(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved = tuple(path.resolve() for path in paths)
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"观察图片不存在：{missing}")
    return resolved


def _find_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise TeacherModelError(f"找不到 CLI 可执行文件：{name}")
    return executable


def _claude_path(path: Path, executable: str) -> str:
    if os.name != "nt" and executable.lower().endswith(".exe"):
        resolved = path.resolve().as_posix()
        if resolved.startswith("/mnt/") and len(resolved) > 6:
            drive = resolved[5].upper()
            return drive + ":\\" + resolved[7:].replace("/", "\\")
    return str(path.resolve())


def _run_cli(
    command: list[str], prompt: str, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TeacherModelError(f"CLI 调用超过 {timeout_seconds} 秒") from error
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()[-1000:]
        raise TeacherModelError(
            f"CLI 退出码 {completed.returncode}" + (f"：{diagnostic}" if diagnostic else "")
        )
    return completed
