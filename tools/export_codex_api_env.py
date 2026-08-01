"""从全局 Codex 配置生成当前 shell 可 source 的教师 API export。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Any


def find_codex_home(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.getenv("TAO_CODEX_HOME") or os.getenv("CODEX_HOME")
    if configured:
        candidate = Path(configured).expanduser()
        if (candidate / "config.toml").is_file():
            return candidate.resolve()
    native = Path.home() / ".codex"
    if (native / "config.toml").is_file():
        return native.resolve()
    if sys.platform.startswith("linux") and Path("/mnt/c/Users").is_dir():
        candidates = sorted(
            path
            for path in Path("/mnt/c/Users").glob("*/.codex")
            if (path / "config.toml").is_file() and (path / "auth.json").is_file()
        )
        if len(candidates) == 1:
            return candidates[0].resolve()
        if len(candidates) > 1:
            raise RuntimeError("检测到多个 Windows .codex，请通过 --codex-home 显式指定")
    raise FileNotFoundError("找不到全局 .codex；请通过 --codex-home 显式指定")


def load_exports(codex_home: Path) -> dict[str, str]:
    config_path = codex_home / "config.toml"
    auth_path = codex_home / "auth.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少 Codex 配置：{config_path}")
    if not auth_path.is_file():
        raise FileNotFoundError(f"缺少 Codex 认证：{auth_path}")

    config: dict[str, Any] = tomllib.loads(config_path.read_text(encoding="utf-8"))
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    model = str(config.get("model", "")).strip()
    provider_name = str(config.get("model_provider", "openai"))
    provider = config.get("model_providers", {}).get(provider_name, {})
    api_url = str(provider.get("base_url") or config.get("openai_base_url") or "").strip()
    env_key = str(provider.get("env_key", "")).strip()
    api_key = str(os.getenv(env_key, "") if env_key else auth.get("OPENAI_API_KEY", "")).strip()

    missing = [
        name
        for name, value in (("model", model), ("provider base_url", api_url), ("API key", api_key))
        if not value
    ]
    if missing:
        raise RuntimeError("全局 Codex 配置缺少：" + ", ".join(missing))
    return {"API_KEY": api_key, "API_MODEL": model, "API_URL": api_url}


def shell_exports(values: dict[str, str]) -> str:
    return "\n".join(f"export {name}={shlex.quote(value)}" for name, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="将全局 .codex 参数导出为教师模型 API 环境变量")
    parser.add_argument("--codex-home", type=Path)
    parser.add_argument("--check", action="store_true", help="只输出脱敏配置，不输出 shell export")
    arguments = parser.parse_args()
    codex_home = find_codex_home(arguments.codex_home)
    values = load_exports(codex_home)
    if arguments.check:
        print(f"CODEX_HOME={codex_home}")
        print(f"API_MODEL={values['API_MODEL']}")
        print(f"API_URL={values['API_URL']}")
        print("API_KEY=<redacted>")
        return
    print(shell_exports(values))


if __name__ == "__main__":
    main()
