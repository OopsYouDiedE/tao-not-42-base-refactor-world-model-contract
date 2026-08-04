"""检查环境并把机器和人工可读报告写入 runs。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_tools.artifacts import atomic_write_json, atomic_write_text  # noqa: E402
from shared_tools.authentication import (  # noqa: E402
    check_github_authentication,
    check_huggingface_authentication,
)
from shared_tools.environment_checks import check_environment, detect_accelerator  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="检查项目运行环境")
    parser.add_argument("--accelerator", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-github-auth", action="store_true")
    parser.add_argument("--skip-huggingface-auth", action="store_true")
    parser.add_argument("--skip-optional-auth", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    accelerator = detect_accelerator() if arguments.accelerator == "auto" else arguments.accelerator
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = arguments.output or Path("runs/environment_checks") / timestamp
    checks = check_environment(accelerator)
    github = check_github_authentication(
        skip=arguments.skip_optional_auth or arguments.skip_github_auth
    )
    huggingface = check_huggingface_authentication(
        skip=arguments.skip_optional_auth or arguments.skip_huggingface_auth
    )
    payload = {
        "accelerator": accelerator,
        "checks": [asdict(result) for result in checks],
        "authentication": {
            "github": asdict(github),
            "huggingface": asdict(huggingface),
        },
    }
    atomic_write_json(output / "environment-report.json", payload)
    lines = [
        "# 环境检查",
        "",
        f"计算后端：`{accelerator}`",
        "",
        "| 检查 | 状态 | 结论 |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| `{item.check_id}` | `{item.status}` | {item.summary} |" for item in checks)
    lines.extend(
        (
            f"| `auth:github` | `{github.status}` | {github.remediation or '已完成'} |",
            f"| `auth:huggingface` | `{huggingface.status}` | {huggingface.remediation or '已完成'} |",
        )
    )
    atomic_write_text(output / "REPORT.md", "\n".join(lines) + "\n")
    print(output)
    return 1 if any(result.status == "failed" for result in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
