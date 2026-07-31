"""监控 GPU 空闲状态并留下可机读告警。"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def gpu_sample() -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    utilization, memory = output.split(",", maxsplit=1)
    return int(utilization.strip()), int(memory.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--idle-samples", type=int, default=2)
    arguments = parser.parse_args()
    idle_count = 0
    while True:
        utilization, memory = gpu_sample()
        idle_count = idle_count + 1 if utilization == 0 else 0
        payload = {
            "timestamp": time.time(),
            "utilization_percent": utilization,
            "memory_mib": memory,
            "consecutive_idle_samples": idle_count,
            "alert": idle_count >= arguments.idle_samples,
        }
        arguments.status.parent.mkdir(parents=True, exist_ok=True)
        arguments.status.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if payload["alert"]:
            raise SystemExit(2)
        time.sleep(arguments.interval)


if __name__ == "__main__":
    main()
