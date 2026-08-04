"""运行六条真实本地视觉策略 CraftGround 闭环轨迹。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from online_environment_interaction_agents import LocalVisionPolicyBackend
from shared_tools import atomic_write_json

from .run_four_teacher_trajectories import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--baseline-world", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port-base", type=int, default=19900)
    parser.add_argument("--action-budget-ticks", type=int, default=512)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--socket-ipc", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    arguments = parser.parse_args()
    backend = LocalVisionPolicyBackend(
        arguments.model,
        adapter=arguments.adapter,
        load_in_4bit=not arguments.no_4bit,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        max_new_tokens=arguments.max_new_tokens,
    )
    run(
        arguments.output,
        action_budget_ticks=arguments.action_budget_ticks,
        max_generations=arguments.max_generations,
        warmup_ticks=arguments.warmup_ticks,
        backend_name="local-vision-policy",
        backend=backend,
        port_base=arguments.port_base,
        use_shared_memory=not arguments.socket_ipc,
        baseline_world_path=arguments.baseline_world,
        target_log_count=1,
        trajectory_count=6,
        initialization_workers=1,
        environment_count=1,
        rollout_workers=1,
    )
    records_path = arguments.output / "on-policy-generations.json"
    atomic_write_json(
        records_path,
        {
            "provider": backend.provider,
            "model": backend.model,
            "adapter": backend.adapter,
            "policy_version": backend.policy_version,
            "generations": [asdict(record) for record in backend.records()],
        },
    )
    print(records_path)


if __name__ == "__main__":
    main()
