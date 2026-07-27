# -*- coding: utf-8 -*-
"""闭环验收脚本：大模型逐段控制 CraftGround，收集原木 → 开背包 → 合成工作台。

用法（WSL，需 ALSOFT_DRIVERS=null 规避 OpenAL 崩溃）：
    PYTHONPATH=<repo> python -m rl_training_environments.craftground.run_llm_log_collection \
        --seed 2026 --model claude-sonnet-5 --output-directory runs/llm-log-collection/v1

产物：帧 PNG + trajectory.json + 提示词快照，供人工写轨迹 md。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

from rl_training_environments.craftground.llm_segment_controller import (
    AnthropicSegmentClient,
    ControllerLimits,
    SegmentControllerSession,
)
from rl_training_environments.craftground.segment_text_codec import MAX_SEGMENT_TICKS

# 目标物品的 translation_key 尾段。橡木与深色橡木都算原木。
LOG_ITEM_NAMES = ("oak_log", "dark_oak_log", "birch_log", "spruce_log", "jungle_log",
                  "acacia_log", "cherry_log", "mangrove_log", "pale_oak_log")
PLANK_ITEM_NAMES = ("oak_planks", "dark_oak_planks", "birch_planks", "spruce_planks",
                    "jungle_planks", "acacia_planks", "cherry_planks", "mangrove_planks",
                    "pale_oak_planks")
REQUIRED_LOG_COUNT = 4
# 徒手挖石头掉落物为空，必须先有木镐，所以 stone 阶段隐含整条合成链。
STONE_ITEM_NAMES = ("cobblestone", "stone", "cobbled_deepslate", "deepslate")
REQUIRED_STONE_COUNT = 3


def count_matching(counts: Dict[str, int], names: Tuple[str, ...]) -> int:
    return sum(count for name, count in counts.items() if name in names)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="2026")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--max-total-ticks", type=int, default=5000)
    parser.add_argument("--max-wall-clock-seconds", type=float, default=2100.0)
    parser.add_argument("--port", type=int, default=8451)
    parser.add_argument("--chain-to-stone", action="store_true",
                        help="原木阶段达成后自动接续石头阶段（沿用同一局与同一份记忆）")
    parser.add_argument("--stone-rounds", type=int, default=24,
                        help="石头阶段额外给多少轮")
    parser.add_argument("--stone-wall-clock-seconds", type=float, default=1800.0,
                        help="石头阶段额外给多少墙钟秒")
    parser.add_argument("--stage", default="collect",
                        choices=("collect", "full", "stone"),
                        help="collect=只收集原木；full=再合成工作台；stone=再下洞采石头")
    return parser.parse_args()


def build_environment(seed: str, port: int, commands):
    from craftground import CraftGroundEnvironment
    from craftground.initial_environment_config import (
        Difficulty, GameMode, InitialEnvironmentConfig, WorldType,
    )
    from craftground.environment.action_space import ActionSpaceVersion
    from craftground.screen_encoding_modes import ScreenEncodingMode

    config = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        world_type=WorldType.DEFAULT,
        gamemode=GameMode.SURVIVAL,
        difficulty=Difficulty.PEACEFUL,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        seed=seed,
        requires_biome_info=True,
        # 准心射线：告诉模型"瞄的是什么方块、多远"。v2 缺这个，模型只能靠反复撞墙测距。
        request_raycast=True,
        initial_extra_commands=list(commands),
    )
    return CraftGroundEnvironment(
        config, action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        port=port, find_free_port=True, verbose=False,
    )


def make_goal_check(stage: str):
    """产出 goal_check(counts, observation) → (done, note)。"""

    def goal_check(counts: Dict[str, int], observation) -> Tuple[bool, str]:
        log_count = count_matching(counts, LOG_ITEM_NAMES)
        plank_count = count_matching(counts, PLANK_ITEM_NAMES)
        table_count = counts.get("crafting_table", 0)
        if stage == "collect":
            if log_count >= REQUIRED_LOG_COUNT:
                return True, f"原木 {log_count} 个（≥{REQUIRED_LOG_COUNT}）"
            return False, ""
        if stage == "stone":
            stone_count = count_matching(counts, STONE_ITEM_NAMES)
            if stone_count >= REQUIRED_STONE_COUNT:
                return True, (f"石头 {stone_count} 个（≥{REQUIRED_STONE_COUNT}），"
                              f"原木 {log_count}，工作台 {table_count}")
            return False, ""
        if table_count >= 1:
            return True, f"工作台 {table_count} 个，原木 {log_count}，木板 {plank_count}"
        return False, ""

    return goal_check


def main() -> None:
    arguments = parse_arguments()
    output_directory = Path(arguments.output_directory)
    frame_directory = output_directory / "frames"
    # trajectory.json 会被整体覆盖，frames 必须跟着清空：否则重跑同一目录会留下上一次的
    # 帧，json 描述本次而目录里混着上次的图，同名帧还会被部分覆盖——这种不同步比丢数据
    # 更难查。要保留旧结果就换 --output-directory。
    stale_frames = sorted(frame_directory.glob("*.png")) if frame_directory.exists() else []
    for stale in stale_frames:
        stale.unlink()
    if stale_frames:
        print(f"[run] 清理上次残留帧 {len(stale_frames)} 张", flush=True)
    frame_directory.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    if not base_url or not auth_token:
        raise SystemExit("需要环境变量 ANTHROPIC_BASE_URL 与 ANTHROPIC_AUTH_TOKEN")

    commands = ["gamemode survival @p", "time set day", "weather clear",
                "gamerule doDaylightCycle false", "gamerule doMobSpawning false"]

    # 任务文本只陈述目标与验收口径，不给分步做法——怎么做是模型要解决的问题。
    collect_task = (
        f"收集 {REQUIRED_LOG_COUNT} 个原木。你出生在一片森林里。\n"
        f"  每轮会告诉你背包里有什么，原木数攒到 {REQUIRED_LOG_COUNT} 即算完成。"
    )
    if arguments.stage == "collect":
        task_text = collect_task
    elif arguments.stage == "stone":
        task_text = (
            collect_task
            + f"\n  拿到原木后，再采到 {REQUIRED_STONE_COUNT} 块石头（cobblestone）。"
            "\n  石头在地表的裸岩、山体或洞穴里。徒手挖石头得不到任何东西。"
        )
    else:
        task_text = (
            collect_task
            + f"\n  攒够 {REQUIRED_LOG_COUNT} 个原木后，再合成出 1 个工作台（crafting_table）。"
        )

    print(f"[run] 冷启动环境 seed={arguments.seed} ...", flush=True)
    started = time.time()
    environment = build_environment(arguments.seed, arguments.port, commands)
    environment.reset()
    print(f"[run] 冷启动完成 {time.time() - started:.1f}s", flush=True)

    client = AnthropicSegmentClient(base_url, auth_token, arguments.model)
    limits = ControllerLimits(
        max_rounds=arguments.max_rounds,
        max_total_ticks=arguments.max_total_ticks,
        max_wall_clock_seconds=arguments.max_wall_clock_seconds,
    )
    session = SegmentControllerSession(
        environment, client, task_text, limits, frame_directory, commands,
    )

    print("[run] 快速回档到初始状态 ...", flush=True)
    session.fast_reset()
    spawn_full = session._current_observation["full"]
    print(f"[run] 出生点 biome={spawn_full.biome_info.biome_name} "
          f"pos=({spawn_full.x:.1f},{spawn_full.y:.1f},{spawn_full.z:.1f})", flush=True)

    stop_reason = session.run(make_goal_check(arguments.stage))
    print(f"[run] 结束：{stop_reason}", flush=True)
    reached_goal = stop_reason.startswith("目标达成")

    # 达成后自动接续下一阶段：原木 → 石头。沿用同一 session，背包与世界都不重置，
    # 事件日志与真值台账继续累积，这样后一阶段能用上前一阶段学到的设备事实。
    # rounds_done 是**该阶段结束时的累计轮数**（同一 session 跨阶段续跑，round_index 不重置），
    # 不是本阶段自己跑了几轮。first_round/last_round 把区间写明，免得下游误读成阶段轮数。
    stage_results = [{"stage": arguments.stage, "stop_reason": stop_reason,
                      "rounds_done": len(session.rounds),
                      "first_round": 1, "last_round": len(session.rounds)}]
    if reached_goal and arguments.stage == "collect" and arguments.chain_to_stone:
        print("[run] 原木阶段达成，自动接续石头阶段 ...", flush=True)
        session.prompt_builder.task_text = (
            f"你已经拿到原木了。接下来采到 {REQUIRED_STONE_COUNT} 块石头（cobblestone）。\n"
            "  石头在地表的裸岩、山体或洞穴里。徒手挖石头得不到任何东西。"
        )
        session.limits.max_rounds = len(session.rounds) + arguments.stone_rounds
        session.limits.max_wall_clock_seconds += arguments.stone_wall_clock_seconds
        session.limits.max_total_ticks += arguments.stone_rounds * MAX_SEGMENT_TICKS
        collect_rounds = len(session.rounds)
        stone_reason = session.run(make_goal_check("stone"))
        print(f"[run] 石头阶段结束：{stone_reason}", flush=True)
        stage_results.append({"stage": "stone", "stop_reason": stone_reason,
                              "rounds_done": len(session.rounds),
                              "first_round": collect_rounds + 1,
                              "last_round": len(session.rounds)})
        stop_reason = stone_reason

    payload = {
        "seed": arguments.seed,
        "model": arguments.model,
        "stage": arguments.stage,
        "task_text": task_text,
        "spawn_biome": spawn_full.biome_info.biome_name,
        "spawn_position": [spawn_full.x, spawn_full.y, spawn_full.z],
        "stop_reason": stop_reason,
        "stage_results": stage_results,
        "total_ticks": session.total_ticks,
        "wall_clock_seconds": time.time() - session.started_at,
        "limits": asdict(limits),
        "rounds": [asdict(round_outcome) for round_outcome in session.rounds],
    }
    (output_directory / "trajectory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[run] 已写 {output_directory / 'trajectory.json'}", flush=True)
    environment.close()


if __name__ == "__main__":
    main()
