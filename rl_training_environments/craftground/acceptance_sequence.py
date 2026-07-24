"""CraftGround 渲染器验收：跑一条动作序列，每秒截图一次并全程录像。

对外接口：
    python -m rl_training_environments.craftground.acceptance_sequence
        [--steps N] [--seed S] [--output-dir DIR] [--seconds-per-shot 1.0]

无头服务器上必须用 xvfb-run 启动（需要 DISPLAY）：
    xvfb-run -a python -m rl_training_environments.craftground.acceptance_sequence ...

流程：reset → 循环 step 一条预设动作序列 → 每步把 RGB 帧写进 mp4（全程录像），
并按挂钟时间每满 seconds_per_shot 秒抽当前帧存一张 png（每秒截图）。产出：
    sequence.mp4              — 全程录像
    shots/shot_<秒>_f<帧>.png — 每秒一张截图
    summary.json             — 帧数/时长/截图清单/成就等验收指标
所有产物落在 runs/ 之下（gitignored）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from craftground.screen_encoding_modes import ScreenEncodingMode

from rl_training_environments.craftground.environment import (
    DISCRETE_TO_V2,
    MinecraftCraftGroundEnvironment,
)


# 一条动作丰富的验收序列（离散动作 ID，见 environment.DISCRETE_TO_V2）：
# 前进、转头四方向、跳、攻击、边走边看，循环铺满整段，保证画面持续变化。
ACTION_CYCLE = [
    1, 1, 1,        # forward ×3
    15, 15,         # look right
    1, 1,           # forward
    16, 16,         # look left
    12, 12,         # forward+sprint
    13,             # look down
    14,             # look up
    6,              # forward+jump
    8, 8,           # forward+attack（挖/打）
    23, 24,         # forward+look right / left
    19, 20,         # big yaw right / left
]


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) RGB uint8 → OpenCV BGR uint8，保证可写入 mp4/png。"""
    frame = np.ascontiguousarray(rgb)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def run(steps: int, seed: int, output_dir: Path, seconds_per_shot: float,
        video_fps: float) -> dict:
    """执行验收序列，返回汇总字典（同时落盘 mp4 / 截图 / summary.json）。

    Args:
        steps: 总步数（每步一个动作、渲染一帧）。
        seed: 环境种子，用于可复现。
        output_dir: 产物目录。
        seconds_per_shot: 截图间隔（挂钟秒）。
        video_fps: 录像回放帧率。

    Returns:
        验收指标汇总。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shots_dir = output_dir / "shots"
    shots_dir.mkdir(exist_ok=True)

    env = MinecraftCraftGroundEnvironment(
        seed=seed, max_steps=steps + 10,
        screen_encoding_mode=ScreenEncodingMode.RAW,
    )
    rgb = env.reset()
    height, width = rgb.shape[:2]

    writer = cv2.VideoWriter(
        str(output_dir / "sequence.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps, (width, height),
    )

    start = time.time()
    next_shot_at = 0.0            # 下一张截图的相对秒阈值（0 秒先截一张开局）
    shots: list[dict] = []
    achievements: set[int] = set()
    nonblack_frames = 0

    for step_index in range(steps):
        action_id = ACTION_CYCLE[step_index % len(ACTION_CYCLE)]
        rgb, reward, done, info = env.step(action_id)
        bgr = _to_bgr(rgb)
        writer.write(bgr)
        if bgr.mean() > 1.0:
            nonblack_frames += 1
        for ach in info.get("successes", []) or []:
            achievements.add(int(ach))

        elapsed = time.time() - start
        if elapsed >= next_shot_at:
            second = int(round(next_shot_at))
            name = f"shot_{second:04d}s_f{step_index:05d}.png"
            labeled = bgr.copy()
            cv2.rectangle(labeled, (0, 0), (width, 24), (0, 0, 0), -1)
            cv2.putText(labeled, f"t={elapsed:5.1f}s step={step_index} act={action_id}",
                        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(shots_dir / name), labeled)
            shots.append({"file": name, "elapsed_s": round(elapsed, 3),
                          "step": step_index, "action_id": action_id,
                          "frame_mean": round(float(bgr.mean()), 2)})
            next_shot_at += seconds_per_shot

        if done:
            rgb = env.reset()

    writer.release()
    env.close()
    total_elapsed = time.time() - start

    summary = {
        "steps": steps,
        "seed": seed,
        "frame_size": [int(width), int(height)],
        "total_wall_seconds": round(total_elapsed, 2),
        "video_fps": video_fps,
        "seconds_per_shot": seconds_per_shot,
        "shots_taken": len(shots),
        "nonblack_frames": nonblack_frames,
        "achievements_unlocked": sorted(achievements),
        "shots": shots,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CraftGround 每秒截图 + 录像验收序列")
    parser.add_argument("--steps", type=int, default=120, help="总步数")
    parser.add_argument("--seed", type=int, default=0, help="环境种子")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/craftground-acceptance"), help="产物目录")
    parser.add_argument("--seconds-per-shot", type=float, default=1.0, help="截图间隔（秒）")
    parser.add_argument("--video-fps", type=float, default=10.0, help="录像回放帧率")
    args = parser.parse_args()

    summary = run(args.steps, args.seed, args.output_dir,
                  args.seconds_per_shot, args.video_fps)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n产物目录: {args.output_dir.resolve()}")
    print(f"录像: {args.output_dir / 'sequence.mp4'}")
    print(f"截图: {args.output_dir / 'shots'}/  （{summary['shots_taken']} 张）")


if __name__ == "__main__":
    main()
