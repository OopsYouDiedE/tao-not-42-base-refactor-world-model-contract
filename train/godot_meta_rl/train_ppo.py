"""锁步 SB3 PPO 训练 40 个并行 Godot 环境（聚光灯瞄准·离散控制）。对照基线，最简单。

用法: python train/godot_meta_rl/train_ppo.py [总步数]（默认 16000）。
异步对照见 train_ppo_async.py（线程）/ train_ppo_2proc.py（双进程）。
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack

from utils.godot_rl import shared_mem_env as E
from utils.godot_rl.launch import launch_godot, kill_godot
from utils.godot_rl.ppo_factory import build_model
from train.godot_meta_rl.vec_env import GodotVecEnv, RolloutProgress, N_STACK


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 16000

    log_path = os.path.join(E.PROJECT_DIR, "_train_ppo_godot.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    # 240Hz / 24 = 10Hz 决策频率
    proc = launch_godot(log=log, extra_env={"RL_FIXED_STEPS": "24"})
    ok = False
    try:
        print(f"连接 {E.NUM_ENVS} 个并行 Godot 环境 ...")
        venv = GodotVecEnv(connect_timeout_s=60)
        venv = VecMonitor(venv)                        # 记录每回合奖励/长度，便于看 ep_rew_mean
        venv = VecFrameStack(venv, n_stack=N_STACK)    # 图像沿通道堆 4 帧 → 让模型感知角速度
        print(f"已连接。帧堆叠={N_STACK}。构建 PPO(MultiInputPolicy) 并开始训练。\n")

        model = build_model(venv, verbose=1, with_null_logger=False)
        print(f"设备: {model.device}")
        t0 = time.perf_counter()
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=RolloutProgress())
        dt = time.perf_counter() - t0

        model.save(os.path.join(E.PROJECT_DIR, "ppo_spotlight_discrete"))
        sps = total_timesteps / dt if dt else 0.0
        print(f"\n训练完成：{total_timesteps} 步，用时 {dt:.1f}s（{sps:.0f} env-steps/s，40 并行）。")
        print("模型已保存：ppo_spotlight_discrete.zip")
        venv.close()
        ok = True
        return 0
    finally:
        kill_godot(proc)
        log.close()
        if ok:
            try:
                os.remove(log_path)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
