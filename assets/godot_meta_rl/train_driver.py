"""
轻量冒烟/理智检查：不依赖 SB3，用随机离散动作驱动 40 环境握手管线，
验证新共享内存布局(10连续+30离散动作 / 5字段元数据)端到端打通。

校验：收到图像、跨环境一致(方差0)、帧号严格+1(无丢帧)、sim_dt/reward/done 正常流动。
真正的 PPO 训练见 train_ppo.py。
"""

import os
import subprocess
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import rl_train_env as E

# 复用 rl_train_env 中的路径配置
GODOT_EXE = E.GODOT_EXE
PROJECT_DIR = E.PROJECT_DIR
TRAIN_SCENE = E.TRAIN_SCENE

MEASURE_S = 6.0


def main():
    log_path = os.path.join(PROJECT_DIR, "_train_smoke.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([GODOT_EXE, "--path", PROJECT_DIR, TRAIN_SCENE],
                            stdout=log, stderr=subprocess.STDOUT)
    ok = False
    try:
        env = E.GodotTrainEnv(connect_timeout_s=40)
        print(f"已连接 {E.NUM_ENVS} 环境。随机离散动作冒烟 {MEASURE_S:.0f}s ...")

        assert env.wait_obs(5000), "未收到首帧"
        _ = env.read_meta()

        cycles = 0
        frame_ids = []
        max_var = 0.0
        total_reward = 0.0
        dones = 0
        img_nonzero = False
        last_sim_dt = 0.0

        env.send_action(np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32),
                        np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32))

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < MEASURE_S:
            if not env.wait_obs(2000):
                break
            imgs = env.read_images()
            meta = env.read_meta()
            cycles += 1
            frame_ids.append(int(meta[0, E.M_FRAME]))
            max_var = max(max_var, float(np.var(meta[:, E.M_FRAME])))
            total_reward += float(meta[:, E.M_REWARD].mean())
            dones += int((meta[:, E.M_DONE] > 0.5).sum())
            last_sim_dt = float(meta[0, E.M_SIM_DT])
            if not img_nonzero and imgs.any():
                img_nonzero = True

            # 随机策略：disc[0..3] 各 0/1 加速键（演示用，真正训练见 train_ppo.py）
            disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
            disc[:, :4] = np.random.randint(0, 2, size=(E.NUM_ENVS, 4))
            cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
            env.send_action(cont, disc)

        dt = time.perf_counter() - t0
        env.close()

        ids = np.array(frame_ids)
        gaps = np.diff(ids) if len(ids) >= 2 else np.array([1])
        lossless = bool(np.all(gaps == 1))

        print("-" * 56)
        print(f"回合速率        : {cycles/dt:.1f}/s   ({cycles} 回合 / {dt:.2f}s)")
        print(f"收到图像(非零)  : {img_nonzero}")
        print(f"跨环境帧号方差  : {max_var}  (期望 0)")
        print(f"帧号严格+1无丢  : {lossless}  (最大跳 {int(gaps.max())})")
        print(f"sim_dt(steps*dt): {last_sim_dt:.5f}")
        print(f"累计命中(done)  : {dones}")
        print(f"平均奖励/回合   : {total_reward/max(cycles,1):+.3f}")
        ok = img_nonzero and lossless and (max_var == 0.0)
        print(f"=> {'[ PASS ] 训练管线打通' if ok else '[ FAIL ]'}")
        return 0 if ok else 1
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            proc.kill()
        log.close()
        if ok:
            try:
                os.remove(log_path)   # 成功则清理日志（保留失败日志以便排查）
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
