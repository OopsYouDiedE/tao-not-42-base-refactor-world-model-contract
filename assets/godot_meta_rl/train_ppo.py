"""
用 Stable-Baselines3 PPO 训练 40 个并行 Godot 环境（聚光灯瞄准·离散控制）。

把 Main.cs 编排的 40 环境握手包装成 SB3 的 VecEnv：
  - 观测 = Dict{ image:(128,128,3)uint8, sim_dt:(1,)float32 }  ← sim_dt 即“物理步数×步长”，喂给模型
  - 动作 = MultiBinary(4)  ← 本任务用通用离散接口的前 4 个槽位(上/下/左/右加速键)
  - 40 个子环境锁步推进，天然就是 num_envs=40 的向量化环境，且无丢帧。

用法: python train_ppo.py [总步数]    （默认 16000）
"""

import os
import subprocess
import sys
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

import rl_train_env as E

N_STACK = 4   # 帧堆叠帧数：单帧看不见角速度，连续 4 帧让 CNN 能推出"正在以多快转"


class RolloutProgress(BaseCallback):
    """每个 rollout 结束打一行紧凑进度，便于看 ep_rew_mean 趋势是否上升。"""

    def __init__(self):
        super().__init__()
        self._t0 = None

    def _on_training_start(self):
        self._t0 = time.perf_counter()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf:
            rews = [ep["r"] for ep in buf]
            lens = [ep["l"] for ep in buf]
            rew_mean = sum(rews) / len(rews)
            len_mean = sum(lens) / len(lens)
        else:
            rew_mean = float("nan")
            len_mean = float("nan")
        el = time.perf_counter() - self._t0
        sps = self.num_timesteps / el if el else 0.0
        print(f"[progress] steps={self.num_timesteps:>7d}  ep_rew_mean={rew_mean:+.3f}  "
              f"ep_len_mean={len_mean:6.1f}  n_eps={len(buf)}  {sps:.0f} sps  {el:.0f}s",
              flush=True)

# 复用 rl_train_env 中的路径配置
GODOT_EXE = E.GODOT_EXE
PROJECT_DIR = E.PROJECT_DIR
TRAIN_SCENE = E.TRAIN_SCENE

N_BUTTONS = 4   # disc[0..3] = 上/下/左/右


class GodotVecEnv(VecEnv):
    """把 40 个锁步 Godot 环境暴露为 SB3 VecEnv（num_envs=40）。"""

    def __init__(self, connect_timeout_s=60.0):
        self.env = E.GodotTrainEnv(connect_timeout_s=connect_timeout_s)
        obs_space = spaces.Dict({
            "image": spaces.Box(0, 255, (E.IMAGE_HEIGHT, E.IMAGE_WIDTH, E.CHANNELS), dtype=np.uint8),
            "sim_dt": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
        })
        act_space = spaces.MultiBinary(N_BUTTONS)
        super().__init__(E.NUM_ENVS, obs_space, act_space)
        self._actions = np.zeros((E.NUM_ENVS, N_BUTTONS), np.int32)
        self._meta = None

    # ---- 核心：reset / step ----
    def reset(self):
        assert self.env.wait_obs(10000), "reset：未收到首帧观测"
        return self._read_obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.int32).reshape(E.NUM_ENVS, N_BUTTONS)

    def step_wait(self):
        disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
        disc[:, :N_BUTTONS] = self._actions
        cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
        self.env.send_action(cont, disc)
        assert self.env.wait_obs(10000), "step：未收到观测"

        obs = self._read_obs()
        rewards = self._meta[:, E.M_REWARD].astype(np.float32)
        dones = self._meta[:, E.M_DONE] > 0.5
        infos = []
        for i in range(E.NUM_ENVS):
            info = {}
            if dones[i]:
                # Godot 已在发布该终止观测后自动重置该环境；按 SB3 约定提供终止观测。
                info["terminal_observation"] = {"image": obs["image"][i], "sim_dt": obs["sim_dt"][i]}
                info["TimeLimit.truncated"] = False
            infos.append(info)
        return obs, rewards, np.asarray(dones), infos

    def _read_obs(self):
        imgs = self.env.read_images().copy()
        self._meta = self.env.read_meta()
        sim_dt = self._meta[:, E.M_SIM_DT:E.M_SIM_DT + 1].astype(np.float32).copy()
        return {"image": imgs, "sim_dt": sim_dt}

    def close(self):
        self.env.close()

    # ---- VecEnv 其余抽象方法（PPO 基本训练用不到，给最小实现）----
    def _resolve(self, indices):
        if indices is None:
            return list(range(E.NUM_ENVS))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._resolve(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None for _ in self._resolve(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._resolve(indices)]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 16000

    log_path = os.path.join(PROJECT_DIR, "_train_ppo_godot.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    
    run_env = os.environ.copy()
    run_env["RL_FIXED_STEPS"] = "24"  # 240Hz / 24 = 10Hz 决策频率
    
    proc = subprocess.Popen([GODOT_EXE, "--path", PROJECT_DIR, TRAIN_SCENE],
                            stdout=log, stderr=subprocess.STDOUT, env=run_env)
    ok = False
    try:
        print(f"连接 {E.NUM_ENVS} 个并行 Godot 环境 ...")
        venv = GodotVecEnv(connect_timeout_s=60)
        venv = VecMonitor(venv)        # 记录每回合奖励/长度，便于看 ep_rew_mean
        venv = VecFrameStack(venv, n_stack=N_STACK)   # 图像沿通道堆 4 帧 → 让模型感知角速度
        print(f"已连接。帧堆叠={N_STACK}。构建 PPO(MultiInputPolicy) 并开始训练。\n")

        model = PPO(
            "MultiInputPolicy", venv,
            n_steps=64, batch_size=256, n_epochs=4,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            verbose=1, device="auto",
        )
        print(f"设备: {model.device}")
        t0 = time.perf_counter()
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=RolloutProgress())
        dt = time.perf_counter() - t0

        model.save(os.path.join(PROJECT_DIR, "ppo_spotlight_discrete"))
        sps = total_timesteps / dt if dt else 0.0
        print(f"\n训练完成：{total_timesteps} 步，用时 {dt:.1f}s（{sps:.0f} env-steps/s，40 并行）。")
        print("模型已保存：ppo_spotlight_discrete.zip")
        venv.close()
        ok = True
        return 0
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            proc.kill()
        log.close()
        if ok:
            try:
                os.remove(log_path)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
