"""把 40 个锁步 Godot 环境暴露为 SB3 VecEnv（聚光灯瞄准·离散控制专用，不可复用 → 属 train 层）。

对外接口：GodotVecEnv（SB3 VecEnv 适配器）、RolloutProgress（每 rollout 打一行进度回调）、
N_STACK / N_BUTTONS（本任务的帧堆叠数与离散按钮数）。

观测 = Dict{ image:(128,128,3)uint8, sim_dt:(1,)float32 }；动作 = MultiBinary(4)（通用离散接口前 4 槽 上/下/左/右）。
40 个子环境锁步推进，天然是 num_envs=40 的向量化环境，且无丢帧。
"""

import time

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from utils.godot_rl import shared_mem_env as E

N_STACK = 4    # 帧堆叠帧数：单帧看不见角速度，连续 4 帧让 CNN 能推出"正在以多快转"
N_BUTTONS = 4  # disc[0..3] = 上/下/左/右


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
        # 软件渲染/Linux 首个真实渲染帧含一次性着色器编译；预热吃掉它再返回首观测。
        assert self.env.warmup(timeout_ms=120000, frames=2), "reset：预热未收到渲染帧"
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
