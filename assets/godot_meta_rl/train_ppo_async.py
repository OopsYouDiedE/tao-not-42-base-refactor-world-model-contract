"""
B：异步 actor-learner 训练（解耦 learner），配合 A 的 atlas 单次回读。

与 train_ppo.py(锁步 SB3 .learn) 的本质区别：把"梯度更新"从"采集线程"挪走，Godot 不再因训练空转。
  - 采集线程(主线程)：持续 握手→behavior 策略推理(no_grad)→送 action→写 rollout buffer。
  - 学习线程：rollout buffer 满就跑 PPO 更新(复用 SB3 PPO.train)；更新后把权重拷给 behavior(双缓冲)。
  - 双缓冲(2 个 rollout buffer + 2 份策略)：采集线程在学习线程训练上一份时，并行采下一份 →
    Godot 空转时间从"整段训练时长"降到"max(0, 训练时长-采集时长)"。权重 staleness ≤ 1 次更新。
  - torch 反传 / win32 等待都释放 GIL → 两线程真重叠；on-policy 语义保持(每步基于当前帧、最多落后 1 更新)。

复用 SB3 的 MultiInputPolicy 网络/优化器/DictRolloutBuffer/GAE/PPO.train()；只自己写采集与权重双缓冲。
对照基线: python train_ppo.py（锁步）。用法: python train_ppo_async.py [总步数]（默认 16000）。
"""

import copy
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import obs_as_tensor

import train_ppo as base   # 复用 GodotVecEnv / Godot 启动常量 / N_STACK / N_BUTTONS
import rl_train_env as E

N_STEPS = 64
BATCH_SIZE = 256
N_EPOCHS = 4

_PROFILE = bool(os.environ.get("ASYNC_PROFILE"))


def build_model(venv):
    """与 train_ppo.py 完全相同的 PPO 超参，保证只变"异步"这一个变量。"""
    model = PPO(
        "MultiInputPolicy", venv,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
        verbose=0, device="auto",
    )
    # 不走 .learn() → 没有自动配置 logger；PPO.train() 会用到，给一个空 logger。
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model


def make_buffer(model):
    return DictRolloutBuffer(
        model.n_steps, model.observation_space, model.action_space,
        device=model.device, gae_lambda=model.gae_lambda,
        gamma=model.gamma, n_envs=model.n_envs,
    )


def collect_rollout(venv, behavior, buf, last_obs, last_starts, device, weights_lock):
    """采一整段 rollout(N_STEPS×n_envs)写进 buf 并算好 GAE。返回 (新 last_obs, 新 last_starts, 本段完成回合的奖励列表)。"""
    buf.reset()
    ep_rews = []
    dones = last_starts  # 占位，循环里会被覆盖
    prof = _PROFILE
    t_inf = t_step = t_add = 0.0
    for _ in range(buf.buffer_size):
        if prof: _t = time.perf_counter()
        with torch.no_grad():
            obs_t = obs_as_tensor(last_obs, device)
            with weights_lock:                       # behavior 权重可能正被学习线程换页
                actions, values, log_probs = behavior(obs_t)
        actions_np = actions.cpu().numpy()
        if prof: _a = time.perf_counter(); t_inf += _a - _t
        new_obs, rewards, dones, infos = venv.step(actions_np)
        if prof: _b = time.perf_counter(); t_step += _b - _a
        buf.add(last_obs, actions_np, rewards, last_starts, values, log_probs)
        last_obs = new_obs
        last_starts = dones
        if prof: t_add += time.perf_counter() - _b
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                ep_rews.append(ep["r"])
    if prof:
        n = buf.buffer_size
        print(f"    [prof] /步: 推理={1000*t_inf/n:.1f}ms  venv.step={1000*t_step/n:.1f}ms  "
              f"buffer.add={1000*t_add/n:.1f}ms", flush=True)
    # rollout 末尾：用 behavior 对最后观测估值，做 GAE 自举。
    with torch.no_grad():
        obs_t = obs_as_tensor(last_obs, device)
        with weights_lock:
            last_values = behavior.predict_values(obs_t)
    buf.compute_returns_and_advantage(last_values=last_values, dones=dones)
    return last_obs, last_starts, ep_rews


def learner_loop(model, behavior, ready_q, free_q, weights_lock, stop_evt):
    """后台：取满了的 buffer 跑 PPO 更新，更新完把权重拷给 behavior(双缓冲，staleness≤1)。"""
    while True:
        try:
            buf = ready_q.get(timeout=0.5)
        except queue.Empty:
            if stop_evt.is_set():
                break
            continue
        if buf is None:
            break
        if os.environ.get("ASYNC_NO_LEARN"):          # 诊断：学习线程空转，量采集线程单独的吞吐天花板
            free_q.put(buf)
            continue
        model.rollout_buffer = buf
        model.policy.set_training_mode(True)
        model.train()                                 # 复用 SB3 的 PPO 更新(clip/value/entropy/GAE 全不变)
        with weights_lock:
            behavior.load_state_dict(model.policy.state_dict())
        free_q.put(buf)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 16000

    log_path = os.path.join(base.PROJECT_DIR, "_train_ppo_async_godot.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    run_env = os.environ.copy()
    run_env["RL_FIXED_STEPS"] = "24"   # 240Hz / 24 = 10Hz 决策频率（与基线一致）

    proc = subprocess.Popen([base.GODOT_EXE, "--path", base.PROJECT_DIR, base.TRAIN_SCENE],
                            stdout=log, stderr=subprocess.STDOUT, env=run_env)
    ok = False
    try:
        print(f"连接 {E.NUM_ENVS} 个并行 Godot 环境 ...")
        venv = base.GodotVecEnv(connect_timeout_s=60)
        venv = VecMonitor(venv)
        venv = VecFrameStack(venv, n_stack=base.N_STACK)
        print(f"已连接。帧堆叠={base.N_STACK}。构建 PPO + 启动异步 actor-learner。\n")

        model = build_model(venv)
        # PPO.__init__ 会在 venv 之上再自动套一层 VecTransposeImage(图像 HWC→CHW)；
        # 必须步进这层之后的 env，否则喂给 CNN 的通道维不对。
        venv = model.env
        device = model.device
        print(f"设备: {device}")

        # behavior：采集线程专用的推理副本；model.policy 由学习线程训练，训练完拷过来。
        behavior = copy.deepcopy(model.policy).to(device)
        behavior.set_training_mode(False)

        weights_lock = threading.Lock()
        stop_evt = threading.Event()
        free_q = queue.Queue()
        ready_q = queue.Queue()
        for _ in range(2):                             # 双缓冲：一份在采、一份在学
            free_q.put(make_buffer(model))

        learner = threading.Thread(
            target=learner_loop,
            args=(model, behavior, ready_q, free_q, weights_lock, stop_evt),
            daemon=True)
        learner.start()

        last_obs = venv.reset()
        last_starts = np.ones(E.NUM_ENVS, dtype=bool)
        recent_rews = deque(maxlen=100)

        steps_done = 0
        rollouts = 0
        t0 = time.perf_counter()
        while steps_done < total_timesteps:
            model._current_progress_remaining = 1.0 - steps_done / max(1, total_timesteps)
            buf = free_q.get()                         # 双缓冲都在飞 → 在此阻塞 = 学习真的成了瓶颈(staleness 仍≤1)
            last_obs, last_starts, ep_rews = collect_rollout(
                venv, behavior, buf, last_obs, last_starts, device, weights_lock)
            ready_q.put(buf)
            steps_done += buf.buffer_size * E.NUM_ENVS
            model.num_timesteps = steps_done
            rollouts += 1
            recent_rews.extend(ep_rews)

            el = time.perf_counter() - t0
            sps = steps_done / el if el else 0.0
            rew_mean = (sum(recent_rews) / len(recent_rews)) if recent_rews else float("nan")
            print(f"[async] steps={steps_done:>7d}  ep_rew_mean={rew_mean:+.3f}  "
                  f"n_eps={len(recent_rews)}  {sps:.0f} sps  rollouts={rollouts}  {el:.0f}s",
                  flush=True)

        stop_evt.set()
        ready_q.put(None)
        learner.join(timeout=10)

        dt = time.perf_counter() - t0
        model.save(os.path.join(base.PROJECT_DIR, "ppo_spotlight_discrete_async"))
        sps = total_timesteps / dt if dt else 0.0
        print(f"\n训练完成：{total_timesteps} 步，用时 {dt:.1f}s（{sps:.0f} env-steps/s，40 并行，异步）。")
        print("模型已保存：ppo_spotlight_discrete_async.zip")
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


if __name__ == "__main__":
    sys.exit(main())
