"""线程版异步 actor-learner 训练（解耦 learner），配合 Main.cs 的 atlas 单次回读。

与锁步 train_ppo.py 的本质区别：把"梯度更新"从"采集线程"挪走，Godot 不再因训练空转。
  - 采集线程(主)：握手→behavior 推理(no_grad)→送 action→写 rollout buffer。
  - 学习线程：buffer 满就跑 PPO 更新(复用 SB3 PPO.train)；更新后把权重拷给 behavior(双缓冲，staleness≤1)。
复用 SB3 的 MultiInputPolicy/优化器/DictRolloutBuffer/GAE/PPO.train()；只自己写采集与权重双缓冲。

用法: python train/godot_meta_rl/train_ppo_async.py [总步数]（默认 16000）。对照: train_ppo.py（锁步）。
"""

import copy
import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
from stable_baselines3.common.utils import obs_as_tensor

from utils.godot_rl import shared_mem_env as E
from utils.godot_rl.launch import launch_godot, kill_godot
from utils.godot_rl.ppo_factory import build_model, make_buffer
from train.godot_meta_rl.vec_env import GodotVecEnv, N_STACK

_PROFILE = bool(os.environ.get("ASYNC_PROFILE"))


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

    log_path = os.path.join(E.PROJECT_DIR, "_train_ppo_async_godot.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = launch_godot(log=log, extra_env={"RL_FIXED_STEPS": "24"})
    ok = False
    try:
        print(f"连接 {E.NUM_ENVS} 个并行 Godot 环境 ...")
        venv = GodotVecEnv(connect_timeout_s=60)
        venv = VecMonitor(venv)
        venv = VecFrameStack(venv, n_stack=N_STACK)
        print(f"已连接。帧堆叠={N_STACK}。构建 PPO + 启动异步 actor-learner。\n")

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
        model.save(os.path.join(E.PROJECT_DIR, "ppo_spotlight_discrete_async"))
        sps = total_timesteps / dt if dt else 0.0
        print(f"\n训练完成：{total_timesteps} 步，用时 {dt:.1f}s（{sps:.0f} env-steps/s，40 并行，异步）。")
        print("模型已保存：ppo_spotlight_discrete_async.zip")
        venv.close()
        ok = True
        return 0
    finally:
        kill_godot(proc)
        log.close()
        _ = ok


if __name__ == "__main__":
    sys.exit(main())
