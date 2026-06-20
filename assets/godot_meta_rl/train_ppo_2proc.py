"""
B+杠杆2：双进程 actor-learner（去 GIL）。配合 A 的 atlas 单次回读。

线程版(train_ppo_async.py)实测：学习线程激活时采集的 CPU 工作被 GIL 饿到——buffer.add 6→14ms、
venv.step 22→29ms，而推理(GPU)只 +3ms。即 736→1003 sps 的差距【主要是 GIL 争用】。本文件把学习
挪到【独立进程】消掉 GIL 争用。

进程划分：
  - 采集进程(主)：步进 Godot + behavior 推理 + 填 rollout buffer + 算 GAE。
  - 学习进程(子)：复用 SB3 PPO.train 更新策略，把新权重 state_dict 回传。
数据通道（避免 503MB 拷贝）：
  - 图像 obs(每 buffer ~503MB) 放【共享内存】(2 块双缓冲)，两进程就地读写，零拷贝。
  - 其余小数组(action/value/log_prob/GAE/reward/episode_start，共 ~100KB) + 权重(~10MB) 走 Queue(便宜)。
  - 双缓冲 + free/ready 队列握手：采集只在学习 put(idx) 回 free_q 后才复用该 SHM 块 → 无竞争，staleness≤1。

用法: python train_ppo_2proc.py [总步数]（默认 16000）。对照: train_ppo_async.py(线程版) / train_ppo.py(锁步)。
"""

import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import deque
from multiprocessing import shared_memory

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, VecEnv
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import obs_as_tensor

import train_ppo as base
import rl_train_env as E

PPO_HP = dict(n_steps=64, batch_size=256, n_epochs=4,
              gamma=0.99, gae_lambda=0.95, ent_coef=0.01)
SHM_IMG = ["godotrl_imgbuf0", "godotrl_imgbuf1"]   # 双缓冲图像共享内存名


def _small_keys():
    return ["sim_dt", "actions", "values", "log_probs", "advantages",
            "returns", "rewards", "episode_starts"]


def _extract_small(buf):
    """从一个填满+算好 GAE 的 buffer 取出除图像外的小数组(拷贝，走 Queue)。"""
    return {
        "sim_dt": buf.observations["sim_dt"].copy(),
        "actions": buf.actions.copy(),
        "values": buf.values.copy(),
        "log_probs": buf.log_probs.copy(),
        "advantages": buf.advantages.copy(),
        "returns": buf.returns.copy(),
        "rewards": buf.rewards.copy(),
        "episode_starts": buf.episode_starts.copy(),
    }


def _bind_small(buf, img_view, small):
    """把一个 buffer 的内部数组指向给定图像视图 + 小数组，供 PPO.train 直接读(零额外拷贝)。"""
    buf.observations["image"] = img_view
    buf.observations["sim_dt"] = small["sim_dt"]
    buf.actions = small["actions"]
    buf.values = small["values"]
    buf.log_probs = small["log_probs"]
    buf.advantages = small["advantages"]
    buf.returns = small["returns"]
    buf.rewards = small["rewards"]
    buf.episode_starts = small["episode_starts"]


# ============================ 学习进程 ============================
def learner_proc(obs_space, act_space, img_shape, ready_q, weights_q, free_q):
    """独立进程：复用 SB3 PPO.train 更新策略，把新权重回传给采集进程。"""
    try:
        _learner_main(obs_space, act_space, img_shape, ready_q, weights_q, free_q)
    except Exception:
        import traceback
        with open(os.path.join(base.PROJECT_DIR, "_learner_err.log"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise


def _learner_main(obs_space, act_space, img_shape, ready_q, weights_q, free_q):
    class _Dummy(VecEnv):
        def __init__(s): VecEnv.__init__(s, E.NUM_ENVS, obs_space, act_space)
        def reset(s): return None
        def step_async(s, a): pass
        def step_wait(s): return None
        def close(s): pass
        def get_attr(s, n, i=None): return [None] * E.NUM_ENVS
        def set_attr(s, n, v, i=None): pass
        def env_method(s, *a, indices=None, **k): return []
        def env_is_wrapped(s, w, i=None): return [False] * E.NUM_ENVS

    model = PPO("MultiInputPolicy", _Dummy(), device="auto", verbose=0, **PPO_HP)
    model.set_logger(Logger(folder=None, output_formats=[]))
    model._current_progress_remaining = 1.0          # 没走 .learn()，train() 的 clip_range 调度需要它

    img_shms = [shared_memory.SharedMemory(name=n) for n in SHM_IMG]
    img_views = [np.ndarray(img_shape, dtype=np.uint8, buffer=s.buf) for s in img_shms]

    # 把初始权重发给采集进程，使两边对齐。
    weights_q.put({k: v.detach().cpu() for k, v in model.policy.state_dict().items()})

    buf = DictRolloutBuffer(model.n_steps, obs_space, act_space, device=model.device,
                            gae_lambda=model.gae_lambda, gamma=model.gamma, n_envs=model.n_envs)
    while True:
        item = ready_q.get()
        if item is None:
            break
        idx, small = item
        _bind_small(buf, img_views[idx], small)
        buf.pos = model.n_steps
        buf.full = True
        buf.generator_ready = False                  # 复用 buffer 对象：让 get() 重新 flatten
        model.rollout_buffer = buf                    # 让 PPO.train 读这个(已填+GAE)的 buffer
        model.policy.set_training_mode(True)
        model.train()
        weights_q.put({k: v.detach().cpu() for k, v in model.policy.state_dict().items()})
        free_q.put(idx)
    for s in img_shms:
        s.close()


# ============================ 采集进程(主) ============================
def collect_into(venv, behavior, buf, img_view, last_obs, last_starts, device):
    """填一整段 rollout：图像 obs 直接写进 SHM 视图(img_view)，其余写进 buf 自带数组；末尾算 GAE。"""
    buf.reset()
    buf.observations["image"] = img_view          # reset 会重建数组 → 重新指向 SHM
    ep_rews = []
    dones = last_starts
    for _ in range(buf.buffer_size):
        with torch.no_grad():
            actions, values, log_probs = behavior(obs_as_tensor(last_obs, device))
        actions_np = actions.cpu().numpy()
        new_obs, rewards, dones, infos = venv.step(actions_np)
        buf.add(last_obs, actions_np, rewards, last_starts, values, log_probs)
        last_obs = new_obs
        last_starts = dones
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                ep_rews.append(ep["r"])
    with torch.no_grad():
        last_values = behavior.predict_values(obs_as_tensor(last_obs, device))
    buf.compute_returns_and_advantage(last_values=last_values, dones=dones)
    return last_obs, last_starts, ep_rews


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 16000

    log_path = os.path.join(base.PROJECT_DIR, "_train_ppo_2proc_godot.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    run_env = os.environ.copy()
    run_env["RL_FIXED_STEPS"] = "24"
    proc = subprocess.Popen([base.GODOT_EXE, "--path", base.PROJECT_DIR, base.TRAIN_SCENE],
                            stdout=log, stderr=subprocess.STDOUT, env=run_env)
    learner = None
    img_shms = []
    ok = False
    try:
        print(f"连接 {E.NUM_ENVS} 个并行 Godot 环境 ...")
        venv = base.GodotVecEnv(connect_timeout_s=60)
        venv = VecMonitor(venv)
        venv = VecFrameStack(venv, n_stack=base.N_STACK)
        model = PPO("MultiInputPolicy", venv, device="auto", verbose=0, **PPO_HP)
        venv = model.env                             # 含 VecTransposeImage(CHW)
        device = model.device
        behavior = model.policy
        print(f"已连接。设备={device}。启动双进程 actor-learner。\n")

        obs_space = model.observation_space
        act_space = model.action_space
        img_shape = (model.n_steps, model.n_envs) + obs_space["image"].shape
        img_nbytes = int(np.prod(img_shape))         # uint8

        # 创建 2 块图像共享内存(若上次异常残留先清掉)。
        img_views = []
        for name in SHM_IMG:
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
            s = shared_memory.SharedMemory(create=True, size=img_nbytes, name=name)
            img_shms.append(s)
            img_views.append(np.ndarray(img_shape, dtype=np.uint8, buffer=s.buf))

        ctx = mp.get_context("spawn")
        ready_q, weights_q, free_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        learner = ctx.Process(target=learner_proc,
                              args=(obs_space, act_space, img_shape, ready_q, weights_q, free_q),
                              daemon=True)
        learner.start()

        behavior.load_state_dict({k: v.to(device) for k, v in weights_q.get().items()})  # 初始对齐
        behavior.set_training_mode(False)

        buf = DictRolloutBuffer(model.n_steps, obs_space, act_space, device=device,
                                gae_lambda=model.gae_lambda, gamma=model.gamma, n_envs=model.n_envs)
        free_idx = deque([0, 1])                      # 双缓冲都空闲

        last_obs = venv.reset()
        last_starts = np.ones(E.NUM_ENVS, dtype=bool)
        recent = deque(maxlen=100)
        steps_done = 0
        rollouts = 0
        t0 = time.perf_counter()
        while steps_done < total_timesteps:
            while not free_q.empty():                 # 回收学习进程用完的缓冲
                free_idx.append(free_q.get())
            while not weights_q.empty():              # 加载最新权重(staleness≤1)
                behavior.load_state_dict({k: v.to(device) for k, v in weights_q.get().items()})
            if not free_idx:                          # 双缓冲都在飞 → 等学习进程腾出一块
                free_idx.append(free_q.get())
            idx = free_idx.popleft()

            last_obs, last_starts, ep_rews = collect_into(
                venv, behavior, buf, img_views[idx], last_obs, last_starts, device)
            ready_q.put((idx, _extract_small(buf)))
            steps_done += buf.buffer_size * E.NUM_ENVS
            rollouts += 1
            recent.extend(ep_rews)

            el = time.perf_counter() - t0
            sps = steps_done / el if el else 0.0
            rew = (sum(recent) / len(recent)) if recent else float("nan")
            print(f"[2proc] steps={steps_done:>7d}  ep_rew_mean={rew:+.3f}  n_eps={len(recent)}  "
                  f"{sps:.0f} sps  rollouts={rollouts}  {el:.0f}s", flush=True)

        ready_q.put(None)
        # 收最后一份权重存盘。
        while not weights_q.empty():
            behavior.load_state_dict({k: v.to(device) for k, v in weights_q.get().items()})
        learner.join(timeout=15)

        dt = time.perf_counter() - t0
        model.save(os.path.join(base.PROJECT_DIR, "ppo_spotlight_discrete_2proc"))
        print(f"\n训练完成：{total_timesteps} 步，用时 {dt:.1f}s（{total_timesteps/dt:.0f} env-steps/s，双进程）。")
        venv.close()
        ok = True
        return 0
    finally:
        if learner is not None and learner.is_alive():
            learner.terminate()
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            proc.kill()
        for s in img_shms:
            try:
                s.close(); s.unlink()
            except Exception:
                pass
        log.close()
        _ = ok


if __name__ == "__main__":
    sys.exit(main())
