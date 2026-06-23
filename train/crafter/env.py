"""Crafter 向量化环境封装 (train/crafter/env.py)。

对外接口:
    VecCrafterEnv — 顺序并行的 n_envs 个 Crafter 实例,管理成就检测与 AD 示范提取。

注: Crafter 使用 gym(非 gymnasium)接口,step 返回 (obs, rew, done, info)。
obs 归一化: uint8 (H,W,C) → float32 (C,H,W) [0,1]。
"""
import copy

import numpy as np
import torch
import crafter

from train.crafter.ad_buffer import ACHIEVEMENTS


class VecCrafterEnv:
    """顺序向量化 Crafter 环境。

    每个 env 在主进程内顺序执行,不使用 multiprocessing(Colab 兼容)。
    内部维护长度 max_history 的滑窗 (prev_obs, action) 历史,
    用于在成就解锁时提取 AD 示范段。

    Args:
        n_envs:      并行环境数。
        device:      obs tensor 目标设备。
        seed:        第 i 个 env 使用 seed+i 初始化(如 crafter.Env 支持)。
        max_history: 每个 env 保留的最近步历史长度(≥ demo_len)。

    返回的 obs: (n_envs, 3, H, W) float32 on device。
    """

    OBS_H, OBS_W = 64, 64

    def __init__(self, n_envs: int, device: str = "cuda", seed: int = 0,
                 max_history: int = 128, state_cache=None, p_resume: float = 0.0):
        self.n_envs = n_envs
        self.device = device
        self.max_history = max_history
        # Go-Explore 课程(可选):done 时以 p_resume 从 state_cache 空降存档点而非 fresh reset;
        # 解锁新成就时把当前状态存档。state_cache=None ⇒ 永远 fresh(评测 env 用)。
        self.state_cache = state_cache
        self.p_resume = p_resume
        self._rng = np.random.RandomState(seed + 9973)

        self.envs = [crafter.Env() for _ in range(n_envs)]

        # 滑窗历史: (prev_obs, action) 对 — 在 step() 前捕获
        self._hist_obs: list[list] = [[] for _ in range(n_envs)]
        self._hist_act: list[list[int]] = [[] for _ in range(n_envs)]

        # 当前 episode 已解锁的成就集合(按 env)
        self._prev_ach: list[dict] = [{} for _ in range(n_envs)]

        # 每个 env 当前 obs(在 step 前有效,用于写入历史)
        self._cur_obs: list = [None] * n_envs

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self):
        """重置所有 env。返回 (n_envs, 3, H, W) float32 tensor。"""
        obs_list = []
        for i, env in enumerate(self.envs):
            raw = env.reset()
            proc = self._proc(raw)
            obs_list.append(proc)
            self._cur_obs[i] = proc
            self._prev_ach[i] = {}
            self._hist_obs[i].clear()
            self._hist_act[i].clear()
        return torch.stack(obs_list).to(self.device)

    # ── Go-Explore 快照 / 恢复 ────────────────────────────────────────────────
    def snapshot(self, i: int) -> dict:
        """深拷贝 env i 的当前世界状态为存档 bundle(独立于后续步进)。"""
        cur = self._prev_ach[i]
        unlocked = {a for a in ACHIEVEMENTS if cur.get(a, 0) > 0}
        return {
            "env": copy.deepcopy(self.envs[i]),
            "obs": self._cur_obs[i].clone(),
            "prev_ach": dict(cur),
            "unlocked": unlocked,
        }

    def restore(self, i: int, bundle: dict) -> torch.Tensor:
        """把 env i 恢复到 bundle 存档点(深拷贝避免多次恢复别名),返回该处 obs(CPU tensor)。"""
        self.envs[i] = copy.deepcopy(bundle["env"])
        self._prev_ach[i] = dict(bundle["prev_ach"])
        self._hist_obs[i].clear()
        self._hist_act[i].clear()
        self._cur_obs[i] = bundle["obs"].clone()
        return self._cur_obs[i]

    def step(self, actions):
        """执行一步。

        Args:
            actions: (n_envs,) int,可为 Tensor 或 ndarray。

        Returns:
            obs:              (n_envs, 3, H, W) float32 tensor on device。
            rewards:          (n_envs,) float32 tensor on device。
            dones:            (n_envs,) float32 tensor on device (0/1)。
            infos:            list[dict],长度 n_envs。
            new_achievements: list[(env_idx, ach_name, obs_hist, act_hist)]
                              其中 obs_hist/act_hist 为截取 demo_len 的 CPU list。
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        obs_list, rew_list, done_list, infos = [], [], [], []
        new_achievements = []

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            a = int(a)

            # 步进前先将当前 obs 写入历史 → (prev_obs, action) 语义
            hist_o = self._hist_obs[i]
            hist_a = self._hist_act[i]
            hist_o.append(self._cur_obs[i])   # CPU float32 tensor
            hist_a.append(a)
            if len(hist_o) > self.max_history:
                hist_o.pop(0)
                hist_a.pop(0)

            raw_obs, rew, done, info = env.step(a)
            proc = self._proc(raw_obs)

            # 检测新解锁成就
            cur_ach = info.get("achievements", {})
            prev = self._prev_ach[i]
            had_new = False
            for ach in ACHIEVEMENTS:
                if cur_ach.get(ach, 0) > 0 and prev.get(ach, 0) == 0:
                    new_achievements.append((
                        i, ach,
                        list(hist_o),   # 已含 demo_len 步的 prev_obs
                        list(hist_a),
                    ))
                    had_new = True
            self._prev_ach[i] = dict(cur_ach)

            if done:
                # Go-Explore:以 p_resume 从存档点空降续探,否则 fresh reset。
                bundle = None
                if (self.state_cache is not None and len(self.state_cache) > 0
                        and self._rng.rand() < self.p_resume):
                    bundle = self.state_cache.sample()
                if bundle is not None:
                    proc = self.restore(i, bundle)
                    info["resumed_unlocked"] = set(bundle["unlocked"])
                else:
                    raw_obs = env.reset()
                    proc = self._proc(raw_obs)
                    self._prev_ach[i] = {}
                    self._hist_obs[i].clear()
                    self._hist_act[i].clear()

            self._cur_obs[i] = proc
            # 解锁新成就且未结束 ⇒ 存档"刚踏入新阶段"那一刻(不存终止/死亡态)。
            if self.state_cache is not None and had_new and not done:
                snap = self.snapshot(i)
                self.state_cache.push(snap, len(snap["unlocked"]))
            obs_list.append(proc)
            rew_list.append(float(rew))
            done_list.append(float(done))
            infos.append(info)

        obs_t = torch.stack(obs_list).to(self.device)
        rew_t = torch.tensor(rew_list, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done_list, dtype=torch.float32, device=self.device)
        return obs_t, rew_t, done_t, infos, new_achievements

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _proc(obs: np.ndarray) -> torch.Tensor:
        """(H, W, C) uint8 → (C, H, W) float32 [0, 1] CPU tensor。"""
        return torch.from_numpy(
            obs.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

    def close(self):
        """串行版无子进程,空操作(与 SubprocVecCrafterEnv 接口对齐)。"""
        pass


# ── 子进程并行向量化(吞吐优化)────────────────────────────────────────────────
def _crafter_worker(remote, n_local, seed0, demo_len, max_history):
    """承载 n_local 个 Crafter env 的工作子进程(spawn 启动,不触碰 CUDA)。

    一个子进程内串行步进 n_local 个 env —— 把每步 IPC 往返从 n_envs 次降到 n_workers 次,
    让每次往返携带 n_local·(~2.6ms) 的真实计算,从而摊薄同步开销(Crafter 单步极廉价时
    一核一 env 会被 IPC 拖死)。多 worker 间由 OS 调度到不同核并行。

    协议:
        ('reset', None)            → 返回 obs uint8 [n_local,H,W,C]。
        ('step', actions:int[n_local]) → 返回 (obs[n_local,H,W,C], rew[n_local], done[n_local],
                                          demos=list[(local_idx, ach, seg_o uint8[T,H,W,C], seg_a)])。
        ('close', None)            → 退出。
    成就检测 / 滑窗历史 / done 重置逐 env 完成,语义与 VecCrafterEnv 串行版逐字对齐。
    """
    import crafter  # 子进程内独立 import(spawn)
    from train.crafter.ad_buffer import ACHIEVEMENTS
    torch.set_num_threads(1)  # 每 worker 单线程,避免 N 个 worker × 多 BLAS 线程超订核

    envs = [crafter.Env() for _ in range(n_local)]
    cur = [e.reset() for e in envs]                      # 各 (H,W,C) uint8
    hist_o = [[] for _ in range(n_local)]
    hist_a = [[] for _ in range(n_local)]
    prev_ach = [{} for _ in range(n_local)]

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs_out = np.empty((n_local, 64, 64, 3), dtype=np.uint8)
            rew_out = np.empty(n_local, dtype=np.float32)
            done_out = np.empty(n_local, dtype=np.float32)
            demos = []
            for j in range(n_local):
                a = int(data[j])
                hist_o[j].append(cur[j])
                hist_a[j].append(a)
                if len(hist_o[j]) > max_history:
                    hist_o[j].pop(0)
                    hist_a[j].pop(0)

                raw_obs, rew, done, info = envs[j].step(a)
                cur_ach = info.get("achievements", {})
                for ach in ACHIEVEMENTS:
                    if cur_ach.get(ach, 0) > 0 and prev_ach[j].get(ach, 0) == 0:
                        seg_o = np.asarray(hist_o[j][-demo_len:], dtype=np.uint8)
                        seg_a = list(hist_a[j][-demo_len:])
                        demos.append((j, ach, seg_o, seg_a))
                prev_ach[j] = dict(cur_ach)

                if done:
                    raw_obs = envs[j].reset()
                    hist_o[j].clear()
                    hist_a[j].clear()
                    prev_ach[j] = {}
                cur[j] = raw_obs
                obs_out[j] = raw_obs
                rew_out[j] = float(rew)
                done_out[j] = 1.0 if done else 0.0
            remote.send((obs_out, rew_out, done_out, demos))
        elif cmd == "reset":
            for j in range(n_local):
                cur[j] = envs[j].reset()
                hist_o[j].clear()
                hist_a[j].clear()
                prev_ach[j] = {}
            remote.send(np.asarray(cur, dtype=np.uint8))
        elif cmd == "close":
            remote.close()
            return


class SubprocVecCrafterEnv:
    """子进程并行的 Crafter 向量环境 —— VecCrafterEnv 的高吞吐替代品。

    n_envs 个 env 分摊到 n_workers 个子进程(每进程 ~n_envs/n_workers 个,各占一核),
    消除单核串行步进的 CPU 墙;每进程承载多 env 以摊薄每步 IPC(Crafter 单步太廉价,
    一核一 env 会被同步开销吃光)。step() 返回签名与 VecCrafterEnv 完全一致(含
    new_achievements),训练循环零改动可换用。obs 以 uint8 批量跨进程传输降低 IPC。

    Args:
        n_envs / device / seed: 同 VecCrafterEnv。
        n_workers:   子进程数;None ⇒ 取 min(n_envs, cpu-2) 的甜点(留核给主进程+GPU)。
        demo_len:    新成就时回传的示范步数(主进程仍可再截断)。
        max_history: 子进程滑窗历史长度。
    """

    OBS_H, OBS_W = 64, 64

    def __init__(self, n_envs: int, device: str = "cuda", seed: int = 0,
                 n_workers: int = None, demo_len: int = 64, max_history: int = 128):
        import multiprocessing as mp
        import os
        # 限制 BLAS/OMP 线程数(spawn 子进程继承 env;须在 numpy import 前生效):
        # N 个 worker 各开多线程会超订 16 核,导致 env 步进被上下文切换串行化。
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(_v, "1")
        self.n_envs = n_envs
        self.device = device
        if n_workers is None:
            n_workers = max(1, min(n_envs, (os.cpu_count() or 4) - 2))
        n_workers = min(n_workers, n_envs)
        self.n_workers = n_workers

        # env 全局索引按 worker 连续切片(np.array_split 处理不整除)。
        splits = np.array_split(np.arange(n_envs), n_workers)
        self._slices = [(int(s[0]), len(s)) for s in splits]   # (global_start, n_local)

        ctx = mp.get_context("spawn")
        self._remotes, self._procs = [], []
        for (start, n_local) in self._slices:
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_crafter_worker,
                args=(child, n_local, seed + start, demo_len, max_history),
                daemon=True,
            )
            p.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(p)

    @staticmethod
    def _batch_to_tensor(raw: np.ndarray) -> torch.Tensor:
        """[N,H,W,C] uint8 → [N,C,H,W] float32 [0,1] CPU tensor。"""
        return torch.from_numpy(
            raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)

    def reset(self):
        for r in self._remotes:
            r.send(("reset", None))
        chunks = [self._batch_to_tensor(r.recv()) for r in self._remotes]
        return torch.cat(chunks, dim=0).to(self.device)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions)
        for r, (start, n_local) in zip(self._remotes, self._slices):
            r.send(("step", actions[start:start + n_local]))

        obs_chunks, rew_chunks, done_chunks = [], [], []
        new_achievements = []
        for r, (start, n_local) in zip(self._remotes, self._slices):
            obs_out, rew_out, done_out, demos = r.recv()
            obs_chunks.append(self._batch_to_tensor(obs_out))
            rew_chunks.append(torch.from_numpy(rew_out))
            done_chunks.append(torch.from_numpy(done_out))
            for local_j, ach, seg_o, seg_a in demos:
                global_idx = start + local_j
                obs_hist = [torch.from_numpy(
                    seg_o[t].transpose(2, 0, 1).astype(np.float32) / 255.0)
                    for t in range(seg_o.shape[0])]
                new_achievements.append((global_idx, ach, obs_hist, seg_a))

        obs_t = torch.cat(obs_chunks, dim=0).to(self.device)
        rew_t = torch.cat(rew_chunks, dim=0).to(self.device)
        done_t = torch.cat(done_chunks, dim=0).to(self.device)
        return obs_t, rew_t, done_t, [{}] * self.n_envs, new_achievements

    def close(self):
        for r in self._remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
