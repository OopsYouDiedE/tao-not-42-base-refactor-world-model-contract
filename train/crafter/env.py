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
