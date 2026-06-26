"""官方 AD 用的精简子进程向量化 Crafter 环境 (train/crafter/ad_env.py)。

对外接口:
    TASKS         — Crafter 22 成就名(固定顺序,= ad_buffer.ACHIEVEMENTS)。
    ADVecCrafterEnv — 子进程并行 Crafter,step 返回 (obs, rewards, dones, infos),
                      infos 含每步 22 维 achievements 计数 / successes、episode 长度与回报。

与 VecCrafterEnv(DreamerV3 + BC 版共用)分开:官方 AD 不需要示范滑窗历史 / Go-Explore,
只需逐步 successes 向量(供 memory 更新与 score 统计),故另起精简实现。
obs: uint8 (H,W,C) → float32 (C,H,W) [0,1]。接口对齐 snu-mllab/Achievement-Distillation 的 VecPyTorch。
"""
import os

import numpy as np
import torch

from train.crafter.ad_buffer import ACHIEVEMENTS as TASKS

N_TASKS = len(TASKS)


def _worker(remote, n_local, seed0):
    """承载 n_local 个 Crafter env 的子进程(spawn,不触碰 CUDA)。

    协议:
        ('reset', None) → obs uint8 [n_local,H,W,C]。
        ('step', actions int[n_local]) → (obs[n_local,H,W,C] uint8, rew[n_local] f32,
            done[n_local] f32, ach[n_local,22] i32, eplen[n_local] i32, eprew[n_local] f32)。
            ach 为步后(reset 前)的 episode 累计成就计数;eplen/eprew 仅在 done 处有效(否则 -1)。
        ('close', None) → 退出。
    """
    import crafter
    torch.set_num_threads(1)

    envs = [crafter.Env(seed=int(seed0) + j) for j in range(n_local)]
    cur = [e.reset() for e in envs]
    ep_len = [0 for _ in range(n_local)]
    ep_rew = [0.0 for _ in range(n_local)]

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs_out = np.empty((n_local, 64, 64, 3), dtype=np.uint8)
            rew_out = np.empty(n_local, dtype=np.float32)
            done_out = np.empty(n_local, dtype=np.float32)
            ach_out = np.zeros((n_local, N_TASKS), dtype=np.int32)
            eplen_out = np.full(n_local, -1, dtype=np.int32)
            eprew_out = np.full(n_local, -1.0, dtype=np.float32)

            for j in range(n_local):
                a = int(data[j])
                raw_obs, rew, done, info = envs[j].step(a)
                ep_len[j] += 1
                ep_rew[j] += float(rew)

                cur_ach = info.get("achievements", {})
                ach_out[j] = [int(cur_ach.get(t, 0)) for t in TASKS]

                if done:
                    eplen_out[j] = ep_len[j]
                    eprew_out[j] = ep_rew[j]
                    raw_obs = envs[j].reset()
                    ep_len[j] = 0
                    ep_rew[j] = 0.0

                cur[j] = raw_obs
                obs_out[j] = raw_obs
                rew_out[j] = float(rew)
                done_out[j] = 1.0 if done else 0.0

            remote.send((obs_out, rew_out, done_out, ach_out, eplen_out, eprew_out))
        elif cmd == "reset":
            for j in range(n_local):
                cur[j] = envs[j].reset()
                ep_len[j] = 0
                ep_rew[j] = 0.0
            remote.send(np.asarray(cur, dtype=np.uint8))
        elif cmd == "close":
            remote.close()
            return


class ADVecCrafterEnv:
    """子进程并行 Crafter 向量环境(官方 AD 接口)。

    Args:
        nproc:     并行环境数。
        device:    obs/张量目标设备。
        seed:      第 i 个 env 用 seed+i 初始化。
        n_workers: 子进程数;None ⇒ min(nproc, cpu-2)。

    step 返回:
        obs:     (nproc, 3, 64, 64) float32 [0,1] on device。
        rewards: (nproc, 1) float32 on device。
        dones:   (nproc, 1) float32 on device。
        infos:   dict —
            "achievements":   (nproc, 22) long on device(episode 累计计数)。
            "successes":      (nproc, 22) long on device((achievements>0))。
            "dones":          (nproc,) bool on device。
            "episode_lengths":(nproc,) long on device(仅 done 处有效,否则 -1)。
            "episode_rewards":(nproc,) float32 on device(仅 done 处有效,否则 -1)。
    """

    def __init__(self, nproc: int, device: str = "cuda", seed: int = 0, n_workers: int = None):
        import multiprocessing as mp
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(_v, "1")
        self.nproc = nproc
        self.device = device
        if n_workers is None:
            n_workers = max(1, min(nproc, (os.cpu_count() or 4) - 2))
        n_workers = min(n_workers, nproc)
        self.n_workers = n_workers

        splits = np.array_split(np.arange(nproc), n_workers)
        self._slices = [(int(s[0]), len(s)) for s in splits]

        ctx = mp.get_context("spawn")
        self._remotes, self._procs = [], []
        for (start, n_local) in self._slices:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, n_local, seed + start), daemon=True)
            p.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(p)

    @staticmethod
    def _to_obs(raw: np.ndarray) -> torch.Tensor:
        """[N,H,W,C] uint8 → [N,C,H,W] float32 [0,1] CPU tensor。"""
        return torch.from_numpy(raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)

    def reset(self) -> torch.Tensor:
        for r in self._remotes:
            r.send(("reset", None))
        chunks = [self._to_obs(r.recv()) for r in self._remotes]
        return torch.cat(chunks, dim=0).to(self.device)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(-1)
        for r, (start, n_local) in zip(self._remotes, self._slices):
            r.send(("step", actions[start:start + n_local]))

        obs_c, rew_c, done_c, ach_c, eplen_c, eprew_c = [], [], [], [], [], []
        for r, _ in zip(self._remotes, self._slices):
            obs_out, rew_out, done_out, ach_out, eplen_out, eprew_out = r.recv()
            obs_c.append(self._to_obs(obs_out))
            rew_c.append(torch.from_numpy(rew_out))
            done_c.append(torch.from_numpy(done_out))
            ach_c.append(torch.from_numpy(ach_out))
            eplen_c.append(torch.from_numpy(eplen_out))
            eprew_c.append(torch.from_numpy(eprew_out))

        obs = torch.cat(obs_c, dim=0).to(self.device)
        rewards = torch.cat(rew_c, dim=0).unsqueeze(-1).to(self.device)
        dones = torch.cat(done_c, dim=0).unsqueeze(-1).to(self.device)
        achievements = torch.cat(ach_c, dim=0).long().to(self.device)
        successes = (achievements > 0).long()
        episode_lengths = torch.cat(eplen_c, dim=0).long().to(self.device)
        episode_rewards = torch.cat(eprew_c, dim=0).to(self.device)

        infos = {
            "achievements": achievements,
            "successes": successes,
            "dones": dones.squeeze(-1).bool(),
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
        }
        return obs, rewards, dones, infos

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
