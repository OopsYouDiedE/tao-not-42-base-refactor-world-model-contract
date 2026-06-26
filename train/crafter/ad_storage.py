"""PPO+AD on-policy rollout 存储 (train/crafter/ad_storage.py)。

对外接口:
    RolloutStorage — 收集 nstep×nproc 转移,维护成就 memory 状态,算 GAE,迭代 PPO minibatch。

memory 语义(承官方 AD):每当某 env 在某步解锁新成就(successes 改变),把 memory state 更新为
normalize(enc(next_obs) − enc(cur_obs));done 时 successes/timesteps/state 清零。
结构 1:1 复现 snu-mllab/Achievement-Distillation 的 storage.py。
"""
from typing import Dict, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from train.crafter.ad_env import N_TASKS


class RolloutStorage:
    """on-policy 轨迹存储。

    Args:
        nstep:   每 env 步数。
        nproc:   并行环境数。
        obs_shape: (C, H, W)。
        hidsize: memory state 维度。
        device:  存储设备。
    """

    def __init__(self, nstep, nproc, obs_shape, hidsize, device):
        self.nstep = nstep
        self.nproc = nproc
        self.device = device

        self.obs = torch.zeros(nstep + 1, nproc, *obs_shape, device=device)
        self.actions = torch.zeros(nstep, nproc, 1, device=device).long()
        self.rewards = torch.zeros(nstep, nproc, 1, device=device)
        self.masks = torch.ones(nstep + 1, nproc, 1, device=device)
        self.vpreds = torch.zeros(nstep + 1, nproc, 1, device=device)
        self.log_probs = torch.zeros(nstep, nproc, 1, device=device)
        self.returns = torch.zeros(nstep, nproc, 1, device=device)
        self.advs = torch.zeros(nstep, nproc, 1, device=device)
        self.successes = torch.zeros(nstep + 1, nproc, N_TASKS, device=device).long()
        self.timesteps = torch.zeros(nstep + 1, nproc, 1, device=device).long()
        self.states = torch.zeros(nstep + 1, nproc, hidsize, device=device)

        self.step = 0

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    def get_inputs(self, step: int) -> Dict[str, torch.Tensor]:
        return {"obs": self.obs[step], "states": self.states[step]}

    def insert(self, obs, latents, actions, rewards, masks, vpreds, log_probs, successes, model, **kwargs):
        prev_successes = self.successes[self.step]
        prev_states = self.states[self.step]
        prev_timesteps = self.timesteps[self.step]

        timesteps = prev_timesteps + 1

        # 解锁新成就 → 更新 memory state
        success_conds = (successes != prev_successes).any(dim=-1, keepdim=True)
        if success_conds.any():
            with torch.no_grad():
                next_latents = model.encode(obs)
            states = F.normalize(next_latents - latents, dim=-1)
            states = torch.where(success_conds, states, prev_states)
        else:
            states = prev_states

        # done 清零
        done_conds = masks == 0
        successes = torch.where(done_conds, 0, successes)
        timesteps = torch.where(done_conds, 0, timesteps)
        states = torch.where(done_conds, 0, states)

        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.vpreds[self.step].copy_(vpreds)
        self.log_probs[self.step].copy_(log_probs)
        self.successes[self.step + 1].copy_(successes)
        self.timesteps[self.step + 1].copy_(timesteps)
        self.states[self.step + 1].copy_(states)

        self.step = (self.step + 1) % self.nstep

    def reset(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.successes[0].copy_(self.successes[-1])
        self.timesteps[0].copy_(self.timesteps[-1])
        self.states[0].copy_(self.states[-1])
        self.step = 0

    def compute_returns(self, gamma: float, gae_lambda: float):
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = (
                self.rewards[step]
                + gamma * self.vpreds[step + 1] * self.masks[step + 1]
                - self.vpreds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.vpreds[step]
            self.advs[step] = gae
        # I1: ε=1e-8 沿用官方(优势标准化非危险除法,不触发 I1 的 1e-4 下界要求)
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def get_data_loader(self, nbatch: int) -> Iterator[Dict[str, torch.Tensor]]:
        ndata = self.nstep * self.nproc
        assert ndata >= nbatch
        batch_size = ndata // nbatch
        sampler = BatchSampler(SubsetRandomSampler(range(ndata)), batch_size=batch_size, drop_last=True)

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        states = self.states[:-1].reshape(-1, *self.states.shape[2:])
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        vtargs = self.returns.reshape(-1, *self.returns.shape[2:])
        log_probs = self.log_probs.reshape(-1, *self.log_probs.shape[2:])
        advs = self.advs.reshape(-1, *self.advs.shape[2:])

        for inds in sampler:
            yield {
                "obs": obs[inds],
                "states": states[inds],
                "actions": actions[inds],
                "vtargs": vtargs[inds],
                "log_probs": log_probs[inds],
                "advs": advs[inds],
            }
