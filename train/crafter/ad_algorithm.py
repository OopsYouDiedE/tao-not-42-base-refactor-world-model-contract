"""PPO + Achievement Distillation 训练算法 (train/crafter/ad_algorithm.py)。

对外接口:
    TrajBuffer       — 跨多次 rollout 缓存段、切分轨迹、抽取成就(goal),
                       产出 intra-traj 预测 与 cross-traj 匹配(最优传输)两类数据流。
    PPOADAlgorithm   — PPG 式双阶段:先若干轮 PPO,再每 aux_freq 次跑辅助蒸馏阶段。

算法见 knowledge/ppo_ad.md。1:1 复现 snu-mllab/Achievement-Distillation 的 algorithm/ppo_ad.py。
跨轨成就匹配用熵正则部分最优传输(POT 的 entropic_partial_wasserstein),仅在辅助损失的
数据装配阶段,不进 forward/rollout —— 合 I6(不稳定组合优化只在损失里)。
"""
import copy
from collections import deque
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from ot.partial import entropic_partial_wasserstein

from net.ppo_ad.model import PPOADModel
from train.crafter.ad_storage import RolloutStorage


class TrajBuffer:
    """缓存最近 maxlen 次 rollout 的段,解析成轨迹并抽取成就(goal)。

    Args:
        maxlen: 缓存的 rollout 段数(= aux_freq)。
        device: 张量目标设备。
    """

    def __init__(self, maxlen: int, device: torch.device):
        self.segs: List[Dict[str, torch.Tensor]] = deque(maxlen=maxlen)
        self.trajs: List[Dict[str, torch.Tensor]] = []
        self.device = device

    def __len__(self):
        return len(self.segs)

    def insert(self, seg: Dict[str, torch.Tensor]):
        self.segs.append(seg)

    def parse_segs(self):
        self.trajs.clear()

        obs = torch.cat([seg["obs"][:-1] for seg in self.segs], dim=0)
        actions = torch.cat([seg["actions"] for seg in self.segs], dim=0)
        states = torch.cat([seg["states"][:-1] for seg in self.segs], dim=0)
        returns = torch.cat([seg["returns"] for seg in self.segs], dim=0)
        masks = torch.cat([seg["masks"][:-1] for seg in self.segs], dim=0)
        rewards = torch.cat([seg["rewards"] for seg in self.segs], dim=0)
        successes = torch.cat([seg["successes"][:-1] for seg in self.segs], dim=0)

        nproc = obs.shape[1]
        for p in range(nproc):
            masks_p = masks[:, p]
            done_conds_p = (masks_p == 0).squeeze(dim=-1)
            done_steps_p = sorted(done_conds_p.nonzero().squeeze(dim=-1).tolist())

            for start, end in zip(done_steps_p[:-1], done_steps_p[1:]):
                self.trajs.append({
                    "obs": obs[start:end, p],
                    "actions": actions[start:end, p],
                    "old_states": states[start:end, p],
                    "old_vtargs": returns[start:end, p],
                    "rewards": rewards[start:end, p],
                    "successes": successes[start:end, p],
                })

    def preprocess_trajs(self):
        for traj in self.trajs:
            goals = self.get_goals(traj["obs"], traj["rewards"], traj["successes"])
            traj.update(goals)

    def get_goals(self, obs, rewards, successes) -> Dict[str, torch.Tensor]:
        # 成就步:由奖励>0.1 检出,并与 successes 变化交叉校验
        goal_steps_r = (rewards[:-1] > 0.1).squeeze(dim=-1).nonzero().squeeze(dim=-1) + 1
        goal_steps_s = (successes[1:] != successes[:-1]).any(dim=-1).nonzero().squeeze(dim=-1) + 1
        assert torch.equal(goal_steps_r, goal_steps_s)
        goal_steps = goal_steps_r

        if len(goal_steps) == 0:
            goal_obs = torch.zeros(0, *obs.shape[1:])
            goal_next_obs = torch.zeros(0, *obs.shape[1:])
        else:
            goal_obs = obs[goal_steps - 1]
            goal_next_obs = obs[goal_steps]

        return {"goal_steps": goal_steps, "goal_obs": goal_obs, "goal_next_obs": goal_next_obs}

    def get_next_goals(self, goal_steps, goal_obs, goal_next_obs, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        next_goal_obs, next_goal_next_obs = [], []
        goal_steps = sorted(set([0] + goal_steps.tolist() + [len(obs)]))

        for i, (start, end) in enumerate(zip(goal_steps[:-1], goal_steps[1:])):
            if i == len(goal_steps) - 2:
                next_goal_ob = obs[-1].unsqueeze(dim=0)
                next_goal_next_ob = torch.zeros_like(obs[-1]).unsqueeze(dim=0)
            else:
                next_goal_ob = goal_obs[i].unsqueeze(dim=0)
                next_goal_next_ob = goal_next_obs[i].unsqueeze(dim=0)
            next_goal_obs.append(next_goal_ob.repeat_interleave(end - start, dim=0))
            next_goal_next_obs.append(next_goal_next_ob.repeat_interleave(end - start, dim=0))

        return torch.cat(next_goal_obs, dim=0), torch.cat(next_goal_next_obs, dim=0)

    def get_pred_data_loader(self, max_batch_size: int = 512) -> Iterator[Dict[str, torch.Tensor]]:
        for i in torch.randperm(len(self.trajs)):
            traj = self.trajs[i]
            obs = traj["obs"]
            actions = traj["actions"]
            old_states = traj["old_states"]
            old_vtargs = traj["old_vtargs"]
            goal_steps = traj["goal_steps"]

            if len(goal_steps) == 0:
                continue

            next_goal_obs, next_goal_next_obs = self.get_next_goals(
                goal_steps, traj["goal_obs"], traj["goal_next_obs"], obs)
            assert len(obs) == len(next_goal_obs)

            ndata = len(obs)
            rand_steps = torch.randperm(ndata)
            sampler = BatchSampler(SubsetRandomSampler(range(ndata)), batch_size=max_batch_size, drop_last=False)

            for inds in sampler:
                rinds = rand_steps[inds]
                yield {
                    "anc_goal_obs": next_goal_obs[inds].to(self.device),
                    "anc_goal_next_obs": next_goal_next_obs[inds].to(self.device),
                    "pos_obs": obs[inds].to(self.device),
                    "pos_actions": actions[inds].to(self.device),
                    "pos_old_states": old_states[inds].to(self.device),
                    "pos_old_vtargs": old_vtargs[inds].to(self.device),
                    "neg_obs": obs[rinds].to(self.device),
                    "neg_actions": actions[rinds].to(self.device),
                    "neg_old_states": old_states[rinds].to(self.device),
                    "neg_old_vtargs": old_vtargs[rinds].to(self.device),
                }

    def get_match_data_loader(self, model: PPOADModel, max_batch_size: int = 512) -> Iterator[Dict[str, torch.Tensor]]:
        trajs = [traj for traj in self.trajs if len(traj["goal_steps"]) > 0]
        ntraj = len(trajs)

        for i in torch.randperm(ntraj):
            traj_s = trajs[i]
            obs_s = traj_s["obs"]
            old_states_s = traj_s["old_states"]
            old_vtargs_s = traj_s["old_vtargs"]
            goal_obs_s = traj_s["goal_obs"]
            goal_next_obs_s = traj_s["goal_next_obs"]

            with torch.no_grad():
                states_s = model.get_states(goal_obs_s.to(self.device), goal_next_obs_s.to(self.device))

            anc_goal_obs, anc_goal_next_obs = [], []
            pos_goal_obs, pos_goal_next_obs = [], []
            neg_goal_obs, neg_goal_next_obs = [], []

            inds = torch.randperm(ntraj - 1)[:16]
            for j in inds:
                if j >= i:
                    j += 1
                traj_t = trajs[j]
                goal_obs_t = traj_t["goal_obs"]
                goal_next_obs_t = traj_t["goal_next_obs"]

                with torch.no_grad():
                    states_t = model.get_states(goal_obs_t.to(self.device), goal_next_obs_t.to(self.device))

                a = np.ones(len(states_s))
                b = np.ones(len(states_t))
                M = 1 - torch.einsum("ik,jk->ij", states_s, states_t).cpu().numpy()
                T = entropic_partial_wasserstein(a, b, M, reg=0.05, numItermax=100)
                T = torch.from_numpy(T).float()
                row_inds, col_inds = torch.where(T > 0.5)
                if len(row_inds) == 0:
                    continue

                anc_goal_obs.append(goal_obs_s[row_inds])
                anc_goal_next_obs.append(goal_next_obs_s[row_inds])
                pos_goal_obs.append(goal_obs_t[col_inds])
                pos_goal_next_obs.append(goal_next_obs_t[col_inds])
                rand_inds = torch.randint(len(goal_obs_t), (len(col_inds),))
                neg_goal_obs.append(goal_obs_t[rand_inds])
                neg_goal_next_obs.append(goal_next_obs_t[rand_inds])

            if len(anc_goal_obs) == 0:
                continue

            anc_goal_obs = torch.cat(anc_goal_obs, dim=0)
            anc_goal_next_obs = torch.cat(anc_goal_next_obs, dim=0)
            pos_goal_obs = torch.cat(pos_goal_obs, dim=0)
            pos_goal_next_obs = torch.cat(pos_goal_next_obs, dim=0)
            neg_goal_obs = torch.cat(neg_goal_obs, dim=0)
            neg_goal_next_obs = torch.cat(neg_goal_next_obs, dim=0)

            ndata = len(anc_goal_obs)
            sampler = BatchSampler(SubsetRandomSampler(range(ndata)), batch_size=max_batch_size, drop_last=False)
            rand_inds = torch.randint(len(obs_s), (ndata,))
            obs = obs_s[rand_inds]
            old_states = old_states_s[rand_inds]
            old_vtargs = old_vtargs_s[rand_inds]

            for inds2 in sampler:
                yield {
                    "anc_goal_obs": anc_goal_obs[inds2].to(self.device),
                    "anc_goal_next_obs": anc_goal_next_obs[inds2].to(self.device),
                    "pos_goal_obs": pos_goal_obs[inds2].to(self.device),
                    "pos_goal_next_obs": pos_goal_next_obs[inds2].to(self.device),
                    "neg_goal_obs": neg_goal_obs[inds2].to(self.device),
                    "neg_goal_next_obs": neg_goal_next_obs[inds2].to(self.device),
                    "obs": obs[inds2].to(self.device),
                    "old_states": old_states[inds2].to(self.device),
                    "old_vtargs": old_vtargs[inds2].to(self.device),
                }


class PPOADAlgorithm:
    """PPG 式双阶段 PPO+AD 训练器。

    Args:
        model: PPOADModel。
        ppo_nepoch/ppo_nbatch/clip_param/vf_loss_coef/ent_coef/lr/max_grad_norm: PPO 阶段。
        aux_freq/aux_nepoch/pi_dist_coef/vf_dist_coef: 辅助蒸馏阶段。
        device: 设备。
    """

    def __init__(self, model: PPOADModel, ppo_nepoch, ppo_nbatch, clip_param,
                 vf_loss_coef, ent_coef, lr, max_grad_norm, aux_freq, aux_nepoch,
                 pi_dist_coef, vf_dist_coef, device):
        self.model = model
        self.ppo_nepoch = ppo_nepoch
        self.ppo_nbatch = ppo_nbatch
        self.clip_param = clip_param
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_count = 0

        self.aux_freq = aux_freq
        self.aux_nepoch = aux_nepoch
        self.pi_dist_coef = pi_dist_coef
        self.vf_dist_coef = vf_dist_coef

        self.buffer = TrajBuffer(maxlen=aux_freq, device=device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.match_optimizer = optim.Adam(model.parameters(), lr=lr)
        self.pred_optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage) -> Dict[str, float]:
        self.model.train()

        keys = ["obs", "actions", "states", "returns", "masks", "rewards", "successes"]
        self.buffer.insert({key: storage[key].cpu() for key in keys})

        # ── PPO 阶段 ──────────────────────────────────────────────────────────
        pi_loss_epoch = vf_loss_epoch = entropy_epoch = 0.0
        nupdate = 0
        for _ in range(self.ppo_nepoch):
            for batch in storage.get_data_loader(self.ppo_nbatch):
                losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                loss = (losses["pi_loss"] + self.vf_loss_coef * losses["vf_loss"]
                        - self.ent_coef * losses["entropy"])
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                pi_loss_epoch += losses["pi_loss"].item()
                vf_loss_epoch += losses["vf_loss"].item()
                entropy_epoch += losses["entropy"].item()
                nupdate += 1

        train_stats = {
            "pi_loss": pi_loss_epoch / nupdate,
            "vf_loss": vf_loss_epoch / nupdate,
            "entropy": entropy_epoch / nupdate,
        }
        self.ppo_count += 1

        # ── 辅助蒸馏阶段(每 aux_freq 次)────────────────────────────────────────
        if self.ppo_count % self.aux_freq == 0:
            self.buffer.parse_segs()
            self.buffer.preprocess_trajs()

            old_model = copy.deepcopy(self.model)
            old_model.eval()

            match_loss_epoch = pred_loss_epoch = pi_dist_epoch = vf_dist_epoch = 0.0
            match_nupdate = pred_nupdate = 0

            for _ in range(self.aux_nepoch):
                for batch in self.buffer.get_match_data_loader(self.model):
                    losses = self.model.compute_match_losses(**batch, old_model=old_model)
                    loss = (losses["match_loss"] + self.pi_dist_coef * losses["pi_dist"]
                            + self.vf_dist_coef * losses["vf_dist"])
                    self.match_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.match_optimizer.step()
                    match_loss_epoch += losses["match_loss"].item()
                    pi_dist_epoch += losses["pi_dist"].item()
                    vf_dist_epoch += losses["vf_dist"].item()
                    match_nupdate += 1

                for batch in self.buffer.get_pred_data_loader():
                    losses = self.model.compute_pred_losses(**batch, old_model=old_model)
                    loss = (losses["pred_loss"] + self.pi_dist_coef * losses["pi_dist"]
                            + self.vf_dist_coef * losses["vf_dist"])
                    self.pred_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.pred_optimizer.step()
                    pred_loss_epoch += losses["pred_loss"].item()
                    pi_dist_epoch += losses["pi_dist"].item()
                    vf_dist_epoch += losses["vf_dist"].item()
                    pred_nupdate += 1

            train_stats.update({
                "match_loss": match_loss_epoch / max(match_nupdate, 1),
                "pred_loss": pred_loss_epoch / max(pred_nupdate, 1),
                "pi_dist": pi_dist_epoch / max(match_nupdate + pred_nupdate, 1),
                "vf_dist": vf_dist_epoch / max(match_nupdate + pred_nupdate, 1),
            })

        return train_stats
