"""PPO + Achievement Distillation 模型 (net/ppo_ad/model.py)。

对外接口:
    PPOADModel — IMPALA 编码器 + memory 状态 + FiLM 动作调制 + 成就表示,
                 含 PPO 损失与两类对比蒸馏损失(intra-traj 预测 / cross-traj 匹配)。

算法见 knowledge/ppo_ad.md。结构 1:1 复现 snu-mllab/Achievement-Distillation(NeurIPS 2023):
    - 成就表示 state = normalize(enc(goal_next) − enc(goal));rollout 中每解锁新成就即更新 memory。
    - intra-traj:FiLM(latent, action) → MLP 预测 next-goal 表示,InfoNCE(正=真动作 / 负=随机动作)。
    - cross-traj:用最优传输匹配跨轨成就表示后做 InfoNCE(匹配在 train 端损失里算,见 I6)。
数值约定:get_states / next_goal_preds 末尾 L2 normalize(I4);温度 0.1。
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import (ImpalaCNN, FanInInitReLULayer, FanInMLP,
                    CategoricalActionHead, ScaledMSEHead)


class PPOADModel(nn.Module):
    """PPO+AD 主模型。

    Args:
        obs_shape:   观测形状 (C, H, W)。
        num_actions: 离散动作数。
        hidsize:     编码器投影维度(latent 与 memory state 维度)。
        impala_kwargs:          传给 ImpalaCNN(chans/outsize/nblock/...)。
        init_norm_kwargs:       卷积归一化(batch_norm=False, group_norm_groups=1)。
        dense_init_norm_kwargs: dense/MLP 归一化(layer_norm=True)。
        action_head_kwargs / mse_head_kwargs: 头初始化配置。
        nhidlayer:   AD MLP 隐藏层数。
        temperature: InfoNCE 温度。
        use_memory:  是否把 memory state 拼到 latent 供 π/V(默认 True)。

    主要张量契约:
        obs:    (B, C, H, W) float32 [0,1]
        states: (B, hidsize) float32  —— 当前成就 memory(rollout 维护)
        actions:(B, 1) long
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        hidsize: int,
        impala_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        nhidlayer: int = 1,
        temperature: float = 0.1,
        use_memory: bool = True,
    ):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.num_actions = num_actions
        self.hidsize = hidsize
        self.use_memory = use_memory
        self.temperature = temperature

        # 编码器:IMPALA 卷积塔 → dense outsize → linear hidsize
        self.enc = ImpalaCNN(
            self.obs_shape,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        outsize = impala_kwargs["outsize"]
        self.linear = FanInInitReLULayer(
            outsize, hidsize, layer_type="linear", **dense_init_norm_kwargs
        )

        # 头:memory 拼接时输入维度翻倍
        head_size = 2 * hidsize if use_memory else hidsize
        self.pi_head = CategoricalActionHead(
            insize=head_size, num_actions=num_actions, **action_head_kwargs
        )
        self.vf_head = ScaledMSEHead(insize=head_size, outsize=1, **mse_head_kwargs)

        # AD 层:动作调制(FiLM)+ next-goal 预测
        self.action_mlp = FanInMLP(
            insize=num_actions, nhidlayer=nhidlayer, outsize=2 * hidsize,
            hidsize=hidsize, dense_init_norm_kwargs=dense_init_norm_kwargs,
        )
        self.next_goal_pred_mlp = FanInMLP(
            insize=2 * hidsize, nhidlayer=nhidlayer, outsize=hidsize,
            hidsize=hidsize, dense_init_norm_kwargs=dense_init_norm_kwargs,
        )

    # ── 编码 / 前向 ─────────────────────────────────────────────────────────────
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.enc(obs))

    def forward(self, obs: torch.Tensor, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        latents = self.encode(obs)
        if self.use_memory:
            pi_latents = vf_latents = torch.cat([latents, states], dim=-1)
        else:
            pi_latents = vf_latents = latents
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)
        return {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }

    @torch.no_grad()
    def act(self, obs: torch.Tensor, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """采样动作 + 反归一化价值(eval 态)。"""
        assert not self.training
        outputs = self.forward(obs, states)
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)
        vpreds = self.vf_head.denormalize(outputs["vpreds"])
        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})
        return outputs

    # ── 成就表示 / FiLM / next-goal 预测 ────────────────────────────────────────
    def film(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """用动作的 FiLM 调制 latent:(1+γ)·latent + β。"""
        onehot = torch.eye(self.num_actions, device=actions.device)[actions.squeeze(dim=-1)]
        action_latents = self.action_mlp(onehot)
        gamma, beta = torch.chunk(action_latents, 2, dim=-1)
        return (1 + gamma) * latents + beta

    def get_states(self, goal_obs: torch.Tensor, goal_next_obs: torch.Tensor) -> torch.Tensor:
        """成就表示 = normalize(enc(goal_next) − enc(goal));全零 obs 视作"无成就"置零。"""
        bsz = goal_obs.shape[0]
        zero_obs = (goal_obs.reshape(bsz, -1) == 0).all(dim=-1, keepdim=True)
        zero_next = (goal_next_obs.reshape(bsz, -1) == 0).all(dim=-1, keepdim=True)
        goal_latents = self.encode(goal_obs)
        goal_next_latents = self.encode(goal_next_obs)
        goal_latents = torch.where(zero_obs, 0, goal_latents)
        goal_next_latents = torch.where(zero_next, 0, goal_next_latents)
        states = goal_next_latents - goal_latents
        return F.normalize(states, dim=-1)

    def get_next_goal_preds(
        self, latents: torch.Tensor, actions: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        latents = self.film(latents, actions)
        if self.use_memory:
            latents = torch.cat([latents, states], dim=-1)
        preds = self.next_goal_pred_mlp(latents)
        return F.normalize(preds, dim=-1)

    # ── PPO 损失 ───────────────────────────────────────────────────────────────
    def compute_losses(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        vtargs: torch.Tensor,
        advs: torch.Tensor,
        clip_param: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(obs, states)
        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = torch.exp(new_log_probs - log_probs)
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -torch.min(advs * ratio, advs * ratio_clipped).mean()
        entropy = self.pi_head.entropy(pi_logits).mean()
        vf_loss = self.vf_head.mse_loss(outputs["vpreds"], vtargs).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

    # ── intra-trajectory 预测损失 ───────────────────────────────────────────────
    def compute_pred_losses(
        self,
        anc_goal_obs: torch.Tensor,
        anc_goal_next_obs: torch.Tensor,
        pos_obs: torch.Tensor,
        pos_actions: torch.Tensor,
        pos_old_states: torch.Tensor,
        pos_old_vtargs: torch.Tensor,
        neg_obs: torch.Tensor,
        neg_actions: torch.Tensor,
        neg_old_states: torch.Tensor,
        neg_old_vtargs: torch.Tensor,
        old_model: "PPOADModel",
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            anc_states = self.get_states(anc_goal_obs, anc_goal_next_obs)

        pos_outputs = self.forward(pos_obs, states=pos_old_states)
        pos_old_outputs = old_model.act(pos_obs, states=pos_old_states)
        pos_preds = self.get_next_goal_preds(pos_outputs["latents"], pos_actions, pos_old_states)

        neg_outputs = self.forward(neg_obs, states=neg_old_states)
        neg_old_outputs = old_model.act(neg_obs, states=neg_old_states)
        neg_preds = self.get_next_goal_preds(neg_outputs["latents"], neg_actions, neg_old_states)

        pos_logits = torch.einsum("bk,bk->b", anc_states, pos_preds)
        neg_logits = torch.einsum("bk,bk->b", anc_states, neg_preds)
        logits = torch.stack([pos_logits, neg_logits], dim=-1) / self.temperature
        targets = torch.zeros(len(logits), device=logits.device).long()
        pred_loss = F.cross_entropy(logits, targets)

        pi_logits = torch.cat([pos_outputs["pi_logits"], neg_outputs["pi_logits"]], dim=0)
        old_pi_logits = torch.cat([pos_old_outputs["pi_logits"], neg_old_outputs["pi_logits"]], dim=0)
        pi_dist = self.pi_head.kl_divergence(pi_logits, old_pi_logits).mean()

        vpreds = torch.cat([pos_outputs["vpreds"], neg_outputs["vpreds"]], dim=0)
        old_vtargs = torch.cat([pos_old_vtargs, neg_old_vtargs], dim=0)
        vf_dist = self.vf_head.mse_loss(vpreds, old_vtargs).mean()

        return {"pred_loss": pred_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}

    # ── cross-trajectory 匹配损失 ───────────────────────────────────────────────
    def compute_match_losses(
        self,
        anc_goal_obs: torch.Tensor,
        anc_goal_next_obs: torch.Tensor,
        pos_goal_obs: torch.Tensor,
        pos_goal_next_obs: torch.Tensor,
        neg_goal_obs: torch.Tensor,
        neg_goal_next_obs: torch.Tensor,
        obs: torch.Tensor,
        old_states: torch.Tensor,
        old_vtargs: torch.Tensor,
        old_model: "PPOADModel",
    ) -> Dict[str, torch.Tensor]:
        anc_states = self.get_states(anc_goal_obs, anc_goal_next_obs)
        with torch.no_grad():
            pos_states = self.get_states(pos_goal_obs, pos_goal_next_obs)
            neg_states = self.get_states(neg_goal_obs, neg_goal_next_obs)

        outputs = self.forward(obs, states=old_states)
        old_outputs = old_model.act(obs, states=old_states)

        pos_logits = torch.einsum("bk,bk->b", anc_states, pos_states)
        neg_logits = torch.einsum("bk,bk->b", anc_states, neg_states)
        logits = torch.stack([pos_logits, neg_logits], dim=-1) / self.temperature
        targets = torch.zeros(len(logits), device=logits.device).long()
        match_loss = F.cross_entropy(logits, targets)

        pi_dist = self.pi_head.kl_divergence(outputs["pi_logits"], old_outputs["pi_logits"]).mean()
        vf_dist = self.vf_head.mse_loss(outputs["vpreds"], old_vtargs).mean()

        return {"match_loss": match_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}

    def get_initial_states(self, nproc: int, device: torch.device) -> torch.Tensor:
        """rollout 起始 memory state(零向量)。(nproc, hidsize)。"""
        return torch.zeros(nproc, self.hidsize, device=device)
