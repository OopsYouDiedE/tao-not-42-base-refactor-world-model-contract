"""DreamerV3 想象式 actor-critic (net/dreamerv3/behavior.py)。

对外接口:
    ImagBehavior — 在世界模型想象轨迹上训练的离散策略 + two-hot 价值,含慢靶 critic。

从 blocks 组装:策略/价值头用 blocks.MLP,动作分布用 blocks.OneHotDist(unimix + 直通梯度),
价值分布用 blocks.DiscDist(two-hot symexp),λ-return 用 blocks.lambda_return,
想象 rollout 用 blocks.static_scan。回报标准化用 RewardEMA(5/95 分位)。设计见 [knowledge/dreamer.md]。
"""
import copy

import torch
import torch.nn as nn

from blocks import MLP, OneHotDist, DiscDist, lambda_return, static_scan
from net.dreamerv3.config import DreamerV3Config


class RewardEMA:
    """想象回报的 5%/95% 分位 EMA,用于把优势归一到稳定尺度(scale ≥ 1)。"""

    def __init__(self, alpha=1e-2):
        self.alpha = alpha

    def __call__(self, x, ema_vals):
        flat = x.detach().flatten()
        q = torch.quantile(
            flat, torch.tensor([0.05, 0.95], device=x.device, dtype=flat.dtype))
        ema_vals.mul_(1.0 - self.alpha).add_(self.alpha * q)
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)   # I1:尺度下界 1
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class ImagBehavior(nn.Module):
    """想象 actor-critic。

    Args:
        cfg: DreamerV3Config(读 horizon/discount/disc_lambda/actor_entropy/
             value_decay/units/mlp_layers/reward_bins/num_actions)。
        feat_dim: 世界状态特征维(= RSSM stoch_flat + deter)。
    """

    def __init__(self, cfg: DreamerV3Config, feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.actor = MLP(feat_dim, cfg.num_actions, hidden=cfg.units,
                         layers=cfg.mlp_layers)
        self.value = MLP(feat_dim, cfg.reward_bins, hidden=cfg.units,
                         layers=cfg.mlp_layers)
        self.slow_value = copy.deepcopy(self.value)
        for p in self.slow_value.parameters():
            p.requires_grad_(False)
        self.reward_ema = RewardEMA()
        self.register_buffer("ema_vals", torch.zeros(2))

    # ── 分布封装 ────────────────────────────────────────────────────────────
    def actor_dist(self, feat):
        """feat → 离散动作 OneHot 分布(unimix + 直通梯度)。"""
        return OneHotDist(self.actor(feat).float(), unimix_ratio=self.cfg.unimix_ratio)

    def value_dist(self, feat):
        """feat → two-hot symexp 价值分布(在线 critic)。"""
        logits = self.value(feat)
        return DiscDist(logits, device=logits.device)

    def slow_value_dist(self, feat):
        """feat → 慢靶 critic 价值分布(EMA 副本,作正则锚)。"""
        logits = self.slow_value(feat)
        return DiscDist(logits, device=logits.device)

    # ── 想象 ────────────────────────────────────────────────────────────────
    def imagine(self, start, dynamics):
        """从后验起点状态做 H 步先验 rollout(动作由当前 actor 采样)。

        Args:
            start: 后验状态 dict,字段首维 B、次维 T;内部展平成 B·T 条想象起点。
            dynamics: RSSM(提供 get_feat / img_step)。

        Returns:
            feats:   [H, B·T, feat_dim] 各步状态特征。
            actions: [H, B·T, num_actions] one-hot 动作。
        """
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        state = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            st = prev[0]
            feat = dynamics.get_feat(st)
            action = self.actor_dist(feat.detach()).sample()
            succ = dynamics.img_step(st, action)
            return succ, feat, action

        idx = torch.arange(self.cfg.horizon, device=state["deter"].device)
        _, feats, actions = static_scan(step, (idx,), (state, None, None))
        return feats, actions

    # ── 损失 ──────────────────────────────────────────────────────────────────
    def loss(self, start, world_model):
        """想象轨迹上的 actor + critic 损失(世界模型不接收梯度)。

        Returns:
            actor_loss, value_loss: 标量。
            metrics: dict[str, float]。
        """
        dynamics = world_model.dynamics
        # 想象 rollout 与回报目标全程 no_grad:reinforce 不需要沿 rollout 的路径梯度
        # (actor 仅由显式 logπ·advantage 接收梯度,critic 在 detach 特征上回归),
        # 故切断 rollout 图既省显存又不改学习信号。
        with torch.no_grad():
            feats, actions = self.imagine(start, dynamics)
            reward = world_model.reward_dist(feats).mode()              # [H, N, 1]
            cont = world_model.cont_dist(feats).mean                    # [H, N, 1] ∈(0,1)
            discount = self.cfg.discount * cont
            value = self.value_dist(feats).mode()                       # [H, N, 1]
            target = torch.stack(lambda_return(
                reward[:-1], value[:-1], discount[:-1], bootstrap=value[-1],
                lambda_=self.cfg.disc_lambda, axis=0), dim=1)           # [H-1, N, 1]
            weights = torch.cumprod(
                torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0)
            offset, scale = self.reward_ema(target, self.ema_vals)
            adv = ((target - offset) / scale) - ((value[:-1] - offset) / scale)
        policy = self.actor_dist(feats.detach())
        logpi = policy.log_prob(actions.detach())[:-1].unsqueeze(-1)    # [H-1, N, 1]
        entropy = policy.entropy()[:-1].unsqueeze(-1)
        actor_loss = -(logpi * adv.detach() + self.cfg.actor_entropy * entropy)
        actor_loss = (weights[:-1] * actor_loss).mean()

        # critic(two-hot 回归 λ-return + 慢靶正则)
        value_dist = self.value_dist(feats[:-1].detach())
        with torch.no_grad():
            slow_target = self.slow_value_dist(feats[:-1].detach()).mode()
        value_loss = -value_dist.log_prob(target.detach())             # [H-1, N]
        value_loss = value_loss - value_dist.log_prob(slow_target.detach())
        value_loss = (weights[:-1].squeeze(-1) * value_loss).mean()

        metrics = {
            "actor": actor_loss.item(),
            "value": value_loss.item(),
            "entropy": entropy.mean().item(),
            "imag_reward": reward.mean().item(),
            "return_scale": scale.item(),
        }
        return actor_loss, value_loss, metrics

    def update_slow(self):
        """慢靶 critic 向在线 critic 做一步 EMA 混合。"""
        mix = self.cfg.value_decay
        for s, d in zip(self.value.parameters(), self.slow_value.parameters()):
            d.data.mul_(1.0 - mix).add_(mix * s.data)
