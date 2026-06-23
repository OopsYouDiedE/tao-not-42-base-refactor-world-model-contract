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
import torch.nn.functional as F

from blocks import MLP, OneHotDist, DiscDist, lambda_return, static_scan
from net.dreamerv3.config import DreamerV3Config


class GoalActorHead(nn.Module):
    """文本目标条件化的动作打分头(YOLOE 文本-动作对齐:文本点乘判定动作概率)。

    actor 主干把状态特征投到点乘空间 s∈R^gd;每个动作有可学嵌入 A∈R^{A×gd};
    目标文本嵌入经 task_proj + L2 归一得 g∈R^gd。动作 logit_i = ⟨s, A_i ⊙ g⟩
    —— 目标向量逐元素门控动作嵌入后与状态嵌入点乘,故"用文本乘输出判定动作概率"。

    Args:
        feat_dim:    世界状态特征维。
        num_actions: 离散动作数。
        gd:          点乘空间维度。
        text_dim:    目标文本嵌入维(MiniLM = 384)。
        hidden/layers: 主干 MLP 宽度/层数。
    """

    def __init__(self, feat_dim, num_actions, gd, text_dim, hidden, layers):
        super().__init__()
        self.trunk = MLP(feat_dim, gd, hidden=hidden, layers=layers)
        self.task_proj = nn.Linear(text_dim, gd)
        self.action_emb = nn.Parameter(torch.randn(num_actions, gd) / gd ** 0.5)
        # 可学温度(CLIP/YOLOE 式):cos 相似度 ∈[-1,1] 太平,乘 exp(scale) 给可学锐度。
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))

    def forward(self, feat, goal_emb):
        """feat[..., feat_dim], goal_emb[..., text_dim] → logits[..., num_actions]。

        YOLOE/CLIP 式对齐:状态嵌入 s 与"目标门控的动作嵌入"各自 L2 归一做余弦相似,
        乘可学温度 exp(logit_scale)。目标 g 逐元素门控动作嵌入 ⇒ 文本乘判定动作概率。
        """
        s = F.normalize(self.trunk(feat), dim=-1)               # [..., gd]
        g = F.normalize(self.task_proj(goal_emb), dim=-1)       # [..., gd]
        gated = F.normalize(self.action_emb * g.unsqueeze(-2), dim=-1)   # [..., A, gd]
        cos = (s.unsqueeze(-2) * gated).sum(-1)                 # [..., A] ∈[-1,1]
        return self.logit_scale.exp() * cos


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
        self.use_goal = cfg.use_goal
        if cfg.use_goal:
            gd = cfg.goal_dim or cfg.units
            self.actor = GoalActorHead(feat_dim, cfg.num_actions, gd,
                                       cfg.goal_text_dim, cfg.units, cfg.mlp_layers)
        else:
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
    def actor_dist(self, feat, goal=None):
        """feat(+goal) → 离散动作 OneHot 分布(unimix + 直通梯度)。

        use_goal 时 goal 为文本嵌入 [..., goal_text_dim],走文本点乘头。
        """
        logits = self.actor(feat, goal) if self.use_goal else self.actor(feat)
        return OneHotDist(logits.float(), unimix_ratio=self.cfg.unimix_ratio)

    def value_dist(self, feat):
        """feat → two-hot symexp 价值分布(在线 critic)。"""
        logits = self.value(feat)
        return DiscDist(logits, device=logits.device)

    def slow_value_dist(self, feat):
        """feat → 慢靶 critic 价值分布(EMA 副本,作正则锚)。"""
        logits = self.slow_value(feat)
        return DiscDist(logits, device=logits.device)

    # ── 想象 ────────────────────────────────────────────────────────────────
    def imagine(self, start, dynamics, goal=None):
        """从后验起点状态做 H 步先验 rollout(动作由当前 actor 采样)。

        Args:
            start: 后验状态 dict,字段首维 B、次维 T;内部展平成 B·T 条想象起点。
            dynamics: RSSM(提供 get_feat / img_step)。
            goal: use_goal 时为 [B·T, goal_text_dim] 文本嵌入(每条想象起点一个目标,
                  rollout 全程恒定);否则 None。

        Returns:
            feats:   [H, B·T, feat_dim] 各步状态特征。
            actions: [H, B·T, num_actions] one-hot 动作。
        """
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        state = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            st = prev[0]
            feat = dynamics.get_feat(st)
            action = self.actor_dist(feat.detach(), goal).sample()
            succ = dynamics.img_step(st, action)
            return succ, feat, action

        idx = torch.arange(self.cfg.horizon, device=state["deter"].device)
        _, feats, actions = static_scan(step, (idx,), (state, None, None))
        return feats, actions

    # ── 损失 ──────────────────────────────────────────────────────────────────
    def loss(self, start, world_model, goal=None):
        """想象轨迹上的 actor + critic 损失(世界模型不接收梯度)。

        Args:
            goal: use_goal 时为 [B, T, goal_text_dim] 文本嵌入(每条起点的目标);否则 None。

        Returns:
            actor_loss, value_loss: 标量。
            metrics: dict[str, float]。
        """
        dynamics = world_model.dynamics
        goal_flat = goal.reshape(-1, goal.shape[-1]) if goal is not None else None
        # 想象 rollout 与回报目标全程 no_grad:reinforce 不需要沿 rollout 的路径梯度
        # (actor 仅由显式 logπ·advantage 接收梯度,critic 在 detach 特征上回归),
        # 故切断 rollout 图既省显存又不改学习信号。
        with torch.no_grad():
            feats, actions = self.imagine(start, dynamics, goal_flat)
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
        goal_h = (goal_flat.unsqueeze(0).expand(feats.shape[0], -1, -1)
                  if goal_flat is not None else None)
        policy = self.actor_dist(feats.detach(), goal_h)
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
