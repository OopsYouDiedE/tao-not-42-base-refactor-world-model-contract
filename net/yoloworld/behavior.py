"""YOLO-World-Dreamer 双头行为线 (net/yoloworld/behavior.py)。

对外接口:
    Critic            — 目标条件价值 V^g(φ, g) 的 two-hot symexp 头。
    RewardEMA         — 想象回报 5/95 分位 EMA(优势归一,尺度下界 1)。
    DualHeadBehavior  — 持有 critic + 慢靶,在世界模型想象上算损失。

学习信号锚定 DreamerV3(已验证鲁棒):**主信号** L_actor = 对每条 rollout 候选逐步施加
λ-return 策略梯度 + RewardEMA 归一优势 + 熵正则(256 候选群体 = 天然多样本)。**二级蒸馏**(小权重):
L_cls(YOLOv10 一致性)/ L_align(YOLOE 嵌入对齐,供推理期点乘选序列)/ L_div+L_load(反 slot 坍缩)。
选择线早期回报无差时退化无害,待世界模型把候选回报训出差异后自然激活(见 knowledge/yoloworld.md §5)。

rollout 全程矢量化:[n_start] 起点 × [M+ε] 候选展平成 [N·R] 单批沿 H 一次滚先验。世界模型不接收梯度。
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import MLP, DiscDist, lambda_return
from net.yoloworld.config import YoloWorldConfig
from net.yoloworld.heads import select_score


class Critic(nn.Module):
    """目标条件价值头 V^g(φ, g) → two-hot symexp logits。

    Args:
        cfg: YoloWorldConfig(读 feat_dim/task_dim/task_proj_dim/units/mlp_layers/reward_bins)。

    Forward(feat, task_emb): feat [..., d_φ], task_emb [..., d_g] → logits [..., reward_bins]。
    """

    def __init__(self, cfg: YoloWorldConfig):
        super().__init__()
        self.task_proj = nn.Linear(cfg.task_dim, cfg.task_proj_dim)
        self.value = MLP(cfg.feat_dim + cfg.task_proj_dim, cfg.reward_bins,
                         hidden=cfg.units, layers=cfg.mlp_layers)

    def forward(self, feat, task_emb):
        gt = self.task_proj(task_emb)
        return self.value(torch.cat([feat, gt], dim=-1))


class RewardEMA:
    """想象回报 5%/95% 分位 EMA → 优势归一到稳定尺度(scale ≥ 1,I1)。沿用 DreamerV3。"""

    def __init__(self, alpha=1e-2):
        self.alpha = alpha

    def __call__(self, x, ema_vals):
        flat = x.detach().flatten()
        q = torch.quantile(
            flat, torch.tensor([0.05, 0.95], device=x.device, dtype=flat.dtype))
        ema_vals.mul_(1.0 - self.alpha).add_(self.alpha * q)
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        return ema_vals[0].detach(), scale.detach()


class DualHeadBehavior(nn.Module):
    """双头行为线(DreamerV3 主信号 + 选择二级蒸馏)。

    Args:
        cfg: YoloWorldConfig。

    需在使用前 set_ach_embed(E) 注入 22 句成就描述的冻结嵌入矩阵 E [U, d_g]
    (域常量,由 train/ 用 TaskTextEncoder 算好传入;net/ 不读文件)。
    """

    def __init__(self, cfg: YoloWorldConfig):
        super().__init__()
        self.cfg = cfg
        self.critic = Critic(cfg)
        self.slow_critic = copy.deepcopy(self.critic)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.reward_ema = RewardEMA()
        self.register_buffer("ema_vals", torch.zeros(2))
        self.register_buffer("ach_embed",
                             torch.zeros(cfg.n_achievements, cfg.task_dim))

    def set_ach_embed(self, E: torch.Tensor):
        """注入成就描述嵌入矩阵 E [U, d_g](单位球行向量)。"""
        self.ach_embed.copy_(E.to(self.ach_embed.device))

    def value_dist(self, feat, task_emb):
        logits = self.critic(feat, task_emb)
        return DiscDist(logits, device=logits.device)

    def slow_value_dist(self, feat, task_emb):
        logits = self.slow_critic(feat, task_emb)
        return DiscDist(logits, device=logits.device)

    def update_slow(self):
        """慢靶 critic 向在线 critic 做一步 EMA 混合。"""
        mix = self.cfg.value_decay
        for s, d in zip(self.critic.parameters(), self.slow_critic.parameters()):
            d.data.mul_(1.0 - mix).add_(mix * s.data)

    # ── 损失 ──────────────────────────────────────────────────────────────────
    def loss(self, start, task_emb, proposal_head, world_model):
        """双头行为损失(主 λ-return PG + 二级蒸馏)。

        Args:
            start:    后验起点状态 dict,字段首维 N(子采样后的起点数)。
            task_emb: [N, d_g] 每个起点的任务句向量。
            proposal_head: ProposalHead(小头)。
            world_model:   WorldModel(提供 dynamics / ach_prob / cont_dist)。

        Returns:
            loss:    标量(actor + critic + 二级蒸馏)。
            metrics: dict[str, float]。
        """
        cfg = self.cfg
        dyn = world_model.dynamics
        H, A, K = cfg.plan_horizon, cfg.num_actions, cfg.n_candidates
        N = start["deter"].shape[0]
        device = start["deter"].device
        gamma = cfg.discount

        feat0 = dyn.get_feat(start)                                   # [N, d_φ]
        w = torch.softmax(
            task_emb @ self.ach_embed.t() / cfg.shaping_tau, dim=-1)  # [N, U]

        # ── 小头(学生)在 detach 状态上前向 ──────────────────────────────
        plan_logits, p, e = proposal_head(feat0.detach(), task_emb)   # [N,K,H,A]/[N,K]/[N,K,d_g]
        alpha_logits = select_score(p, e, task_emb, cfg.select_beta)  # [N, K]

        # ── 候选预筛:top-M(按 α)+ ε 随机覆盖 ─────────────────────────
        M, Kx = min(cfg.n_rollout, K), cfg.n_explore
        with torch.no_grad():
            topm = alpha_logits.topk(M, dim=-1).indices              # [N, M]
            if Kx > 0:
                rand = torch.randint(0, K, (N, Kx), device=device)
                cand_idx = torch.cat([topm, rand], dim=1)            # [N, R]
            else:
                cand_idx = topm
        R = cand_idx.shape[1]
        sel_logits = torch.gather(
            plan_logits, 1,
            cand_idx[..., None, None].expand(-1, -1, H, A))          # [N, R, H, A]

        rep = lambda x: x.unsqueeze(1).expand(
            N, R, *x.shape[1:]).reshape(N * R, *x.shape[1:])

        # ── rollout(no_grad,矢量化单批滚先验)→ feats / 势函数塑形奖励 ──
        with torch.no_grad():
            a_idx = torch.distributions.Categorical(logits=sel_logits).sample()
            actions = F.one_hot(a_idx, A).float()                    # [N, R, H, A]
            act_flat = actions.reshape(N * R, H, A)

            st = {k: rep(v).contiguous() for k, v in start.items()}
            feats = []
            for tau in range(H):
                st = dyn.img_step(st, act_flat[:, tau])
                feats.append(dyn.get_feat(st))
            feats = torch.stack(feats, dim=1)                        # [N·R, H, d_φ]
            feats_tf = feats.transpose(0, 1)                         # [H, N·R, d_φ]
            te_tf = rep(task_emb).unsqueeze(0).expand(H, -1, -1)     # [H, N·R, d_g]

            psi = world_model.ach_prob(feats)                        # [N·R, H, U]
            w_rep = rep(w)
            rho = torch.einsum("rhu,ru->rh", psi, w_rep)            # [N·R, H]
            rho0 = (world_model.ach_prob(feat0) * w).sum(-1)
            rho_prev = torch.cat([rep(rho0)[:, None], rho[:, :-1]], dim=1)
            r_tf = (gamma * rho - rho_prev).transpose(0, 1).unsqueeze(-1)   # [H,N·R,1]

            value = self.value_dist(feats_tf, te_tf).mode()          # [H, N·R, 1]
            cont = world_model.cont_dist(feats_tf).mean              # [H, N·R, 1]∈(0,1)
            discount = gamma * cont
            # λ-return 目标 + 折扣权重 + RewardEMA 归一优势(DreamerV3 主信号)
            target = torch.stack(lambda_return(
                r_tf[:-1], value[:-1], discount[:-1], bootstrap=value[-1],
                lambda_=cfg.disc_lambda, axis=0), dim=1)             # [H-1, N·R, 1]
            weights = torch.cumprod(
                torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0)
            offset, scale = self.reward_ema(target, self.ema_vals)
            adv = (target - offset) / scale - (value[:-1] - offset) / scale

        # ── 主信号 L_actor:λ-return 策略梯度 + 熵(对每条候选逐步)──────
        logp_full = F.log_softmax(sel_logits, dim=-1)               # [N,R,H,A]
        logp_a = (logp_full * actions).sum(-1)                      # [N,R,H]
        prob = logp_full.exp()
        ent_step = -(prob * logp_full).sum(-1)                      # [N,R,H]
        logp_tf = logp_a.reshape(N * R, H).transpose(0, 1).unsqueeze(-1)   # [H,N·R,1]
        ent_tf = ent_step.reshape(N * R, H).transpose(0, 1).unsqueeze(-1)
        actor_term = logp_tf[:-1] * adv.detach() + cfg.actor_entropy * ent_tf[:-1]
        loss_actor = -(weights[:-1] * actor_term).mean()

        # ── L_critic:two-hot 回归 λ-return + 慢靶正则 ───────────────────
        vdist = self.value_dist(feats_tf[:-1].detach(), te_tf[:-1])
        with torch.no_grad():
            slow = self.slow_value_dist(feats_tf[:-1].detach(), te_tf[:-1]).mode()
        loss_value = (-vdist.log_prob(target.detach())
                      - vdist.log_prob(slow.detach()))
        loss_value = (weights[:-1].squeeze(-1) * loss_value).mean()

        # ── 二级:选择蒸馏 + 嵌入对齐(小权重)───────────────────────────
        with torch.no_grad():
            ret = (weights[:-1] * r_tf[:-1]).sum(0).reshape(N, R) \
                + (weights[-1] * value[-1]).reshape(N, R)           # 候选回报(排序用)
            ret_z = (ret - ret.mean(1, keepdim=True)) / (ret.std(1, keepdim=True) + 1e-4)
            t_soft = torch.softmax(ret_z / cfg.teacher_temp, dim=-1)  # [N, R]
            disc_h = gamma ** torch.arange(H, device=device, dtype=feats.dtype)
            e_g = torch.einsum("rhu,h->ru", psi, disc_h) @ self.ach_embed
            ehat = F.normalize(e_g, dim=-1).reshape(N, R, -1)        # [N, R, d_g]
        sel_alpha = torch.gather(alpha_logits, 1, cand_idx)         # [N, R]
        loss_cls = -(t_soft * F.log_softmax(sel_alpha, dim=-1)).sum(-1).mean()
        sel_e = torch.gather(e, 1, cand_idx[..., None].expand(-1, -1, e.shape[-1]))
        loss_align = (1.0 - (sel_e * ehat).sum(-1)).mean()

        # ── 二级:反 slot 坍缩(全 K,batch 归约)──────────────────────────
        eye = torch.eye(K, device=device)
        ebar = F.normalize(e.mean(0), dim=-1)                       # [K, d_g]
        loss_div = (ebar @ ebar.t() - eye).pow(2).triu(1).sum() / max(K * (K - 1) // 2, 1)
        abar = F.softmax(alpha_logits, dim=-1).mean(0)             # [K]
        loss_load = (abar * (abar + 1e-8).log()).sum()

        loss = (cfg.plan_scale * loss_actor + loss_value
                + cfg.cls_scale * loss_cls + cfg.align_scale * loss_align
                + cfg.div_scale * loss_div + cfg.load_scale * loss_load)
        metrics = {
            "actor": loss_actor.item(),
            "value": loss_value.item(),
            "cls": loss_cls.item(),
            "align": loss_align.item(),
            "div": loss_div.item(),
            "load": loss_load.item(),
            "entropy": ent_step.mean().item(),
            "ret_scale": scale.item(),
            "ret_best": ret.max(dim=1).values.mean().item(),
        }
        return loss, metrics
