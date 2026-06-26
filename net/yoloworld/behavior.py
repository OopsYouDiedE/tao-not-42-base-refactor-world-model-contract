"""YOLO-World-Dreamer 双头行为线 (net/yoloworld/behavior.py)。

对外接口:
    Critic            — 目标条件价值 V^g(φ, g) 的 two-hot symexp 头。
    DualHeadBehavior  — 持有 critic + 慢靶,在世界模型想象上计算双头损失:
                        L_cls(YOLOv10 一致性)/ L_plan(群体基线 REINFORCE)/
                        L_align(YOLOE 对齐)/ L_critic。

rollout 老师全程矢量化:把 [n_start] 个起点 × [M+ε] 个候选展平成单批 [N·R],沿 H 一次性滚先验,
GPU/CPU 都吃满批维。世界模型不接收梯度(行为线对 feat detach)。设计见 knowledge/yoloworld.md §5。
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import MLP, DiscDist
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


class DualHeadBehavior(nn.Module):
    """双头行为线(rollout 老师监督 + 目标条件 critic)。

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
        """双头行为损失。

        Args:
            start:    后验起点状态 dict,字段首维 N(= 子采样后的起点数)。
            task_emb: [N, d_g] 每个起点的任务句向量。
            proposal_head: ProposalHead(小头)。
            world_model:   WorldModel(提供 dynamics / ach_prob)。

        Returns:
            loss:    标量(cls+plan+align+critic 加权和)。
            metrics: dict[str, float]。
        """
        cfg = self.cfg
        dyn = world_model.dynamics
        H, A, K = cfg.plan_horizon, cfg.num_actions, cfg.n_candidates
        N = start["deter"].shape[0]
        device = start["deter"].device
        gamma = cfg.discount

        feat0 = dyn.get_feat(start)                                   # [N, d_φ]
        # 任务对成就的注意力权重 w(g)=softmax(E·g/τ)
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

        # ── rollout 老师(全程 no_grad,矢量化单批滚先验) ──────────────
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

            psi = world_model.ach_prob(feats)                        # [N·R, H, U]
            w_rep = rep(w)                                           # [N·R, U]
            rho = torch.einsum("rhu,ru->rh", psi, w_rep)            # [N·R, H]
            rho0 = (world_model.ach_prob(feat0) * w).sum(-1)        # [N]
            rho_prev = torch.cat([rep(rho0)[:, None], rho[:, :-1]], dim=1)
            r = gamma * rho - rho_prev                               # 势函数塑形 [N·R, H]

            te_rep = rep(task_emb)                                   # [N·R, d_g]
            val = self.value_dist(feats, te_rep[:, None, :].expand(-1, H, -1)
                                  ).mode().squeeze(-1)               # [N·R, H]
            # 折扣回报(bootstrap V(s_H)) → 候选排序 R^k
            disc = gamma ** torch.arange(H, device=device, dtype=r.dtype)
            ret = (disc * r).sum(-1) + gamma ** H * val[:, -1]       # [N·R]
            ret = ret.reshape(N, R)
            t_soft = torch.softmax(ret / cfg.teacher_temp, dim=-1)  # [N, R] 老师信念
            # 老师计划嵌入 ê = normalize(Σ γ^{τ-1} Eᵀ ψ_τ)
            e_g = torch.einsum("rhu,h->ru", psi, disc) @ self.ach_embed
            ehat = F.normalize(e_g, dim=-1).reshape(N, R, -1)        # [N, R, d_g]
            # critic 目标:n-step return-to-go(逆向累积)
            target = torch.empty_like(val)
            acc = val[:, -1]
            for h in reversed(range(H)):
                acc = r[:, h] + gamma * acc
                target[:, h] = acc
            slow = self.slow_value_dist(
                feats, te_rep[:, None, :].expand(-1, H, -1)).mode().squeeze(-1)
            # 群体基线优势(标准化,I1 下界)
            adv = ret - ret.mean(dim=1, keepdim=True)
            adv = adv / (ret.std(dim=1, keepdim=True) + 1e-4)       # [N, R]

        # ── L_cls:KL(t ‖ α)(YOLOv10 一致性) ──────────────────────────
        sel_alpha = torch.gather(alpha_logits, 1, cand_idx)         # [N, R]
        log_alpha = F.log_softmax(sel_alpha, dim=-1)
        loss_cls = -(t_soft * log_alpha).sum(-1).mean()

        # ── L_plan:群体基线 REINFORCE + 熵 ────────────────────────────
        logp = F.log_softmax(sel_logits, dim=-1)                    # [N, R, H, A]
        logp_a = (logp * actions).sum(-1).sum(-1)                   # [N, R] 序列 log π
        prob = logp.exp()
        ent = -(prob * logp).sum(-1).mean()
        loss_plan = -(adv * logp_a).mean() - cfg.actor_entropy * ent

        # ── L_align:e^k → ê^k(YOLOE 对齐) ────────────────────────────
        sel_e = torch.gather(
            e, 1, cand_idx[..., None].expand(-1, -1, e.shape[-1]))  # [N, R, d_g]
        loss_align = (1.0 - (sel_e * ehat).sum(-1)).mean()

        # ── L_critic:two-hot 回归 return + 慢靶正则 ───────────────────
        vdist = self.value_dist(feats, te_rep[:, None, :].expand(-1, H, -1))
        loss_value = (-vdist.log_prob(target.unsqueeze(-1))
                      - vdist.log_prob(slow.unsqueeze(-1))).mean()

        # ── 反候选坍缩(全 K slot,batch 归约,廉价) ───────────────────
        # L_div:batch 平均 slot 嵌入互斥(d_g=384 维,K 可近正交)→ 语义多样。
        eye = torch.eye(K, device=device)
        ebar = F.normalize(e.mean(0), dim=-1)                       # [K, d_g]
        gram = ebar @ ebar.t()                                      # [K, K]
        loss_div = (gram - eye).pow(2).triu(1).sum() / max(K * (K - 1) // 2, 1)
        # L_load:batch 平均选择分布的负熵(↓ = 使用更均衡)→ 均衡 slot 使用。
        abar = F.softmax(alpha_logits, dim=-1).mean(0)              # [K]
        loss_load = (abar * (abar + 1e-8).log()).sum()

        loss = (cfg.cls_scale * loss_cls + cfg.plan_scale * loss_plan
                + cfg.align_scale * loss_align + loss_value
                + cfg.div_scale * loss_div + cfg.load_scale * loss_load)
        metrics = {
            "cls": loss_cls.item(),
            "plan": loss_plan.item(),
            "align": loss_align.item(),
            "value": loss_value.item(),
            "entropy": ent.item(),
            "div": loss_div.item(),
            "load": loss_load.item(),
            "slot_use": float((abar > 1.0 / (4 * K)).sum().item()),
            "ret_mean": ret.mean().item(),
            "ret_best": ret.max(dim=1).values.mean().item(),
        }
        return loss, metrics
