"""DreamerV3 稀疏候选规划器 (net/dreamerv3/planner.py)。

对外接口:
    Planner — 模型内 MPC(random shooting):从密集 actor 采 N 条长 L 候选动作序列,
              用世界模型想象 rollout,按"折扣回报 + α·目标对齐"打分,选最优候选的首动作执行。

这是"密集头带稀疏头"(YOLOE-26 经验)里的稀疏头:密集 goal-actor 当候选**提议分布**,
规划器在世界模型想象里做一步前瞻择优(稀疏),其选中的动作再蒸回密集头(见训练循环)。
候选作 batch 维并行,过 RSSM 仅 L 步,代价与候选数几乎无关(瓶颈是 plan 长度 L)。

只 import torch/blocks 与本包,net/ 纯净;不含训练循环/优化器。设计见 plan / [[crafter-1m-run-config]]。
"""
import torch
import torch.nn.functional as F


class Planner:
    """模型内候选规划器(MPC random shooting)。

    Args:
        agent:           DreamerV3(持有 world_model + behavior)。
        n_candidates:    候选动作序列数 N。
        horizon:         候选序列长度 L。
        discount:        折扣 γ(打分用)。
        goal_align_coef: 目标对齐项权重 α(use_goal 时生效)。
        use_goal:        是否启用文本目标条件化打分。
    """

    def __init__(self, agent, n_candidates, horizon, discount,
                 goal_align_coef, use_goal):
        self.wm = agent.world_model
        self.behavior = agent.behavior
        self.N = n_candidates
        self.L = horizon
        self.discount = discount
        self.alpha = goal_align_coef
        self.use_goal = use_goal

    @torch.no_grad()
    def act(self, latent, goal=None, training=True):
        """从当前 latent 规划一步。

        Args:
            latent:   后验状态 dict(字段首维 B = n_envs)。
            goal:     [B, goal_text_dim] 文本嵌入或 None。
            training: True 候选动作随机采样(探索),False 取众数。

        Returns:
            action_idx:    [B] long(最优候选的首动作)。
            action_onehot: [B, A] float。
        """
        wm, beh, dyn = self.wm, self.behavior, self.wm.dynamics
        N, L = self.N, self.L
        B = latent["deter"].shape[0]
        rep = lambda x: x.repeat_interleave(N, dim=0)              # 每 env 复制 N 份
        state = {k: rep(v) for k, v in latent.items()}
        goal_rep = rep(goal) if (self.use_goal and goal is not None) else None

        feats, actions = [], []
        for _ in range(L):
            feat = dyn.get_feat(state)                            # [B·N, feat]
            dist = beh.actor_dist(feat, goal_rep)
            act = dist.sample() if training else dist.mode()      # [B·N, A]
            feats.append(feat)
            actions.append(act)
            state = dyn.img_step(state, act)
        feats = torch.stack(feats)                                # [L, B·N, feat]
        actions = torch.stack(actions)                            # [L, B·N, A]

        reward = wm.reward_dist(feats).mode().squeeze(-1)         # [L, B·N]
        goal_h = (goal_rep.unsqueeze(0).expand(L, -1, -1)
                  if goal_rep is not None else None)
        value = beh.value_dist(feats, goal_h).mode().squeeze(-1)  # [L, B·N]
        disc = self.discount ** torch.arange(L, device=feats.device).float()
        ret = (disc[:, None] * reward).sum(0) + self.discount ** L * value[-1]  # [B·N]

        # 目标对齐:复用 goal-actor 的 trunk + task_proj,累计访问态在目标嵌入空间的余弦。
        if self.use_goal and goal_rep is not None and self.alpha > 0:
            head = beh.actor                                      # GoalActorHead
            s = F.normalize(head.trunk(feats), dim=-1)            # [L, B·N, gd]
            g = F.normalize(head.task_proj(goal_rep), dim=-1)     # [B·N, gd]
            align = (s * g.unsqueeze(0)).sum(-1)                  # [L, B·N]
            ret = ret + self.alpha * (disc[:, None] * align).sum(0)

        score = ret.reshape(B, N)                                 # [B, N]
        best = score.argmax(dim=1)                                # [B]
        first = actions[0].reshape(B, N, -1)                      # [B, N, A]
        chosen = first[torch.arange(B, device=first.device), best]   # [B, A]
        return chosen.argmax(-1), chosen
