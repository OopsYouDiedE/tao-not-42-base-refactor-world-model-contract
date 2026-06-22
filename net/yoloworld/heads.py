"""YOLO-World-Dreamer 候选小头(稀疏头)(net/yoloworld/heads.py)。

对外接口:
    ProposalHead — DETR 式 K 个 query + 共享解码器:由 (φ, 任务编码 g) 一次前向输出
                   K 条候选动作序列计划 logits、概率 p^k、计划嵌入 e^k。
    select_score — 选择/混合 logit:p + β·(e·g),供 α=softmax 与 argmax 选序列(YOLOE)。

这是"采集动作时只跑的便宜小头":全 K 候选在 (N, K) 维上一次性矢量算完,无递归、无搜索
(YOLO26 端到端)。参数 ~10^5;重计算在 rollout 老师里。设计见 knowledge/yoloworld.md §4/§6。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import MLP
from net.yoloworld.config import YoloWorldConfig


class ProposalHead(nn.Module):
    """K=256 候选动作序列小头。

    Args:
        cfg: YoloWorldConfig(读 n_candidates/plan_horizon/num_actions/query_dim/
             head_hidden/task_dim/task_proj_dim/feat_dim)。

    Forward(feat, task_emb):
        feat:     [N, d_φ] 世界状态特征。
        task_emb: [N, d_g] 任务句向量(单位球)。
      Returns:
        plan_logits: [N, K, H, A] 每条候选的逐步动作 logits。
        p:           [N, K] 候选概率 logit。
        e:           [N, K, d_g] 候选计划嵌入(L2 归一,与 task_emb 同空间)。
    """

    def __init__(self, cfg: YoloWorldConfig):
        super().__init__()
        self.cfg = cfg
        K, H, A = cfg.n_candidates, cfg.plan_horizon, cfg.num_actions
        self.K, self.H, self.A = K, H, A

        # query 单位尺度初始化(DETR 式可学序列查询):若初始化过小会被共享上下文淹没,
        # 导致所有 slot 输出相同(候选坍缩)。配合 L_div/L_load 维持 slot 语义多样。
        self.query = nn.Parameter(torch.randn(K, cfg.query_dim))
        self.task_proj = nn.Linear(cfg.task_dim, cfg.task_proj_dim)
        # 上下文:状态 + 投影后任务 → head_hidden(全 K query 共享)
        self.ctx = MLP(cfg.feat_dim + cfg.task_proj_dim, cfg.head_hidden,
                       hidden=cfg.head_hidden, layers=cfg.mlp_layers)
        # 共享解码器:[query, ctx] → head_hidden → 三路输出头
        self.decoder = MLP(cfg.query_dim + cfg.head_hidden, cfg.head_hidden,
                           hidden=cfg.head_hidden, layers=cfg.mlp_layers)
        self.to_plan = nn.Linear(cfg.head_hidden, H * A)
        self.to_p = nn.Linear(cfg.head_hidden, 1)
        self.to_e = nn.Linear(cfg.head_hidden, cfg.task_dim)

    def forward(self, feat, task_emb):
        N = feat.shape[0]
        K, H, A = self.K, self.H, self.A
        gt = self.task_proj(task_emb)                       # [N, d_g']
        c = self.ctx(torch.cat([feat, gt], dim=-1))         # [N, head_hidden]
        q = self.query.unsqueeze(0).expand(N, -1, -1)       # [N, K, d_q]
        c_k = c.unsqueeze(1).expand(-1, K, -1)              # [N, K, head_hidden]
        dec = self.decoder(torch.cat([q, c_k], dim=-1))     # [N, K, head_hidden]

        plan_logits = self.to_plan(dec).reshape(N, K, H, A)
        p = self.to_p(dec).squeeze(-1)                       # [N, K]
        e = F.normalize(self.to_e(dec), dim=-1)              # [N, K, d_g]
        return plan_logits, p, e


def select_score(p, e, task_emb, beta):
    """选择/混合 logit:p + β·(e·g)(YOLOE 点乘),[N, K]。

    Args:
        p:        [N, K] 候选概率 logit。
        e:        [N, K, d_g] 候选嵌入(单位球)。
        task_emb: [N, d_g] 任务句向量(单位球)。
        beta:     点乘项系数 β。

    Returns:
        [N, K] 选择 logit(经 softmax 即混合权重 α;argmax 即所选序列)。
    """
    dot = torch.einsum("nkd,nd->nk", e, task_emb)
    return p + beta * dot
