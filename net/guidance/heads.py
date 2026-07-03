"""语义奖励头:蒸馏 VLM 片段偏好的目标条件化 shaping 奖励 (net/guidance/heads.py)。

对外接口:
    SemanticRewardHead    — r̂(feat, goal) 的 two-hot symexp 分布头(结构;偏好蒸馏训练在 train/)。
    build_semantic_reward — 由 GuidanceConfig(或字段覆盖)一行构造。

角色定位(见 [knowledge/design_llm_deep_integration.md] §2):本头顶替"稀疏环境奖励"槽位——
VLM 对片段做目标条件化成对偏好,Bradley-Terry 蒸馏进本头;想象训练中它与世界模型 reward 头
相加构成 shaping 项(ImagBehavior.loss 的 reward_fn 参数)。Critic 定义不变,自动学习
"判官分数的期望"。它不替代 Critic(价值/时序信用分配),也不进世界模型动力学。
"""
import torch
import torch.nn as nn

from blocks import MLP, DiscDist
from net.guidance.config import GuidanceConfig


class SemanticRewardHead(nn.Module):
    """目标条件化语义奖励头。

    Args:
        feat_dim: 世界状态特征维(DreamerV3 = stoch_flat + deter;Dreamer4 = S·token_dim)。
        cfg:      GuidanceConfig(读 goal_text_dim/units/mlp_layers/reward_bins)。

    输出经 two-hot symexp 离散分布(I3 构造有界),与 DreamerV3 reward 头同分布族,
    可直接相加进 λ-return。
    """

    def __init__(self, feat_dim: int, cfg: GuidanceConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = MLP(feat_dim + cfg.goal_text_dim, cfg.reward_bins,
                       hidden=cfg.units, layers=cfg.mlp_layers)

    def dist(self, feat, goal):
        """feat [..., feat_dim] + goal [..., goal_text_dim](L2 归一文本嵌入)→ DiscDist。"""
        logits = self.mlp(torch.cat([feat, goal], dim=-1))
        return DiscDist(logits, device=logits.device)

    def reward(self, feat, goal):
        """shaping 奖励点估计 [..., 1](分布众数 × reward_coef)。

        供 ImagBehavior.loss(reward_fn=...) 在 no_grad 下调用;goal 为 None 时返回 0
        (无目标 ⇒ 语义通道自然置零,兼容北极星防火墙评估)。
        """
        if goal is None:
            return torch.zeros(*feat.shape[:-1], 1, device=feat.device, dtype=feat.dtype)
        return self.cfg.reward_coef * self.dist(feat, goal).mode()


def build_semantic_reward(feat_dim: int, **overrides) -> SemanticRewardHead:
    """构造语义奖励头。

    Args:
        feat_dim: 世界状态特征维。
        **overrides: 覆盖 GuidanceConfig 任意字段。
    """
    return SemanticRewardHead(feat_dim, GuidanceConfig(**overrides))
