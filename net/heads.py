"""序列对齐世界模型的解码头。

提供:
    EventVocabHead — 从预测器隐状态预测不可逆事件码(𝒟)分布(目标来自 Δz_inv 的 VQ 索引)。
    AffordanceHead — 从预测器隐状态预测反事实效应幅度 ‖e‖(no-op 判别)。
    SurpriseHead   — K 个轻量未来预测头,头间方差 = 认知 surprise(事件分段用)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EventVocabHead(nn.Module):
    """不可逆事件码分类头(𝒟 通道)。

    监督目标 = 实测 Δz_inv 经 VectorQuantizer 得到的事件索引(编码空间自监督,**非动作标签**)。
    属辅助离散通道:主对齐损失是潜对齐(见 world_model),本头不喧宾夺主。
    """
    def __init__(self, d, event_vocab_size):
        super().__init__()
        self.event_vocab_size = event_vocab_size
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 128), nn.SiLU(),
            nn.Linear(128, event_vocab_size))

    def forward(self, x):
        """x: [B, ..., d] → 事件 logits: [B, ..., event_vocab_size]。"""
        return self.head(x)


class AffordanceHead(nn.Module):
    """反事实效应幅度头:预测 ‖e‖ = ‖ẑ_inv(do a) − ẑ_inv(do null)‖。

    目标 = stop-grad 的实测 ‖e‖。空中跳/无效操作 → ‖e‖≈0 → no-op;捡物/合成 → ‖e‖ 大。
    输出经 softplus 保证非负(I3)。
    """
    def __init__(self, d, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        """x: [B, ..., d] → 非负标量 ‖e‖_hat: [B, ...]。"""
        return F.softplus(self.head(x).squeeze(-1)) + self.eps


class SurpriseHead(nn.Module):
    """K 个轻量未来预测头;头间预测方差 = 认知 surprise(事件分段/异常用)。

    各头独立线性预测 z_inv,集成均值作预测、方差作不确定度。前向只产标量 surprise,
    不改主预测路径(I6 精神:不稳定/统计项进损失或指标,不进主前向)。
    """
    def __init__(self, d, d_inv, n_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, d_inv) for _ in range(n_heads)])

    def forward(self, x):
        """x: [B, ..., d] → (preds [K,B,...,d_inv], surprise [B,...])。"""
        preds = torch.stack([h(x) for h in self.heads], dim=0)   # [K,B,...,d_inv]
        surprise = preds.float().var(dim=0).mean(dim=-1)         # [B,...] fp32(I4)
        return preds, surprise
