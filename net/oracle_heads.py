"""Minecraft 世界模型的 Oracle 辅助评估头 (net/oracle_heads.py)

此文件收拢了用于分析逆动力学和前向预测信息极限的 oracle 评估头。
依据生产代码纯净原则与环境分离原则，net/ 内部不导入任何 domains/。
"""
import torch
import torch.nn as nn

CAMERA_BINS = 11  # 默认相机分箱数，与 domains.minecraft.vpt_action.CAMERA_BINS 一致


class PoolHead(nn.Module):
    """从 patch 平均特征中预测动作的 MLP 头 (仅用于评估逆动力学下界)。"""

    def __init__(self, din=384, camera_bins=CAMERA_BINS, num_keys=20):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(din, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.dx = nn.Linear(256, camera_bins)
        self.dy = nn.Linear(256, camera_bins)
        self.kb = nn.Linear(256, num_keys)

    def forward(self, x):  # [B, din]
        h = self.trunk(x)
        return self.dx(h), self.dy(h), self.kb(h)


class GridHead(nn.Module):
    """保留空间结构的 patch 变换浅层 CNN 头 (用于评估逆动力学空间上限)。"""

    def __init__(self, din=384, gh=9, gw=9, camera_bins=CAMERA_BINS, num_keys=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(din, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GELU(),
        )
        flat = 128 * gh * gw
        self.drop = nn.Dropout(0.3)
        self.dx = nn.Linear(flat, camera_bins)
        self.dy = nn.Linear(flat, camera_bins)
        self.kb = nn.Linear(flat, num_keys)

    def forward(self, x):  # [B, din, gh, gw]
        h = self.drop(self.conv(x).flatten(1))
        return self.dx(h), self.dy(h), self.kb(h)


class PredOracle(nn.Module):
    """隔离的 Δz 预测 Transformer 头 (用于评估前向预测 1-step Bayes 下限)。"""

    def __init__(self, d, A, S, width=384, layers=4, heads=8):
        super().__init__()
        self.slot_in = nn.Linear(d, width)
        self.act_in = nn.Linear(A, width)
        self.act_pos = nn.Parameter(torch.randn(1, S, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, batch_first=True, activation="gelu", dropout=0.0
        )
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(width, d)

    def forward(self, z, a):  # z[B, N, d], a[B, S, A]
        N = z.shape[1]
        x = torch.cat([self.slot_in(z), self.act_in(a) + self.act_pos], 1)
        return self.out(self.blocks(x)[:, :N])
