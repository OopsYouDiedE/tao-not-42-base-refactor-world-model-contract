import torch
import torch.nn as nn
from blocks.primitives import PreLNAttn

class LaneProbe(nn.Module):
    """结构化 per-lane 读出:N 个 slot [B,N,d] → 每轨道 (y_norm, present, hittable)。

    用 n_lanes 个可学 lane-query 对 slot 做 cross-attention(每条 lane 各自抽取相关 slot),
    取代会抹平 per-lane 结构、稀释单 lane 动作效果的 mean 池化。
    """
    def __init__(self, d, n_lanes=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_lanes, d) * 0.02)
        self.attn = PreLNAttn(d, heads=4, mode="cross")
        self.trunk = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.SiLU())
        self.y = nn.Linear(128, 1)
        self.present = nn.Linear(128, 1)
        self.hittable = nn.Linear(128, 1)

    def forward(self, slots):                          # slots: [B, N, d]
        q = self.queries.expand(slots.shape[0], -1, -1)   # [B, n_lanes, d]
        f = self.trunk(self.attn(q, slots))               # [B, n_lanes, 128]
        return {"y_norm": torch.sigmoid(self.y(f).squeeze(-1)),       # [B, n_lanes]
                "present": torch.sigmoid(self.present(f).squeeze(-1)),
                "hittable": torch.sigmoid(self.hittable(f).squeeze(-1))}
