"""世界读出仪器(Probe)。

把一个 slot 的潜向量解码成可与环境 GT 直接对照的音符状态。
这**不是世界模型本体**,而是一根证伪用的探针:如果 Z 里真的存了世界,
就应该能从中读出音符的真实位置/轨道/颜色,哪怕此刻像素被遮挡。

刻意保持极薄(单隐层),避免探针自己"脑补"出世界——
读得准只能归功于 Z 本身,而非这个解码器的容量。
"""
import torch
import torch.nn as nn


class WorldProbeDecoder(nn.Module):
    def __init__(self, d, num_tracks=4, num_colors=4, y_min=-24.0, y_max=280.0):
        super().__init__()
        self.y_min, self.y_max = y_min, y_max
        self.trunk = nn.Sequential(nn.Linear(d, 128), nn.SiLU())
        self.y_head = nn.Linear(128, 1)
        self.track_head = nn.Linear(128, num_tracks)
        self.color_head = nn.Linear(128, num_colors)
        self.exist_head = nn.Linear(128, 1)
        self.speed_head = nn.Linear(128, 1)

    def forward(self, slot):
        # slot: [B, d]
        f = self.trunk(slot)
        y_norm = torch.sigmoid(self.y_head(f)).squeeze(-1)            # [B] ∈ (0,1)
        y_px = y_norm * (self.y_max - self.y_min) + self.y_min        # 还原为像素
        return {
            "y_norm": y_norm,
            "y_px": y_px,
            "track_logits": self.track_head(f),
            "color_logits": self.color_head(f),
            "exist": torch.sigmoid(self.exist_head(f)).squeeze(-1),
            "speed": torch.relu(self.speed_head(f)).squeeze(-1),
        }
