"""L1 primitive 积木库 - 相似度对比与运动流场估计 (blocks/similarity.py)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_FP16 = 1e-4  # I1: fp16 安全 epsilon(绝不用 1e-12)


class LocalCorr(nn.Module):
    """有界半径余弦相关。输入先 ℓ2norm(eps=1e-4, I1)。输出 ∈[-1,1]。禁止全局注意力。"""

    def __init__(self, radius=4):
        super().__init__()
        self.r = radius

    def forward(self, a, b):
        r = self.r
        a = F.normalize(a, dim=1, eps=EPS_FP16)
        b = F.normalize(b, dim=1, eps=EPS_FP16)
        B, C, H, W = a.shape
        b_pad = F.pad(b, (r, r, r, r))
        outs = []
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                bs = b_pad[:, :, dy:dy + H, dx:dx + W]
                outs.append((a * bs).sum(dim=1, keepdim=True))
        return torch.cat(outs, dim=1)                                # [B,(2r+1)^2,H,W]


class SoftArgmaxFlow(nn.Module):
    """corr → 期望位移 ∈[-r,r]。fp32 softmax(I4),输出有界(I3)。"""

    def __init__(self, radius=4, tau=1.0):
        super().__init__()
        self.r, self.tau = radius, tau
        offs = [[dx, dy] for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)]
        self.register_buffer("offsets", torch.tensor(offs, dtype=torch.float32))

    def forward(self, corr):
        w = F.softmax(corr.float() / self.tau, dim=1)                # fp32, I4
        off = self.offsets.to(corr.device)                          # [K,2]
        fx = (w * off[None, :, 0, None, None]).sum(1, keepdim=True)
        fy = (w * off[None, :, 1, None, None]).sum(1, keepdim=True)
        return torch.cat([fx, fy], dim=1).to(corr.dtype)            # [B,2,H,W]


def box_iou(a, b, kind="iou", eps=1e-7):
    """grid_iou. xyxy 框 IoU/GIoU,fp32(I4)。a,b: [...,4]。"""
    a, b = a.float(), b.float()
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter + eps
    iou = inter / union
    if kind == "iou":
        return iou
    lt_c = torch.min(a[..., :2], b[..., :2])
    rb_c = torch.max(a[..., 2:], b[..., 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1] + eps
    return iou - (area_c - union) / area_c
