"""L1 primitive 积木库 - 空间几何与投影 (blocks/spatial.py)"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_FP16 = 1e-4  # I1: fp16 安全 epsilon(绝不用 1e-12)


def _base_grid(h, w, device, dtype=torch.float32):
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0)  # [2,H,W] 顺序 (x,y)


class Warp(nn.Module):
    """局部光流重采样。flow=(dx,dy) 像素位移。坐标 fp32(I4);双线性凸插值 ⇒ 非扩张(I5)。"""

    def forward(self, feat, flow):
        B, C, H, W = feat.shape
        base = _base_grid(H, W, feat.device).unsqueeze(0)            # [1,2,H,W] fp32
        coords = base + flow.float()                                  # fp32, I4
        gx = 2.0 * coords[:, 0] / max(W - 1, 1) - 1.0
        gy = 2.0 * coords[:, 1] / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)                         # [B,H,W,2]
        out = F.grid_sample(feat.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.to(feat.dtype)


class GlobalTransformApply(nn.Module):
    """全局仿射屏幕空间变换。theta:[B,2,3]。fp32 网格(I4),非扩张(I5)。"""

    def forward(self, feat, theta):
        B, C, H, W = feat.shape
        grid = F.affine_grid(theta.float(), (B, C, H, W), align_corners=True)
        out = F.grid_sample(feat.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.to(feat.dtype)


class BEVSplat(nn.Module):
    """图像特征经深度抬升 + 位姿 scatter 到俯视 BEV 格(3D→BEV)。坐标 fp32(I4)。

    像素 →(K_inv)射线 →(×depth)相机系点 →(pose)世界点 →(x,z)量化到 BEV 格 → scatter_add。
    scatter_add 守恒:在范围内像素的特征质量被保留。2D 地图退化为 `GlobalTransformApply`,不需它。
    """

    def __init__(self, bev_hw=(64, 64), x_range=(-10.0, 10.0), z_range=(0.0, 20.0)):
        super().__init__()
        self.Hb, self.Wb = bev_hw
        self.x_range, self.z_range = x_range, z_range

    def forward(self, feat, depth, K_inv, pose):
        # feat[B,C,H,W] depth[B,1,H,W] K_inv[B,3,3] pose[B,4,4](cam→world)
        B, C, H, W = feat.shape
        v, u = torch.meshgrid(torch.arange(H, device=feat.device, dtype=torch.float32),
                              torch.arange(W, device=feat.device, dtype=torch.float32),
                              indexing="ij")
        pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)         # [H,W,3]
        rays = torch.einsum("bij,hwj->bhwi", K_inv.float(), pix)      # [B,H,W,3] fp32
        pts_cam = rays * depth.permute(0, 2, 3, 1).float()           # [B,H,W,3]
        pts_world = (torch.einsum("bij,bhwj->bhwi", pose[:, :3, :3].float(), pts_cam)
                     + pose[:, :3, 3].float()[:, None, None, :])      # [B,H,W,3]
        x, z = pts_world[..., 0], pts_world[..., 2]
        gx = ((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.Wb).long()
        gz = ((z - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * self.Hb).long()
        valid = (gx >= 0) & (gx < self.Wb) & (gz >= 0) & (gz < self.Hb)
        idx = (gz.clamp(0, self.Hb - 1) * self.Wb + gx.clamp(0, self.Wb - 1)).view(B, -1)
        feat_flat = (feat * valid.unsqueeze(1)).flatten(2)           # [B,C,HW],无效像素清零
        bev = feat.new_zeros(B, C, self.Hb * self.Wb)
        bev.scatter_add_(2, idx.unsqueeze(1).expand(B, C, H * W), feat_flat)
        return bev.view(B, C, self.Hb, self.Wb)


def rot6d_to_matrix(x, eps=EPS_FP16):
    """6D → SO(3) Gram-Schmidt(eps fp16 安全, I1)。"""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def make_4x4(R, t):
    B = R.shape[0]
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t.reshape(B, 3)
    return T
