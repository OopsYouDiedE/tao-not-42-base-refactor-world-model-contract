# -*- coding: utf-8 -*-
"""ConvSegHead(G1 阶梯 2 分割头)+ 针孔投影 GT 工厂(net/fovea_twotower)。

原散在 train/fovea_twotower/eval_g1.py,被 TokenHead/训练/评测/质检共用,归位到 net/。
投影 GT 工厂 = 确定性监督的核心件:已知方块世界坐标 + 逐帧位姿 → 前脸掩膜,
零人工标注(raycast 定量核对 178/178,见 eval_g1 --mode gtvis)。

坐标约定:MC 竖直 FOV 70°,眼高 1.62,yaw=0 朝 +z,pitch 正=低头;
输出掩膜在 640×384 pad 坐标系(yolo_unified.pad384,上垫 12px)。
"""
import cv2
import numpy as np
import torch

from net.fovea_twotower.token_stream import CLASSES, EYE_H
from net.fovea_twotower.yolo_unified import PAD_TOP

FOV_V = 70.0                # MC 默认竖直 FOV(deg);正确性由 gtvis raycast 核对背书
W, H = 640, 360             # 原始渲染


def cam_basis(yaw_deg, pitch_deg):
    """MC 约定:yaw=0 朝 +z,右手系;pitch 正=低头。→ (forward, right, up)。"""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([-np.sin(y) * np.cos(p), -np.sin(p), np.cos(y) * np.cos(p)])
    r = np.array([-f[2], 0.0, f[0]])
    r /= np.linalg.norm(r) + 1e-9
    u = np.cross(r, f)
    return f, r, u


def project_block(bx, by, bz, pose):
    """方块前脸(z=bz 平面朝向房间的四角)→ 384×640 pad 坐标点集或 None。

    课程几何:方块嵌在 z=wall_z 的墙里,玩家恒在 z<bz 侧——只有前脸可见。
    8 角点凸包会把不可见侧脸算进 GT(系统性高估),故只投前脸。"""
    x, y, z, yaw, pitch = pose
    eye = np.array([x, y + EYE_H, z])
    f, r, u = cam_basis(yaw, pitch)
    fy = (H / 2) / np.tan(np.radians(FOV_V) / 2)
    pts = []
    for dx in (0, 1):
        for dy in (0, 1):
            v = np.array([bx + dx, by + dy, bz], float) - eye
            zc = v @ f
            if zc < 0.15:
                return None                          # 角点在背后:整块弃(保守)
            pts.append([W / 2 + fy * (v @ r) / zc,
                        H / 2 - fy * (v @ u) / zc + PAD_TOP])
    return np.array(pts, np.float32)


def gt_masks(gt_blocks: dict, pose):
    """{cls:[[x,y,z],..]} + 位姿 → {cls: bool mask [384,640]}(不可见类=全 False)。"""
    out = {}
    for cls, blocks in gt_blocks.items():
        m = np.zeros((384, 640), np.uint8)
        for b in blocks:
            pts = project_block(*b, pose)
            if pts is None:
                continue
            hull = cv2.convexHull(pts.astype(np.int32))
            cv2.fillConvexPoly(m, hull, 1)
        out[cls] = m.astype(bool)
    return out


def gt_label_img(gt, pose):
    """GT 掩膜 → 逐像素标签 [384,640](CLASSES 序,背景=C)。"""
    ms = gt_masks(gt, pose)
    lab = np.full((384, 640), len(CLASSES), np.int64)
    for k, c in enumerate(CLASSES):
        lab[ms[c]] = k
    return lab


class ConvSegHead(torch.nn.Module):
    """P3 post-BN 嵌入图 [1,512,h,w] → 逐像素 C+1 logits(×8 双线性)。~0.7M 参数。

    动机:向量级(每格独立线性)在随机布局终审 0.445<0.5——残差是 8px 格边界锯齿,
    每格独立分类无法表达形状先验;3×3 conv 栈给邻域上下文,仍不动骨干(阶梯§2-2)。"""

    def __init__(self, cin=512, ch=128, ncls=len(CLASSES) + 1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(cin, ch, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(ch, ch, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(ch, ncls, 1))

    def forward(self, emb):                            # [B,512,h,w]
        lg = self.net(emb)
        return torch.nn.functional.interpolate(
            lg, size=(384, 640), mode="bilinear", align_corners=False)
