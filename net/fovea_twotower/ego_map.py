# -*- coding: utf-8 -*-
"""自我中心特征地图(M3 地图模块预研,net/fovea_twotower)。

三种实现共享读写契约:write(pts_ego, feats) / read(pts_ego) / step(dpos_world, dyaw)。
- EgoMapNaive: 均匀格,每步对特征图整体做旋转+平移重采样(文献默认,Neural Map 式);
- EgoMapNorth: 北锚定——地图恒对齐世界朝向,自身运动只producing平移;整格部分用
  torch.roll 无损滚动,亚格残差存偏移寄存器(读写时补),**旋转推迟到读出**。
  等变性由坐标账本保证而非重采样,写入路径零插值损耗;
- EgoMapClip: 北锚定 + clipmap 分级精度:L 级嵌套环,级 l 覆盖半径 R·2^l/2^(L-1),
  近环精、远环粗——空间上的中心凹。读取取包含该点的最细级。

约定:ego 坐标 = 世界位移(北锚定臂)或体坐标(naive 臂),特征 c 维 + 权重 1 维,
读出 = 加权平均 feat/(cnt+eps)。所有读写可微(双线性 splat/采样)。
"""
import numpy as np
import torch
import torch.nn.functional as F


def _splat(grid_f, grid_c, xy_cell, feats):
    """双线性 scatter:xy_cell [N,2](浮点格坐标) feats [N,c] → 累加进 (c,H,W)/(1,H,W)。"""
    H, W = grid_f.shape[-2:]
    x0 = torch.floor(xy_cell[:, 0]); y0 = torch.floor(xy_cell[:, 1])
    fx = xy_cell[:, 0] - x0; fy = xy_cell[:, 1] - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = (x0 + dx).long(); yi = (y0 + dy).long()
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if not ok.any():
            continue
        idx = (yi[ok] * W + xi[ok])
        grid_f.view(-1, H * W).index_add_(1, idx, (feats[ok] * w[ok, None]).T)
        grid_c.view(-1, H * W).index_add_(1, idx, w[ok][None])


def _sample(grid_f, grid_c, xy_cell):
    """双线性读出加权平均特征。xy_cell [N,2] 浮点格坐标 → [N,c]。"""
    H, W = grid_f.shape[-2:]
    g = xy_cell.clone()
    g[:, 0] = g[:, 0] / (W - 1) * 2 - 1
    g[:, 1] = g[:, 1] / (H - 1) * 2 - 1
    g = g.view(1, 1, -1, 2)
    f = F.grid_sample(grid_f[None], g, align_corners=True)[0, :, 0].T
    c = F.grid_sample(grid_c[None], g, align_corners=True)[0, :, 0].T
    return f / (c + 1e-6)


class EgoMapNaive:
    """均匀格 + 每步旋转平移重采样(对照臂:量化累积插值糊)。体坐标系。"""

    def __init__(self, c=8, size=64, half=32.0):
        self.c, self.size, self.half = c, size, half
        self.res = size / (2 * half)                    # cell / 单位
        self.f = torch.zeros(c, size, size)
        self.cnt = torch.zeros(1, size, size)

    def _to_cell(self, xy):
        return (xy + self.half) * self.res

    def step(self, dpos_body, dyaw_rad):
        """体坐标系:自身前移 dpos、转 dyaw → 地图内容反向旋转平移(重采样)。"""
        th = torch.tensor(-dyaw_rad, dtype=torch.float32)
        cs, sn = torch.cos(th), torch.sin(th)
        t = -torch.as_tensor(dpos_body, dtype=torch.float32) * self.res \
            / (self.size / 2)
        A = torch.tensor([[cs, -sn, t[0]], [sn, cs, t[1]]])[None]
        g = F.affine_grid(A, (1, self.c + 1, self.size, self.size),
                          align_corners=False)
        both = torch.cat([self.f, self.cnt])[None]
        both = F.grid_sample(both, g, align_corners=False)[0]
        self.f, self.cnt = both[:self.c], both[self.c:]

    def write(self, pts, feats):
        _splat(self.f, self.cnt, self._to_cell(pts), feats)

    def read(self, pts):
        return _sample(self.f, self.cnt, self._to_cell(pts))


class EgoMapNorth:
    """北锚定:整格 roll + 亚格偏移寄存器,旋转推迟到读出。世界位移坐标系。"""

    def __init__(self, c=8, size=64, half=32.0):
        self.c, self.size, self.half = c, size, half
        self.res = size / (2 * half)
        self.f = torch.zeros(c, size, size)
        self.cnt = torch.zeros(1, size, size)
        self.off = np.zeros(2)                          # 亚格偏移(格单位)

    def _to_cell(self, xy):
        cell = (xy + self.half) * self.res
        return cell + torch.as_tensor(self.off, dtype=torch.float32)

    def step(self, dpos_world, dyaw_rad=0.0):
        """世界系位移;dyaw 只记账不动图(北锚定的全部意义)。"""
        self.off -= np.asarray(dpos_world, float) * self.res
        shift = np.round(self.off).astype(int)
        if shift.any():
            self.f = torch.roll(self.f, (int(shift[1]), int(shift[0])), (1, 2))
            self.cnt = torch.roll(self.cnt, (int(shift[1]), int(shift[0])),
                                  (1, 2))
            # 滚入边清零(新领土未知)
            for d, s in ((2, int(shift[0])), (1, int(shift[1]))):
                if s > 0:
                    self.f.narrow(d, 0, s).zero_(); self.cnt.narrow(d, 0, s).zero_()
                elif s < 0:
                    self.f.narrow(d, self.size + s, -s).zero_()
                    self.cnt.narrow(d, self.size + s, -s).zero_()
            self.off -= shift

    def write(self, pts, feats):
        _splat(self.f, self.cnt, self._to_cell(pts), feats)

    def read(self, pts):
        return _sample(self.f, self.cnt, self._to_cell(pts))


class EgoMapClip(EgoMapNorth):
    """北锚定 + clipmap 分级:L 级各 size×size,级 l 半径 half·2^l/2^(L-1)。

    写入:进所有覆盖该点的级;读出:最细的覆盖级。近精远粗=空间中心凹。"""

    def __init__(self, c=8, size=32, half=32.0, levels=3):
        self.c, self.size, self.levels = c, size, levels
        self.halves = [half * 2 ** l / 2 ** (levels - 1) for l in range(levels)]
        self.maps = [EgoMapNorth(c, size, h) for h in self.halves]

    def step(self, dpos_world, dyaw_rad=0.0):
        for m in self.maps:
            m.step(dpos_world, dyaw_rad)

    def write(self, pts, feats):
        for m in self.maps:
            r = pts.abs().max(-1).values
            ok = r < m.half * 0.98
            if ok.any():
                m.write(pts[ok], feats[ok])

    def read(self, pts):
        out = torch.zeros(len(pts), self.c)
        done = torch.zeros(len(pts), dtype=torch.bool)
        for m in self.maps:                             # 细→粗
            r = pts.abs().max(-1).values
            ok = (r < m.half * 0.98) & ~done
            if ok.any():
                out[ok] = m.read(pts[ok])
                done |= ok
        return out
