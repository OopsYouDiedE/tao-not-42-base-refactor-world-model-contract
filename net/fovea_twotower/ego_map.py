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
    """北锚定:整格 roll + 亚格偏移寄存器,旋转推迟到读出。世界位移坐标系。

    device=None 默认 CPU;传 "cuda" 时状态张量驻 GPU(读写点须同设备,调用侧保证)。
    """

    def __init__(self, c=8, size=64, half=32.0, device=None):
        self.c, self.size, self.half = c, size, half
        self.res = size / (2 * half)
        self.f = torch.zeros(c, size, size, device=device)
        self.cnt = torch.zeros(1, size, size, device=device)
        self.off = np.zeros(2)                          # 亚格偏移(格单位)

    def _to_cell(self, xy):
        cell = (xy + self.half) * self.res
        return cell + torch.as_tensor(self.off, dtype=torch.float32, device=xy.device)

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


class _RelocMixin:
    """地图反向自定位:当前观测 vs 存图窗口互相关,峰=里程计平移误差。

    前提(MC 成立):yaw=本体感受精确(相机动作积分),漂移只在平移
    (碰撞/动量)——匹配退化为纯平移,北锚定下无旋转搜索维。
    坐标账(闭环推导,勿再改符号):存图内容位于 rel_est+e_hist(写入时误差),
    当前观测位于 rel_est+e_now → patch(x)=map(x-(e_now-e_hist)),
    score(s)=Σ patch(x)·map(x+s) 峰在 s*=-(e_now-e_hist)=**ê(修正量)**;
    施加:p̂ += ê 且 map.step(ê),效果 e_now→e_hist。锚点逻辑:绝对定位
    依赖建图早期 e_hist≈0("家附近先建准")——SLAM 同款。"""

    def relocalize(self, pts, feats, window=4, min_ratio=1.15,
                   subcell=True, min_pts=1.0):
        H = W = self.size
        pf = torch.zeros(self.c, H, W)
        pc = torch.zeros(1, H, W)
        _splat(pf, pc, self._to_cell(pts), feats)
        if pc.sum() < min_pts:
            return None
        mmask = (self.cnt[0] > 0.05).float()
        scores = {}
        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                ms = torch.roll(self.f, (-dy, -dx), (1, 2))     # map(x+s)
                mk = torch.roll(mmask, (-dy, -dx), (0, 1))
                num = float((pf * ms).sum())
                den = float(np.sqrt((pf * pf).sum() *
                                    ((ms * ms) * mk).sum()) + 1e-6)
                scores[(dx, dy)] = num / den
        (bx, by), best = max(scores.items(), key=lambda kv: kv[1])
        second = max(v for k, v in scores.items()
                     if abs(k[0] - bx) + abs(k[1] - by) > 1)
        if best < 1e-6 or best / (second + 1e-9) < min_ratio:
            return None                                          # 峰不够锐,弃
        # 亚格精化:峰邻域一维抛物线拟合(整格量化会给修正加 ±0.5 格抖动)
        def para(m1, c0, p1):
            d = m1 - 2 * c0 + p1
            return 0.5 * (m1 - p1) / d if abs(d) > 1e-9 else 0.0
        fx = fy = 0.0
        if subcell and abs(bx) < window and abs(by) < window:
            fx = para(scores[(bx - 1, by)], best, scores[(bx + 1, by)])
            fy = para(scores[(bx, by - 1)], best, scores[(bx, by + 1)])
        return np.array([bx + np.clip(fx, -.5, .5),
                         by + np.clip(fy, -.5, .5)]) / self.res   # 修正量ê(单位)


class EgoMapNorthLoc(_RelocMixin, EgoMapNorth):
    """自定位版;write_cap>0 = 先写者胜(格累计权重达 cap 即封笔)。

    动机:互相关修正把 e_now 拉向存图误差 e_hist;等权持续写入会让 e_hist
    随漂移污染。封笔使 e_hist 冻结在首访时刻(家附近≈0)——锚定加权最简版。"""

    def __init__(self, *a, write_cap=0.0, **kw):
        super().__init__(*a, **kw)
        self.write_cap = write_cap

    def write(self, pts, feats):
        if self.write_cap <= 0:
            return super().write(pts, feats)
        cell = self._to_cell(pts)
        xi = cell[:, 0].round().long().clamp(0, self.size - 1)
        yi = cell[:, 1].round().long().clamp(0, self.size - 1)
        open_ = self.cnt[0, yi, xi] < self.write_cap
        if open_.any():
            super().write(pts[open_], feats[open_])


class EgoMapClip(EgoMapNorth):
    """北锚定 + clipmap 分级:L 级各 size×size,级 l 半径 half·2^l/2^(L-1)。

    写入:进所有覆盖该点的级;读出:最细的覆盖级。近精远粗=空间中心凹。"""

    def __init__(self, c=8, size=32, half=32.0, levels=3, device=None):
        self.c, self.size, self.levels = c, size, levels
        self.halves = [half * 2 ** l / 2 ** (levels - 1) for l in range(levels)]
        self.maps = [EgoMapNorth(c, size, h, device=device) for h in self.halves]

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
        out = torch.zeros(len(pts), self.c, device=pts.device)
        done = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
        for m in self.maps:                             # 细→粗
            r = pts.abs().max(-1).values
            ok = (r < m.half * 0.98) & ~done
            if ok.any():
                out[ok] = m.read(pts[ok])
                done |= ok
        return out


def _bearing_cn(v):
    """世界位移向量 → 八向中文方位(北=+y,东=+x)。"""
    dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    a = (np.degrees(np.arctan2(v[0], v[1])) + 360) % 360
    return dirs[int((a + 22.5) // 45) % 8]


class MapQuery:
    """慢塔查询 API:几何计算全在此(确定性代码),慢塔只消费模板文本。

    契约(scale-plan §4.9 铁律同款):慢塔永远不碰坐标算术——距离/方位/
    覆盖由本层算好,LLM 做离散规划。channels 前 n_cls 维=类 softmax。"""

    def __init__(self, m, class_names):
        self.m, self.names = m, class_names

    def _grid(self, mp):
        cnt = mp.cnt[0]
        f = mp.f / (cnt[None] + 1e-6)
        return f, cnt

    def nearest(self, cls, thr=0.45):
        """最近的 cls 实例:距离+方位(模板文本) / None。"""
        k = self.names.index(cls)
        best = None
        for mp in (self.m.maps if hasattr(self.m, "maps") else [self.m]):
            f, cnt = self._grid(mp)
            mask = (f[k] > thr) & (cnt > 0.05)
            if not mask.any():
                continue
            ys, xs = torch.nonzero(mask, as_tuple=True)
            wx = (xs.float() + 0.5) / mp.res - mp.half - mp.off[0] / mp.res
            wy = (ys.float() + 0.5) / mp.res - mp.half - mp.off[1] / mp.res
            d = torch.sqrt(wx ** 2 + wy ** 2)
            i = int(d.argmin())
            cand = (float(d[i]), np.array([float(wx[i]), float(wy[i])]))
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            return None, f"视界内未见{cls}"
        d, v = best
        return v, f"最近{cls}:{_bearing_cn(v)}{d:.0f}格"

    def survey(self, sectors=8):
        """探索覆盖:八扇区已写格占比 → 未探索方向列表(文本)。"""
        mp = self.m.maps[-1] if hasattr(self.m, "maps") else self.m
        cnt = mp.cnt[0]
        H = W = mp.size
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W),
                                indexing="ij")
        wx = (xx.float() + 0.5) / mp.res - mp.half
        wy = (yy.float() + 0.5) / mp.res - mp.half
        ang = (torch.rad2deg(torch.atan2(wx, wy)) + 360) % 360
        sec = ((ang + 22.5) // 45).long() % sectors
        names8 = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
        unexp = [names8[s] for s in range(sectors)
                 if float((cnt > 0.05)[sec == s].float().mean()) < 0.15]
        return unexp, ("未探索方向:" + "/".join(unexp) if unexp
                       else "四周已大体探索")
