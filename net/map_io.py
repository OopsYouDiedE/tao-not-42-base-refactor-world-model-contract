# -*- coding: utf-8 -*-
"""地图读写接线(2026-07-10 设计文档 §3.2/§7 的落地;视觉前端无关)。

四件东西,全部可微、全部与 DINO/YOLOE 前端解耦:
  ipm_ground  屏幕像素 → 世界位移系地面点(精确针孔几何;结构承载几何,内容靠学)
  MapWriter   逐位置视觉特征 → W_c 投影(唯一可学件) → EgoMapClip.write
  MapReader   EgoMapClip 三级 → K 个 token [feat_c ⊕ pos ⊕ level](进 cross-attention KV)
  AimPin      慢塔 aim 像素 → ipm → 世界系钉点(B1:aim 不做屏幕系零阶保持)

坐标约定(与 net/fovea_twotower/ego_map.py 一致):北=+y,东=+x,世界位移系
(北锚定,自身恒在原点);yaw=0 朝北、顺时针为正(向东转 yaw=+90°);
pitch=0 平视、**向下为正**(MC 口径);单位:格(block)、弧度入参。
CraftGround 实际 yaw 的零点/符号与此差一个常量标定——标定属训练侧,不进本模块。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from net.fovea_twotower.ego_map import EgoMapClip


def ipm_ground(uv: torch.Tensor, yaw: float, pitch: float, eye_h: float = 1.62,
               fov_y_deg: float = 70.0, aspect: float = 16 / 9,
               max_range: float = 64.0) -> tuple[torch.Tensor, torch.Tensor]:
    """归一化屏幕坐标 [N,2](u,v ∈ [0,1],(0,0)=左上)→ 地面点 [N,2](世界位移系 x东,y北)。

    针孔逆投影 + 地平面相交:视线方向在相机系为
        d_cam = [tan(fov_x/2)·(2u-1), tan(fov_y/2)·(1-2v), 1]   (x右, y上, z前)
    经 pitch(绕右轴,向下为正)、yaw(绕竖轴,顺时针为正)旋到世界系,与 y=-eye_h 平面求交。
    返回 (pts[N,2], valid[N]):视线不朝下(d_y≥-1e-6)或交点超 max_range 的置 invalid。
    """
    u, v = uv[:, 0], uv[:, 1]
    tan_y = math.tan(math.radians(fov_y_deg) / 2)
    tan_x = tan_y * aspect
    dx = tan_x * (2 * u - 1)                       # 相机系:右+
    dy = tan_y * (1 - 2 * v)                       # 相机系:上+
    dz = torch.ones_like(dx)                       # 相机系:前+
    cp, sp = math.cos(pitch), math.sin(pitch)      # pitch 向下为正:前下压
    y1 = dy * cp - dz * sp
    z1 = dy * sp + dz * cp
    cy, sy = math.cos(yaw), math.sin(yaw)          # yaw=0 朝北(+y),顺时针为正
    east = dx * cy + z1 * sy
    north = -dx * sy + z1 * cy
    t = -eye_h / y1.clamp(max=-1e-6)               # 与地平面 y=-eye_h 相交
    valid = (y1 < -1e-6) & (t > 0) & (t * torch.hypot(east, north) < max_range * 4)
    pts = torch.stack([east * t, north * t], dim=-1)
    rng = pts.norm(dim=-1)
    valid = valid & (rng < max_range)
    return torch.where(valid[:, None], pts, torch.zeros_like(pts)), valid


class MapWriter(nn.Module):
    """视觉特征落图。W_c: feat_dim→c 是本模块唯一可学参数,由读路径回传梯度训练。"""

    def __init__(self, feat_dim: int, c: int = 8):
        super().__init__()
        self.w_c = nn.Linear(feat_dim, c, bias=False)

    def forward(self, m: EgoMapClip, uv: torch.Tensor, feats: torch.Tensor,
                yaw: float, pitch: float, **ipm_kw) -> int:
        """uv[N,2] 归一屏幕坐标 + feats[N,feat_dim] → 写入 m。返回有效写入点数。"""
        pts, valid = ipm_ground(uv, yaw, pitch, **ipm_kw)
        if not bool(valid.any()):
            return 0
        m.write(pts[valid], self.w_c(feats[valid]))
        return int(valid.sum())


class MapReader(nn.Module):
    """EgoMapClip → [K, token_dim] 地图 token(K = grid² × levels)。

    每级取 grid×grid 均匀采样点(覆盖该级半径),token = W_o([feat_c, x/half, y/half, lv]);
    读出经 grid_sample 可微,梯度可回传到 MapWriter.w_c(经地图状态张量)。
    注意:地图状态在 rollout 间是**常量缓冲**;梯度只在同一计算图内(如 BC 逐段重放)有效。
    """

    def __init__(self, c: int = 8, d_out: int = 64, grid: int = 4):
        super().__init__()
        self.grid, self.d_out = grid, d_out
        self.proj = nn.Linear(c + 3, d_out)

    def forward(self, m: EgoMapClip) -> torch.Tensor:
        toks = []
        n_lv = len(m.maps)
        for lv, mp in enumerate(m.maps):            # 细 → 粗
            g = self.grid
            lin = torch.linspace(-0.9, 0.9, g) * mp.half
            yy, xx = torch.meshgrid(lin, lin, indexing="ij")
            pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            feat = mp.read(pts)                      # [g², c] 可微
            pos = pts / mp.half                      # [-0.9,0.9] 归一
            lvc = torch.full((g * g, 1), lv / max(n_lv - 1, 1))
            toks.append(torch.cat([feat, pos, lvc], dim=-1))
        return self.proj(torch.cat(toks, dim=0))     # [grid²·levels, d_out]


class AimPin:
    """慢塔 aim 的世界系钉点(B1)。零网格零模糊:直接存世界位移坐标,step 精确账本。

    set(uv, yaw, pitch) 下发 tick 调用一次;step(dpos_world) 每 tick 与地图同步调用;
    get() → (xy[2] 世界位移系, age_ticks) 或 (None, None)(未钉/无效/过期)。
    """

    def __init__(self, ttl_ticks: int = 200):
        self.ttl = ttl_ticks
        self.xy: torch.Tensor | None = None
        self.age = 0

    def set(self, uv_xy: tuple[float, float], yaw: float, pitch: float, **ipm_kw) -> bool:
        uv = torch.tensor([[uv_xy[0], uv_xy[1]]], dtype=torch.float32)
        pts, valid = ipm_ground(uv, yaw, pitch, **ipm_kw)
        if bool(valid[0]):
            self.xy, self.age = pts[0].clone(), 0
            return True
        return False                                  # 空中目标/指天:不钉,退回无钉状态

    def step(self, dpos_world) -> None:
        if self.xy is not None:
            self.xy = self.xy - torch.as_tensor(dpos_world, dtype=torch.float32)
            self.age += 1
            if self.age > self.ttl:
                self.xy = None

    def get(self) -> tuple[torch.Tensor | None, int | None]:
        return (self.xy, self.age) if self.xy is not None else (None, None)
