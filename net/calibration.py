# -*- coding: utf-8 -*-
"""感知运动自标定:模型自己做实验,在"脑内"建立场景物理参数(2026-07-10)。

立场(苦涩的教训 + 用户裁决):分辨率、FOV、相机增益、步速、键位语义都是**环境的
参数**,不是可以写死进代码或绑死进训练集的常量。VPT 数据是 px 单位、别的游戏是别的
FOV、键位可以任意重映射——正确形态是:智能体用已知动作当探针,从观测的光流响应里
**测**出这些参数,存成随环境走的状态(physics token / PHYSICS 行),喂给快塔与慢塔。

三件东西,全部纯观测、零外部依赖:
  flow_shift            相位相关求两帧全局平移(FFT,亚像素抛物线精化)
  fit_action_flow_map   通用键位无关标定:flow ~ M·action 最小二乘 ⇒ 哪些动作通道
                        动相机、增益多大——不用知道通道叫什么名字
  SelfCalib             在线标定状态:发已知相机命令→测流→px/单位→FOV;
                        物理参数向量(喂 token 塔 geo)与 PHYSICS 行(喂慢塔 prompt)

纪律:相机/FOV 标定只用观测(可进部署回路);步速标定当前用 env pose(特权,
只进训练侧,标出的常量随 calib 状态携带;将来可换视觉里程计)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img.astype(np.float32).mean(-1)
    return img.astype(np.float32)


def flow_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """相位相关:返回 b 相对 a 的全局平移 (dx, dy),像素,亚像素精化。

    a/b: [H,W] 或 [H,W,3]。窗函数压边缘泄漏;峰邻域一维抛物线拟合(与
    ego_map.relocalize 同款,整格量化会给标定加 ±0.5px 抖动)。
    """
    a, b = _gray(a), _gray(b)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-9
    corr = np.fft.irfft2(r, s=(h, w))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)

    def para(m1, c0, p1):
        d = m1 - 2 * c0 + p1
        return 0.5 * (m1 - p1) / d if abs(d) > 1e-12 else 0.0

    fx = para(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    fy = para(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dx = px + np.clip(fx, -.5, .5)
    dy = py + np.clip(fy, -.5, .5)
    if dx > w / 2:
        dx -= w                                   # 环绕 → 有符号
    if dy > h / 2:
        dy -= h
    return -float(dx), -float(dy)                 # 约定:b 内容右移 ⇒ dx>0


def fit_action_flow_map(flows: np.ndarray, actions: np.ndarray,
                        lam: float = 1e-4) -> tuple[np.ndarray, float]:
    """键位无关标定:最小二乘解 flow[N,2] ≈ actions[N,A] @ M.T,返回 (M[2,A], R²)。

    M 的第 j 列 = 通道 j 每单位动作引起的像素流——不需要知道通道语义;
    相机通道自然得大系数,移动/无关通道≈0。适配任意数据集的任意动作布局。
    """
    a = actions.astype(np.float64)
    f = flows.astype(np.float64)
    am, fm = a.mean(0), f.mean(0)
    ac, fc = a - am, f - fm
    m = np.linalg.solve(ac.T @ ac + lam * len(a) * np.eye(a.shape[1]),
                        ac.T @ fc).T                              # [2,A]
    pred = ac @ m.T + fm
    ss_res = ((f - pred) ** 2).sum()
    ss_tot = ((f - fm) ** 2).sum() + 1e-12
    return m.astype(np.float32), float(1 - ss_res / ss_tot)


@dataclass
class SelfCalib:
    """随环境走的物理参数状态。未标定字段为 None;estimate 类方法只追加证据。"""

    img_w: int = 160
    img_h: int = 90
    px_per_deg_yaw: float | None = None
    px_per_deg_pitch: float | None = None
    step_blocks: float | None = None              # 格/tick(训练侧 pose 标定)
    eye_h: float = 1.62                           # 先验默认,可被证据更新
    _yaw_ev: list = field(default_factory=list)
    _pitch_ev: list = field(default_factory=list)

    # ── 在线:已知命令 → 流响应 ─────────────────────────────
    def update_camera(self, f0: np.ndarray, f1: np.ndarray,
                      cmd_yaw_deg: float, cmd_pitch_deg: float) -> None:
        dx, dy = flow_shift(f0, f1)
        if abs(cmd_yaw_deg) > 1e-3 and abs(cmd_pitch_deg) < 1e-3:
            self._yaw_ev.append(-dx / cmd_yaw_deg)   # 右转 ⇒ 场景左移
        if abs(cmd_pitch_deg) > 1e-3 and abs(cmd_yaw_deg) < 1e-3:
            self._pitch_ev.append(-dy / cmd_pitch_deg)
        if self._yaw_ev:
            self.px_per_deg_yaw = float(np.median(self._yaw_ev))
        if self._pitch_ev:
            self.px_per_deg_pitch = float(np.median(self._pitch_ev))

    def update_locomotion(self, dpos_blocks: float, ticks: int) -> None:
        """训练侧:pose 位移 / tick(特权信息只进训练侧,产出常量随状态携带)。"""
        if ticks > 0:
            self.step_blocks = float(dpos_blocks / ticks)

    # ── 派生量 ─────────────────────────────────────────────
    @property
    def fov_y_deg(self) -> float | None:
        """小角近似:竖直 FOV ≈ H / (px/deg)。像素流是切线关系,小角段近似线性。"""
        p = self.px_per_deg_pitch or self.px_per_deg_yaw
        return self.img_h / p if p else None

    def physics_vector(self) -> np.ndarray:
        """喂 token 塔的物理 token(缺测项置 0,带 1/0 有效位)。[6] float32。"""
        f = self.fov_y_deg
        return np.array([
            (self.px_per_deg_yaw or 0.0) / 10.0, 1.0 if self.px_per_deg_yaw else 0.0,
            (f or 0.0) / 100.0, 1.0 if f else 0.0,
            (self.step_blocks or 0.0) * 5.0, 1.0 if self.step_blocks else 0.0,
        ], np.float32)

    def physics_line(self) -> str:
        """喂慢塔 prompt 的 PHYSICS 行(未标定如实说,不编数)。"""
        parts = []
        if self.px_per_deg_yaw:
            parts.append(f"cam_gain={self.px_per_deg_yaw:.2f}px/deg")
        if self.fov_y_deg:
            parts.append(f"fov_y={self.fov_y_deg:.0f}deg")
        if self.step_blocks:
            parts.append(f"speed={self.step_blocks:.3f}blk/tick")
        return "PHYSICS " + (" ".join(parts) if parts else "uncalibrated")


def probe_plan(unit_deg: float = 4.0) -> list[tuple[float, float]]:
    """开局标定探针动作序列 [(yaw,pitch)…]:小角对称对(净漂移 0),交替轴向。"""
    u = unit_deg
    return [(u, 0), (-u, 0), (-u, 0), (u, 0),
            (0, u), (0, -u), (0, -u), (0, u)]
