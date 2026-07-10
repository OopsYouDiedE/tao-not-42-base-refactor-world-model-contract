# -*- coding: utf-8 -*-
"""感知运动自标定:模型自己做实验,在"脑内"建立场景物理参数(2026-07-10;同日加固)。

立场(苦涩的教训 + 用户裁决):分辨率、FOV、相机增益、控制延迟、响应曲线、控制模式
(位置增量 vs 速度)、步速、键位语义都是**环境的参数**,不是可以写死进代码或绑死进
训练集的常量。智能体用已知动作当探针,从观测响应里**测**出参数,存成随环境走的状态。

探针的隐含假设与自检(加固批,2026-07-10 后半):
  · 场景基本静止 → flow_shift 带峰锐度置信度,低置信证据被丢弃(不污染中位数);
  · 命令→响应延迟未知 → fit_latency 延迟扫描,增益配对按测得延迟错位;
  · 响应可能非线性(鼠标加速度/摇杆死区) → fit_response_curve 三参数曲线;
  · 位置增量(鼠标) vs 速度控制(手柄摇杆) → control_mode 脉冲探针判别;
  · GUI 态无相机响应 → update_camera(in_gui=True) 跳过;GUI 光标增益另有
    cursor_gain_from_diffs(帧差质心中点法,无需光标模板)。
**测不出就如实置 None/无效位,不编数**——未标定时快慢塔各自显式降级。

纪律:相机/FOV/延迟/模式标定只用观测(可进部署回路);步速标定当前用 env pose
(特权,只进训练侧,标出的常量随 calib 状态携带;将来可换视觉里程计)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img.astype(np.float32).mean(-1)
    return img.astype(np.float32)


def flow_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """相位相关全局平移:返回 (dx, dy, conf)。conf = 主峰/邻域外次峰,≈1 即不可信。

    a/b: [H,W] 或 [H,W,3]。Hanning 窗压边缘泄漏;峰邻域一维抛物线亚像素精化
    (与 ego_map.relocalize 同款,整格量化会给标定加 ±0.5px 抖动)。
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
    peak = float(corr[py, px])
    masked = corr.copy()                          # 峰 3px 邻域外的次峰
    yy = (np.arange(h)[:, None] - py + h // 2) % h - h // 2
    xx = (np.arange(w)[None, :] - px + w // 2) % w - w // 2
    masked[(np.abs(yy) <= 3) & (np.abs(xx) <= 3)] = -np.inf
    second = float(masked.max())
    conf = peak / (abs(second) + 1e-9) if second > 0 else 99.0

    def para(m1, c0, p1):
        d = m1 - 2 * c0 + p1
        return 0.5 * (m1 - p1) / d if abs(d) > 1e-12 else 0.0

    fx = para(corr[py, (px - 1) % w], peak, corr[py, (px + 1) % w])
    fy = para(corr[(py - 1) % h, px], peak, corr[(py + 1) % h, px])
    dx = px + np.clip(fx, -.5, .5)
    dy = py + np.clip(fy, -.5, .5)
    if dx > w / 2:
        dx -= w                                   # 环绕 → 有符号
    if dy > h / 2:
        dy -= h
    return -float(dx), -float(dy), conf           # 约定:b 内容右移 ⇒ dx>0


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


def fit_latency(flows: np.ndarray, cmds: np.ndarray,
                max_lag: int = 3) -> tuple[int, float]:
    """延迟扫描:cmd[t] 与 flow[t+k] 的相关随 k 扫描,峰即控制延迟(tick)。

    flows[N,2] = 相邻帧对的流;cmds[N,2] = 对应 tick 发出的 (yaw,pitch) 命令。
    返回 (lag, |corr| 峰值);峰值 <0.3 视为未测出(调用侧判)。
    """
    n = len(cmds)
    best = (0, 0.0)
    for k in range(max_lag + 1):
        if n - k < 3:
            break
        cs = []
        for ax in range(2):
            c, f = cmds[:n - k, ax], flows[k:, ax]
            if c.std() < 1e-9 or f.std() < 1e-9:
                continue                          # 该轴没打探针,不计分也不摊薄
            cs.append(abs(float(np.corrcoef(c, f)[0, 1])))
        s = float(np.mean(cs)) if cs else 0.0
        if s > best[1]:
            best = (k, s)
    return best


def fit_response_curve(cmds: np.ndarray, flows: np.ndarray) -> dict:
    """非线性响应:|flow| = g · max(|cmd|−d, 0)^e 三参数拟合(网格+闭式 g,无 scipy)。

    覆盖鼠标加速度(e>1)、摇杆死区(d>0)、线性(d=0,e=1)。返回
    {deadzone, gain, expo, mse};样本 <6 或全同幅度时退化为线性增益。
    """
    c, f = np.abs(np.asarray(cmds, float)), np.abs(np.asarray(flows, float))
    ok = c > 1e-9
    c, f = c[ok], f[ok]
    if len(c) < 6 or np.ptp(c) < 1e-6:
        g = float((c * f).sum() / ((c * c).sum() + 1e-9))
        return dict(deadzone=0.0, gain=g, expo=1.0, mse=float(((g * c - f) ** 2).mean()))
    best = None
    for d in np.linspace(0.0, 0.5 * c.max(), 11):
        x = np.maximum(c - d, 1e-9)
        for e in np.linspace(0.5, 2.0, 16):
            xe = x ** e
            g = float((xe * f).sum() / ((xe * xe).sum() + 1e-9))
            mse = float(((g * xe - f) ** 2).mean())
            if best is None or mse < best["mse"]:
                best = dict(deadzone=float(d), gain=g, expo=float(e), mse=mse)
    return best


def wrap_deg(a):
    """角度包络到 (-180, 180]。标量或 ndarray。"""
    return -((180.0 - np.asarray(a, np.float64)) % 360.0 - 180.0)


def fit_angle_map(env_deg: np.ndarray, geom_deg: np.ndarray) -> tuple[int, float, float]:
    """拟合环境角与几何角的映射 env ≈ sign·geom + offset(圆周量,符号±1 + 常量偏置)。

    用途:环境上报的 yaw/pitch 零点与旋向是环境参数,不写死——用成对样本
    (env 上报角, 由特权几何算出的角) 实测。返回 (sign, offset_deg, resid_deg);
    resid = 拟合后残差绝对值均值(度),调用侧据此决定采信与否。

    Parameters
    ----------
    env_deg  : [N] float,环境上报角(度)
    geom_deg : [N] float,几何参考角(度,同一时刻)
    """
    env = np.asarray(env_deg, np.float64)
    geo = np.asarray(geom_deg, np.float64)
    best = None
    for s in (1, -1):
        r = wrap_deg(env - s * geo)                   # 应为常量 offset
        c = np.radians(r)
        off = float(np.degrees(np.arctan2(np.sin(c).mean(), np.cos(c).mean())))
        resid = float(np.abs(wrap_deg(r - off)).mean())
        if best is None or resid < best[2]:
            best = (s, off, resid)
    return best


def control_mode(mag_during: float, mag_after: float,
                 ratio: float = 0.3) -> str | None:
    """脉冲探针判别:命令停发后流仍持续 ⇒ 速度控制(摇杆);骤停 ⇒ 位置增量(鼠标)。"""
    if mag_during < 0.5:
        return None                               # 脉冲本身没响应,判不了
    return "velocity" if mag_after > ratio * mag_during else "position"


def _diff_centroid(a: np.ndarray, b: np.ndarray, thr: float = 12.0):
    """两帧差异像素的加权质心(GUI 光标探针用)。差异太少返回 None。"""
    d = np.abs(_gray(a) - _gray(b))
    m = d > max(thr, float(d.max()) * 0.3)
    if m.sum() < 4:
        return None
    ys, xs = np.nonzero(m)
    w = d[m]
    return np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])


def cursor_gain_from_diffs(f0: np.ndarray, f1: np.ndarray, f2: np.ndarray,
                           cmd_units: float):
    """GUI 光标增益(px/单位):连发两次**等量**命令,帧差质心中点法,无需光标模板。

    帧差含"消失旧位 + 出现新位"两团,其质心 = 中点 m_i=(p_{i-1}+p_i)/2;
    m2−m1 = (p2−p0)/2 = gain·cmd ⇒ gain = (m2−m1)/cmd。返回 [gx,gy] 或 None。
    """
    c1 = _diff_centroid(f0, f1)
    c2 = _diff_centroid(f1, f2)
    if c1 is None or c2 is None or abs(cmd_units) < 1e-9:
        return None
    return (c2 - c1) / cmd_units


@dataclass
class SelfCalib:
    """随环境走的物理参数状态。未标定字段为 None;estimate 类方法只追加证据。"""

    img_w: int = 160
    img_h: int = 90
    min_conf: float = 1.2                         # flow 置信门槛(峰/次峰)
    px_per_deg_yaw: float | None = None
    px_per_deg_pitch: float | None = None
    latency_ticks: int | None = None
    mode: str | None = None                       # position | velocity
    curve_yaw: dict | None = None                 # fit_response_curve 结果
    step_blocks: float | None = None              # 格/tick(训练侧 pose 标定)
    eye_h: float = 1.62                           # 先验默认,可被证据更新
    _yaw_ev: list = field(default_factory=list)
    _pitch_ev: list = field(default_factory=list)
    _yaw_pairs: list = field(default_factory=list)  # (cmd, |flow_x|) 供曲线拟合

    # ── 在线:已知命令 → 流响应 ─────────────────────────────
    def update_camera(self, f0: np.ndarray, f1: np.ndarray, cmd_yaw_deg: float,
                      cmd_pitch_deg: float, in_gui: bool = False) -> None:
        """低置信/GUI 态的证据被丢弃——宁可 uncalibrated,不污染中位数。"""
        if in_gui:
            return
        dx, dy, conf = flow_shift(f0, f1)
        if conf < self.min_conf:
            return
        if abs(cmd_yaw_deg) > 1e-3 and abs(cmd_pitch_deg) < 1e-3:
            self._yaw_ev.append(-dx / cmd_yaw_deg)   # 右转 ⇒ 场景左移
            self._yaw_pairs.append((cmd_yaw_deg, abs(dx)))
        if abs(cmd_pitch_deg) > 1e-3 and abs(cmd_yaw_deg) < 1e-3:
            self._pitch_ev.append(-dy / cmd_pitch_deg)
        if self._yaw_ev:
            self.px_per_deg_yaw = float(np.median(self._yaw_ev))
        if self._pitch_ev:
            self.px_per_deg_pitch = float(np.median(self._pitch_ev))

    def fit_curve(self) -> None:
        """幅度多样时拟合非线性响应曲线(死区/增益/指数)。"""
        if len(self._yaw_pairs) >= 6:
            arr = np.asarray(self._yaw_pairs, np.float64)
            self.curve_yaw = fit_response_curve(arr[:, 0], arr[:, 1])

    def update_locomotion(self, dpos_blocks: float, ticks: int) -> None:
        """训练侧:pose 位移 / tick(特权信息只进训练侧,产出常量随状态携带)。"""
        if ticks > 0:
            self.step_blocks = float(dpos_blocks / ticks)

    # ── 派生量 ─────────────────────────────────────────────
    @property
    def yaw_sign(self) -> int | None:
        """cmd 正 yaw 是否顺时针右转(北锚定地图的正 yaw 口径)。几何普适:右转 ⇒
        场景左移 ⇒ 实测增益为正。纯观测导出;未标定返回 None(调用侧显式降级)。"""
        g = self.px_per_deg_yaw
        return None if g is None else (1 if g > 0 else -1)

    @property
    def pitch_sign(self) -> int | None:
        """cmd 正 pitch 是否向下(IPM/地图的正 pitch 口径)。向下看 ⇒ 场景上移 ⇒
        实测增益为正。纯观测导出;未标定返回 None。"""
        g = self.px_per_deg_pitch
        return None if g is None else (1 if g > 0 else -1)

    @property
    def fov_y_deg(self) -> float | None:
        """小角近似:竖直 FOV ≈ H / (px/deg)。像素流是切线关系,小角段近似线性。"""
        p = self.px_per_deg_pitch or self.px_per_deg_yaw
        return self.img_h / abs(p) if p else None            # 符号归 yaw_sign/pitch_sign

    def physics_vector(self) -> np.ndarray:
        """喂 token 塔的物理 token(缺测项置 0,带 1/0 有效位)。[10] float32。"""
        f = self.fov_y_deg
        lat = self.latency_ticks
        return np.array([
            (self.px_per_deg_yaw or 0.0) / 10.0, 1.0 if self.px_per_deg_yaw else 0.0,
            (f or 0.0) / 100.0, 1.0 if f else 0.0,
            (self.step_blocks or 0.0) * 5.0, 1.0 if self.step_blocks else 0.0,
            (lat or 0) / 4.0, 1.0 if lat is not None else 0.0,
            1.0 if self.mode == "velocity" else 0.0, 1.0 if self.mode else 0.0,
        ], np.float32)

    def physics_line(self) -> str:
        """喂慢塔 prompt 的 PHYSICS 行(未标定如实说,不编数)。"""
        parts = []
        if self.px_per_deg_yaw:
            parts.append(f"cam_gain={self.px_per_deg_yaw:.2f}px/deg")
        if self.fov_y_deg:
            parts.append(f"fov_y={self.fov_y_deg:.0f}deg")
        if self.latency_ticks is not None:
            parts.append(f"latency={self.latency_ticks}tick")
        if self.mode:
            parts.append(f"mode={self.mode}")
        if self.curve_yaw and abs(self.curve_yaw["expo"] - 1.0) > 0.15:
            parts.append(f"expo={self.curve_yaw['expo']:.2f}")
        if self.step_blocks:
            parts.append(f"speed={self.step_blocks:.3f}blk/tick")
        return "PHYSICS " + (" ".join(parts) if parts else "uncalibrated")


def probe_plan(unit_deg: float = 4.0) -> list[tuple[float, float]]:
    """单幅度标定探针 [(yaw,pitch)…]:小角对称对(净漂移 0),交替轴向。"""
    u = unit_deg
    return [(u, 0), (-u, 0), (-u, 0), (u, 0),
            (0, u), (0, -u), (0, -u), (0, u)]


def probe_plan_multi(units: tuple[float, ...] = (2.0, 4.0, 8.0)) -> list[tuple[float, float]]:
    """多幅度探针(供响应曲线拟合):逐幅度拼接,整体净漂移仍为 0。"""
    plan: list[tuple[float, float]] = []
    for u in units:
        plan += probe_plan(u)
    return plan
