# -*- coding: utf-8 -*-
"""net/calibration.py 验收:平移+置信 / 键位无关标定 / 延迟扫描 / 响应曲线 /
控制模式 / GUI 光标增益 / FOV 派生 / 诚实降级。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.calibration import (SelfCalib, control_mode, cursor_gain_from_diffs,  # noqa: E402
                             fit_action_flow_map, fit_latency,
                             fit_response_curve, flow_shift, probe_plan,
                             probe_plan_multi)


def _texture(rng, h=90, w=160):
    """带宽受限随机纹理(纯白噪声无亚像素结构,平滑后才像真实画面)。"""
    t = rng.random((h, w)).astype(np.float32)
    k = np.ones((5, 5), np.float32) / 25
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(t, 2, mode="wrap")
    return (sliding_window_view(p, (5, 5)) * k).sum((-1, -2))


def test_flow_shift_recovers_translation_with_confidence():
    rng = np.random.default_rng(0)
    a = _texture(rng)
    for sx, sy in [(5, 0), (0, -3), (-7, 4), (12, -9)]:
        b = np.roll(a, (sy, sx), axis=(0, 1))
        dx, dy, conf = flow_shift(a, b)
        assert abs(dx - sx) < 0.5 and abs(dy - sy) < 0.5, (sx, sy, dx, dy)
        assert conf > 1.5                               # 干净平移:峰远高于次峰


def test_flow_confidence_gates_garbage():
    """两帧独立噪声(无真实平移):conf 低,SelfCalib 丢弃证据保持 uncalibrated。"""
    rng = np.random.default_rng(1)
    a, b = _texture(rng), _texture(np.random.default_rng(99))
    _, _, conf = flow_shift(a, b)
    c = SelfCalib()
    c.update_camera(a, b, 4.0, 0.0)
    assert c.px_per_deg_yaw is None or conf >= c.min_conf  # 低置信 ⇒ 不产出增益


def test_update_camera_gui_gate():
    rng = np.random.default_rng(0)
    a = _texture(rng)
    b = np.roll(a, (0, -8), axis=(0, 1))
    c = SelfCalib()
    c.update_camera(a, b, 4.0, 0.0, in_gui=True)        # GUI 态:证据被拒
    assert c.px_per_deg_yaw is None


def test_fit_action_flow_map_keymap_agnostic():
    rng = np.random.default_rng(1)
    n, a_dim = 400, 8
    acts = rng.normal(size=(n, a_dim)).astype(np.float32)
    m_true = np.zeros((2, a_dim), np.float32)
    m_true[0, 2], m_true[1, 5] = -6.4, 3.1
    flows = acts @ m_true.T + 0.3 * rng.normal(size=(n, 2))
    m, r2 = fit_action_flow_map(flows, acts)
    assert r2 > 0.9
    assert abs(m[0, 2] - (-6.4)) < 0.2 and abs(m[1, 5] - 3.1) < 0.2
    assert np.abs(np.delete(m, [2, 5], axis=1)).max() < 0.2


def test_fit_latency_recovers_lag():
    rng = np.random.default_rng(2)
    n, lag = 30, 2
    cmds = np.zeros((n, 2), np.float32)
    cmds[:, 0] = rng.normal(size=n)
    flows = np.zeros((n, 2), np.float32)
    flows[lag:, 0] = -2.0 * cmds[:n - lag, 0] + 0.1 * rng.normal(size=n - lag)
    k, corr = fit_latency(flows, cmds, max_lag=3)
    assert k == lag and corr > 0.8


def test_fit_response_curve_deadzone_expo():
    rng = np.random.default_rng(3)
    c = rng.uniform(0.5, 10, 200)
    f = 2.0 * np.maximum(c - 1.0, 0) ** 1.5 + 0.2 * rng.normal(size=200)
    p = fit_response_curve(c, f)
    assert abs(p["deadzone"] - 1.0) < 0.6
    assert abs(p["expo"] - 1.5) < 0.25
    assert abs(p["gain"] - 2.0) < 0.6
    lin = fit_response_curve(np.full(10, 4.0), np.full(10, 8.0))  # 全同幅度 ⇒ 线性退化
    assert lin["expo"] == 1.0 and abs(lin["gain"] - 2.0) < 1e-6


def test_control_mode_pulse():
    assert control_mode(mag_during=8.0, mag_after=0.3) == "position"   # 鼠标:骤停
    assert control_mode(mag_during=8.0, mag_after=6.0) == "velocity"   # 摇杆:持续
    assert control_mode(mag_during=0.1, mag_after=0.0) is None          # 没响应:判不了


def test_cursor_gain_midpoint_method():
    """3px 亮点光标 p0→p1→p2(等量命令 4 单位,真增益 2px/单位),中点法恢复增益。"""
    def frame(px, py):
        f = np.zeros((100, 100), np.float32)
        f[py - 1:py + 2, px - 1:px + 2] = 255
        return f
    g = cursor_gain_from_diffs(frame(20, 30), frame(28, 30), frame(36, 30), cmd_units=4.0)
    assert g is not None and abs(g[0] - 2.0) < 0.1 and abs(g[1]) < 0.1
    assert cursor_gain_from_diffs(frame(20, 30), frame(20, 30), frame(20, 30), 4.0) is None


def test_self_calib_camera_fov_and_curve():
    rng = np.random.default_rng(4)
    base = _texture(rng)
    calib = SelfCalib(img_w=160, img_h=90)
    gain = 2.0
    for yaw, pitch in probe_plan_multi((2.0, 4.0, 8.0)):
        shifted = np.roll(base, (int(round(-pitch * gain)), int(round(-yaw * gain))),
                          axis=(0, 1))
        calib.update_camera(base, shifted, yaw, pitch)
    assert abs(calib.px_per_deg_yaw - gain) < 0.2
    assert abs(calib.fov_y_deg - 90 / gain) < 5
    calib.fit_curve()
    assert calib.curve_yaw is not None and abs(calib.curve_yaw["expo"] - 1.0) < 0.2


def test_probe_plans_zero_net_drift():
    for p in (np.array(probe_plan()), np.array(probe_plan_multi((2.0, 4.0, 8.0)))):
        assert p.sum(0)[0] == 0 and p.sum(0)[1] == 0


def test_outputs_uncalibrated_honest():
    c = SelfCalib()
    assert c.physics_line() == "PHYSICS uncalibrated"
    v = c.physics_vector()
    assert v.shape == (10,) and v[1] == 0 and v[3] == 0 and v[5] == 0 and v[7] == 0
    c.update_locomotion(1.2, 10)
    c.latency_ticks, c.mode = 1, "velocity"
    line = c.physics_line()
    assert "speed=0.120blk/tick" in line and "latency=1tick" in line and "mode=velocity" in line
    v2 = c.physics_vector()
    assert v2[5] == 1.0 and v2[7] == 1.0 and v2[8] == 1.0 and v2[9] == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
