# -*- coding: utf-8 -*-
"""net/calibration.py 验收:相位相关平移恢复 / 键位无关动作→流标定 / FOV 派生 / 输出接口。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.calibration import (SelfCalib, fit_action_flow_map, flow_shift,  # noqa: E402
                             probe_plan)


def _texture(rng, h=90, w=160):
    """带宽受限随机纹理(纯白噪声无亚像素结构,平滑后才像真实画面)。"""
    t = rng.random((h, w)).astype(np.float32)
    k = np.ones((5, 5), np.float32) / 25
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(t, 2, mode="wrap")
    return (sliding_window_view(p, (5, 5)) * k).sum((-1, -2))


def test_flow_shift_recovers_known_translation():
    rng = np.random.default_rng(0)
    a = _texture(rng)
    for sx, sy in [(5, 0), (0, -3), (-7, 4), (12, -9)]:
        b = np.roll(a, (sy, sx), axis=(0, 1))
        dx, dy = flow_shift(a, b)
        assert abs(dx - sx) < 0.5 and abs(dy - sy) < 0.5, (sx, sy, dx, dy)


def test_fit_action_flow_map_keymap_agnostic():
    """8 通道动作,只有 2/5 号是相机(增益 -6.4/+3.1 px/单位),其余是移动/噪声通道。
    标定应恢复增益且不需要知道任何通道叫什么。"""
    rng = np.random.default_rng(1)
    n, a_dim = 400, 8
    acts = rng.normal(size=(n, a_dim)).astype(np.float32)
    m_true = np.zeros((2, a_dim), np.float32)
    m_true[0, 2], m_true[1, 5] = -6.4, 3.1
    flows = acts @ m_true.T + 0.3 * rng.normal(size=(n, 2))
    m, r2 = fit_action_flow_map(flows, acts)
    assert r2 > 0.9
    assert abs(m[0, 2] - (-6.4)) < 0.2 and abs(m[1, 5] - 3.1) < 0.2
    off = np.delete(m, [2, 5], axis=1)
    assert np.abs(off).max() < 0.2                      # 无关通道≈0


def test_self_calib_camera_and_fov():
    """已知合成增益 2.0 px/deg:探针序列后 px_per_deg 与 FOV=H/gain 恢复。"""
    rng = np.random.default_rng(2)
    base = _texture(rng)
    calib = SelfCalib(img_w=160, img_h=90)
    gain = 2.0
    for yaw, pitch in probe_plan(unit_deg=4.0):
        shifted = np.roll(base, (int(round(-pitch * gain)), int(round(-yaw * gain))),
                          axis=(0, 1))
        calib.update_camera(base, shifted, yaw, pitch)
    assert abs(calib.px_per_deg_yaw - gain) < 0.15
    assert abs(calib.px_per_deg_pitch - gain) < 0.15
    assert abs(calib.fov_y_deg - 90 / gain) < 4          # H/gain = 45°


def test_probe_plan_zero_net_drift():
    p = np.array(probe_plan())
    assert p.sum(0)[0] == 0 and p.sum(0)[1] == 0         # 对称对:净漂移 0


def test_outputs_uncalibrated_honest():
    """未标定:PHYSICS 行如实说 uncalibrated,向量有效位为 0——不编数。"""
    c = SelfCalib()
    assert c.physics_line() == "PHYSICS uncalibrated"
    v = c.physics_vector()
    assert v.shape == (6,) and v[1] == 0 and v[3] == 0 and v[5] == 0
    c.update_locomotion(1.2, 10)
    assert "speed=0.120blk/tick" in c.physics_line()
    assert c.physics_vector()[5] == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
