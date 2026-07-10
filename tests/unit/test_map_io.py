# -*- coding: utf-8 -*-
"""net/map_io.py 验收:IPM 精确几何 / 写读闭环 / W_c 梯度 / AimPin 运动账本。"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.fovea_twotower.ego_map import EgoMapClip  # noqa: E402
from net.map_io import AimPin, MapReader, MapWriter, ipm_ground  # noqa: E402


def test_ipm_exact_geometry():
    """屏幕中心 + pitch=45° + h=1.62 ⇒ 正前方 1.62 格(tan45=1);yaw 旋转到位。"""
    uv = torch.tensor([[0.5, 0.5]])
    pts, ok = ipm_ground(uv, yaw=0.0, pitch=math.radians(45), eye_h=1.62)
    assert bool(ok[0])
    assert abs(float(pts[0, 0])) < 1e-5                      # 东分量 0
    assert abs(float(pts[0, 1]) - 1.62) < 1e-4               # 北 1.62
    pts_e, ok_e = ipm_ground(uv, yaw=math.radians(90), pitch=math.radians(45))
    assert bool(ok_e[0]) and abs(float(pts_e[0, 0]) - 1.62) < 1e-4  # 东 1.62
    assert abs(float(pts_e[0, 1])) < 1e-4


def test_ipm_sky_invalid():
    """平视/仰视(视线不朝下)必须 invalid——空中目标不落地平面(设计已知缺陷,显式拒绝)。"""
    uv = torch.tensor([[0.5, 0.5], [0.5, 0.1]])              # 中心、偏上
    _, ok = ipm_ground(uv, yaw=0.0, pitch=0.0)
    assert not bool(ok[0])
    _, ok2 = ipm_ground(uv, yaw=0.0, pitch=math.radians(-30))
    assert not ok2.any()


def test_write_read_roundtrip_and_grad():
    """特征写图后 MapReader 能读回非零 token;梯度流到 MapWriter.w_c(唯一可学件)。"""
    torch.manual_seed(0)
    m = EgoMapClip(c=8, size=32, half=32.0, levels=3)
    wr = MapWriter(feat_dim=16, c=8)
    rd = MapReader(c=8, d_out=32, grid=4)
    uv = torch.rand(40, 2) * torch.tensor([1.0, 0.4]) + torch.tensor([0.0, 0.55])  # 下半屏
    feats = torch.randn(40, 16, requires_grad=True)
    n = wr(m, uv, feats, yaw=0.3, pitch=math.radians(40))
    assert n > 0
    toks = rd(m)                                             # [4²·3, 32]
    assert toks.shape == (48, 32) and torch.isfinite(toks).all()
    toks.sum().backward()
    assert wr.w_c.weight.grad is not None and wr.w_c.weight.grad.abs().sum() > 0
    assert feats.grad is not None                            # 可微直通视觉特征


def test_north_anchored_motion_consistency():
    """写入一点后自身移动:step 后按新相对坐标读回同一特征(北锚定账本)。"""
    m = EgoMapClip(c=4, size=64, half=32.0, levels=1)        # 单级免 clip 级间边界
    pt = torch.tensor([[3.0, 5.0]])
    f = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    m.write(pt, f)
    before = m.read(pt)
    m.step((2.0, 1.0))                                       # 向东 2 北 1
    after = m.read(torch.tensor([[1.0, 4.0]]))               # 相对坐标应减去位移
    assert torch.allclose(before, after, atol=1e-4)


def test_aim_pin_lifecycle():
    """钉点:set→step 平移账本→get;视线指天 set 失败;超 TTL 过期。"""
    pin = AimPin(ttl_ticks=3)
    assert pin.set((0.5, 0.5), yaw=0.0, pitch=math.radians(45))
    xy, age = pin.get()
    assert age == 0 and abs(float(xy[1]) - 1.62) < 1e-3
    pin.step((0.0, 1.0))                                     # 向北走 1 格
    xy2, _ = pin.get()
    assert abs(float(xy2[1]) - 0.62) < 1e-3                  # 目标相对变近
    assert not pin.set((0.5, 0.1), yaw=0.0, pitch=0.0)       # 指天:拒钉
    for _ in range(4):
        pin.step((0.0, 0.0))
    assert pin.get() == (None, None)                         # TTL 过期


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
