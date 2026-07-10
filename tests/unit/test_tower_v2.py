# -*- coding: utf-8 -*-
"""v2 塔接线验收:采样=回放同分布 / 梯度真流 / 地图写读真跑 / 降级 / v1-v2 checkpoint 隔离。

mock 前端按 DinoFrontend 协议依赖注入(AGENTS §2:mock 只活在 tests/),CPU 可跑。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.calibration import SelfCalib  # noqa: E402
from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.craftground.action_contract import CAM_BINS, V2_KEYS  # noqa: E402
from train.craftground.tower_v2 import (V2Config, V2Policy, V2Runtime,  # noqa: E402
                                        v2_replay)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MockFrontend:
    """DinoFrontend 同协议 mock:3×4 网格均值池化 × 固定随机投影(确定性,无网络)。"""

    def __init__(self, device=DEV, enc_dim=32):
        g = torch.Generator().manual_seed(0)
        self.proj = torch.randn(3, enc_dim, generator=g)
        self.enc_dim, self.device = enc_dim, device
        gh, gw = 3, 4
        self.n_tokens = gh * gw
        u = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
        v = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        self.uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], -1).to(device)

    def encode(self, img):
        x = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32)
        h, w = x.shape[:2]
        cells = x[:h // 3 * 3, :w // 4 * 4].reshape(3, h // 3, 4, w // 4, 3).mean((1, 3))
        return (cells.reshape(12, 3) @ self.proj).to(self.device)


def _calib(measured=True):
    c = SelfCalib(img_w=160, img_h=90)
    if measured:
        c.px_per_deg_yaw = 2.2      # 正增益 ⇒ 正 cmd=右转/向下(几何普适,见 calibration)
        c.px_per_deg_pitch = 2.1
        c.step_blocks = 0.15
        c.latency_ticks = 1
        c.mode = "position"
    return c


def _runtime(measured=True, seed=0):
    torch.manual_seed(seed)
    vcfg = V2Config(n_frames=2, d=64, map_grid=2, map_levels=2)
    fe = MockFrontend()
    policy = V2Policy(vcfg, fe.enc_dim).to(DEV)
    rt = V2Runtime(policy, fe, DEV)
    rt.begin(_calib(measured), init_cmd_deg=(0.0, 20.0))
    return policy, rt


def _run_ticks(rt, n=6, seed=0):
    rng = np.random.default_rng(seed)
    prev = np.zeros(22, np.float32)
    outs = []
    for t in range(n):
        small = rng.random((90, 160, 3), np.float32)
        if t == 2:                                     # 慢塔刷新一次
            rt.on_slow(dict(subgoal="walk to tree", aim=[600.0, 700.0]))
        cam_l, key_l = rt.tick(small, prev)
        outs.append((cam_l.detach().cpu(), key_l.detach().cpu()))
        prev = np.concatenate([rng.uniform(-1, 1, 2), rng.integers(0, 2, 20)]) \
            .astype(np.float32)
    return outs, prev


def test_tick_shapes_map_written_and_pin():
    """v2 前向真跑:logits 形状、IPM 稠密写图有落点、aim 钉点存活。"""
    policy, rt = _runtime()
    outs, _ = _run_ticks(rt)
    cam_l, key_l = outs[-1]
    assert cam_l.shape == (2, CAM_BINS) and key_l.shape == (len(V2_KEYS),)
    written = sum(float(m.cnt.sum()) for m in rt.map.maps)
    assert written > 0                                  # 地图确实被写(pitch=+20° 朝地)
    xy, age = rt.pin.get()
    assert xy is not None and age >= 0                  # on_slow 的 aim 已钉进世界系
    ex = rt.export()
    assert ex["vis_toks"].shape == (6, 2 * 12, 32)      # S=2 帧堆叠
    assert ex["map_toks"].shape == (6, 2 * 2 * 2, 64)   # grid²·levels
    assert ex["lang_toks"].shape == (6, 48) and ex["geo"].shape == (6, 16)
    assert (ex["lang_toks"][2] != ex["lang_toks"][0]).any()   # 刷新后语言 token 换血


def test_sample_equals_replay():
    """采样 π = 更新 π:记录 token 回放的 logits 与采样端逐 tick 一致(修复①口径)。"""
    policy, rt = _runtime()
    policy.eval()
    rng = np.random.default_rng(1)
    prevs, outs = [np.zeros(22, np.float32)], []
    for t in range(5):
        small = rng.random((90, 160, 3), np.float32)
        cam_l, key_l = rt.tick(small, prevs[-1])
        outs.append((cam_l.detach(), key_l.detach()))
        prevs.append(rng.uniform(-1, 1, 22).astype(np.float32))
    r = dict(rt.export(), prevs=np.stack(prevs[:5]))
    policy.train()                                       # 更新端模式;无 dropout ⇒ 同分布
    cam_r, key_r = v2_replay(policy, r, slice(0, 5), DEV)
    for t in range(5):
        assert torch.allclose(outs[t][0], cam_r[t], atol=1e-4)
        assert torch.allclose(outs[t][1], key_r[t], atol=1e-4)


def test_replay_grads_flow_all_groups():
    """梯度真流:回放损失回传到 tower 全组参数;w_c/map_reader.proj 如实不更新(GRPO 路径)。"""
    policy, rt = _runtime()
    _run_ticks(rt, n=4)
    r = dict(rt.export(), prevs=np.random.default_rng(2)
             .uniform(-1, 1, (4, 22)).astype(np.float32))
    policy.train()
    cam_l, key_l = v2_replay(policy, r, slice(0, 4), DEV)
    (cam_l.sum() + key_l.sum()).backward()
    tw = policy.tower
    for name, p in [("vis_in", tw.vis_in.weight), ("map_in", tw.map_in.weight),
                    ("lang_emb", tw.lang_emb.weight), ("geo_in", tw.geo_in.weight),
                    ("xattn", tw.xattn.in_proj_weight), ("cam_head", tw.cam_head.weight),
                    ("key_head", tw.key_head.weight), ("frame_emb", tw.frame_emb)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert policy.map_writer.w_c.weight.grad is None     # 记录值当常量(文档口径)
    assert policy.map_reader.proj.weight.grad is None


def test_uncalibrated_degrades_explicitly():
    """符号/步速测不出 ⇒ 不写图、不钉点、平移账本置零、physics 有效位为 0——不编数。"""
    policy, rt = _runtime(measured=False)
    _run_ticks(rt, n=3)
    assert not rt.pose_ok
    assert sum(float(m.cnt.sum()) for m in rt.map.maps) == 0
    assert rt.pin.get() == (None, None)
    geo = rt.export()["geo"]
    assert np.all(geo[:, 2:15] == 0)                     # 钉点 3 维 + physics 10 维全零


def test_odometry_sign_and_frame():
    """里程计账本:cmd 正 yaw(实测正增益)= 顺时针;yaw=90° 时前进键位移向东。"""
    policy, rt = _runtime()
    keys = np.zeros(20, np.float32)
    keys[V2_KEYS.index("forward")] = 1
    east, north = rt._odometry(np.array([90.0, 0.0]), keys)
    assert rt.yaw_deg == pytest.approx(90.0)
    assert east == pytest.approx(0.15, abs=1e-6)         # 步速 0.15 格/tick,朝东
    assert north == pytest.approx(0.0, abs=1e-6)
    east2, north2 = rt._odometry(np.array([-90.0, 0.0]), keys)   # 转回北
    assert rt.yaw_deg == pytest.approx(0.0)
    assert north2 == pytest.approx(0.15, abs=1e-6)
    assert rt.pitch_deg == pytest.approx(20.0)           # 标定期 +20° 计入积分起点


def test_v1_v2_checkpoint_isolation():
    """v1 checkpoint 装不进 v2(严格报错,不静默);v1 结构与加载路径不因 v2 存在而变。"""
    cfg = PixelTowerConfig(img_hw=(90, 160), goal_dim=386, n_keys=len(V2_KEYS),
                           camera_bins=CAM_BINS)
    v1 = build_pixel_tower(cfg)
    v1.load_state_dict(v1.state_dict())                  # 自往返严格通过
    policy, _ = _runtime()
    with pytest.raises(RuntimeError):
        policy.load_state_dict(v1.state_dict())


BC_CKPT = Path(__file__).resolve().parents[2] / "runs/checkpoints/bc_vpt/best.pt"


@pytest.mark.skipif(not BC_CKPT.exists(), reason="无 bc_vpt checkpoint")
def test_bc_checkpoint_still_loads_into_v1():
    """--init-from 兼容:现行 bc_vpt best.pt 仍能严格加载进 grpo 的 v1 结构。"""
    cfg = PixelTowerConfig(img_hw=(90, 160), goal_dim=384 + 2, n_keys=len(V2_KEYS),
                           camera_bins=CAM_BINS)
    tower = build_pixel_tower(cfg)
    ck = torch.load(BC_CKPT, map_location="cpu", weights_only=True)
    tower.load_state_dict(ck["tower"])                   # strict=True 默认


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
