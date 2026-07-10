# -*- coding: utf-8 -*-
"""BC 暖启动的契约单测:mu-law 编解码互逆 / torch-numpy 同式 / 键置换 / 窗口堆叠对齐。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.craftground.action_contract import (CAM_BINS, CAM_MAX_DEG,  # noqa: E402
                                               V2_KEYS, bins_to_deg, deg_to_bins,
                                               stack_frames)
from train.craftground.bc_vpt_warmstart import (VPT_TO_V2, bin_center_t,  # noqa: E402
                                                deg_to_bins_t, encode_targets)
from train.minecraft.vpt_dataset import VPT_KEYS  # noqa: E402


def test_mulaw_roundtrip():
    """bin 中心处 encode∘decode 恒等(全 11 bin)。"""
    b = np.arange(CAM_BINS)
    assert (deg_to_bins(bins_to_deg(b)) == b).all()


def test_torch_numpy_same():
    """torch 侧编解码与 numpy 契约同式(随机值 + 边界)。"""
    v = np.concatenate([np.random.default_rng(0).uniform(-1.2, 1.2, 512),
                        [-1.0, 0.0, 1.0]])
    np_bins = deg_to_bins(v * CAM_MAX_DEG)
    t_bins = deg_to_bins_t(torch.from_numpy(v)).numpy()
    assert (np_bins == t_bins).all()
    centers = bin_center_t(torch.arange(CAM_BINS)).numpy() * CAM_MAX_DEG
    assert np.allclose(centers, bins_to_deg(np.arange(CAM_BINS)), atol=1e-5)


def test_key_permutation():
    """VPT 键序 → V2 键序:w→forward, a→left, s→back, d→right,hotbar 同号。"""
    act = torch.zeros(22)
    act[2 + VPT_KEYS.index("key_w")] = 1
    act[2 + VPT_KEYS.index("key_a")] = 1
    act[2 + VPT_KEYS.index("key_hotbar.3")] = 1
    _, keys, _ = encode_targets(act)
    on = {V2_KEYS[i] for i in keys.nonzero().flatten().tolist()}
    assert on == {"forward", "left", "hotbar.3"}


def test_prev_is_quantized_center():
    """prev 相机分量 = 过量化-反量化的 bin 中心(与采样端 prev 分布一致)。"""
    act = torch.zeros(4, 22)
    act[:, 0] = torch.tensor([0.03, -0.4, 1.0, 0.0])
    bins, _, prev = encode_targets(act)
    assert torch.allclose(prev[:, :2], bin_center_t(bins), atol=1e-6)


def test_window_stack_matches_contract():
    """窗口内取历史帧的口径与 stack_frames(episode 级)在 t≥s-1 处逐元素一致。"""
    s, t_n = 4, 9
    imgs = np.random.default_rng(1).integers(0, 255, (t_n, 6, 8, 3), np.uint8)
    full = stack_frames(imgs, s)                                  # [T,3s,H,W]
    img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)            # [T,3,H,W]
    ts = torch.arange(s - 1, t_n - 1)
    idx = ts[:, None] + torch.arange(-(s - 1), 1)[None, :]
    win = img_t[idx].reshape(len(ts), s * 3, 6, 8).numpy()        # 训练器同式
    assert (win == full[s - 1:t_n - 1]).all()
