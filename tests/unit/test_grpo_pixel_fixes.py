# -*- coding: utf-8 -*-
"""2026-07-10 五个 log π 修复的数学验收(CUDA 冒烟;无 GPU 时退 CPU)。

验的是"采样 π = 更新 π"的各个成分与组内单步更新,不验环境/判官(那些要活环境)。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.craftground.grpo_pixel import stack_frames, update  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tower(**kw):
    cfg = PixelTowerConfig(img_hw=(90, 160), goal_dim=386, **kw)
    return build_pixel_tower(cfg).to(DEV)


def test_key_prior_bias():
    """按键先验:bias=logit(0.05) ⇒ 初始期望按键数 ≈ 1,不是 10。"""
    tower = _tower()
    b = tower.key_head.bias.detach()
    assert torch.allclose(b, torch.full_like(b, float(np.log(0.05 / 0.95))), atol=1e-5)
    p = torch.sigmoid(b).mean().item()
    assert abs(p * tower.cfg.n_keys - 1.0) < 0.05


def test_stack_frames_matches_sampling_deque():
    """更新端 stack_frames 与采样端 deque 堆叠逐字节同序(修复①的前提)。"""
    rng = np.random.default_rng(0)
    s, t_n = 4, 9
    imgs = rng.random((t_n, 6, 8, 3), np.float32)
    out = stack_frames(imgs, s)                     # [T,3s,6,8]
    fstack: list[np.ndarray] = []
    for t in range(t_n):                            # 采样端逐 tick 逻辑的逐字节复刻
        fstack.append(imgs[t])
        if len(fstack) < s:
            fstack = [imgs[t]] * (s - len(fstack)) + fstack
        fstack = fstack[-s:]
        ref = np.concatenate(fstack, axis=2).transpose(2, 0, 1)
        np.testing.assert_array_equal(out[t], ref)


def test_forward_shapes_and_eval_train_identical():
    """dropout=0 下 eval/train 前向同分布(修复②的结构保证)。

    容差说明:nn.MultiheadAttention 在 eval/no-grad 走 fast path、train 走常规路径,
    dropout=0 时两者数学同分布,只余 ~4e-7 的内核浮点差(实测)——对 log π 无意义。
    """
    tower = _tower()
    s = tower.cfg.frame_stack
    img = torch.rand(2, 1, 3 * s, 90, 160, device=DEV)
    goal = torch.rand(2, 386, device=DEV)
    prev = torch.rand(2, 1, 22, device=DEV)
    tower.eval()
    with torch.no_grad():
        c1, k1 = tower(img, goal, prev)
    tower.train()
    with torch.no_grad():
        c2, k2 = tower(img, goal, prev)
    assert c1.shape == (2, 1, 1, 2, 11) and k1.shape == (2, 1, 1, 20)
    assert torch.allclose(c1, c2, atol=1e-5) and torch.allclose(k1, k2, atol=1e-5)


def _fake_roll(rng, t_n, cfg):
    return dict(
        imgs=rng.random((t_n, *cfg.img_hw, 3), np.float32),
        prevs=rng.random((t_n, cfg.n_mouse + cfg.n_keys), np.float32),
        goals=rng.random((t_n, cfg.goal_dim), np.float32),
        cam=rng.integers(0, cfg.camera_bins, (t_n, cfg.n_mouse)),
        keys=rng.integers(0, 2, (t_n, cfg.n_keys)).astype(np.int8),
    )


class _CountingOpt:
    def __init__(self, opt):
        self.opt, self.steps = opt, 0

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.steps += 1
        self.opt.step()


def test_update_single_step_per_group_and_grads_flow():
    """修复⑤:整组唯一一次 opt.step;修复④:逐 tick goal 真进梯度;尾段 tick 不丢。"""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tower = _tower()
    cfg = tower.cfg
    opt = _CountingOpt(torch.optim.AdamW(tower.parameters(), lr=1e-4))
    rolls = [_fake_roll(rng, 50, cfg) for _ in range(3)]     # 50 不整除 chunk=16 ⇒ 有尾段
    adv = np.array([1.0, -1.0, 0.0])                          # 第三条 |adv|<1e-6 应跳过
    before = tower.goal_q.weight.detach().clone()
    loss = update(tower, opt, rolls, adv, chunk=16, temp=1.3, device=DEV)
    assert opt.steps == 1                                     # 组内梯度累积,单步
    assert np.isfinite(loss)
    assert not torch.equal(before, tower.goal_q.weight.detach())  # goal 路径真进梯度


def test_update_temperature_changes_objective():
    """修复③:损失确实打在 logits/temp 上——不同 temp 下同一批数据 loss 不同。"""
    rng = np.random.default_rng(1)
    losses = []
    for temp in (1.0, 4.0):
        torch.manual_seed(0)
        tower = _tower()
        opt = _CountingOpt(torch.optim.AdamW(tower.parameters(), lr=0.0))
        rolls = [_fake_roll(np.random.default_rng(2), 20, tower.cfg)]
        losses.append(update(tower, opt, rolls, np.array([1.0]), 8, temp, DEV))
    assert abs(losses[0] - losses[1]) > 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
