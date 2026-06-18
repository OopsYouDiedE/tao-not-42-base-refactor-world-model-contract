"""RSSM + 后继特征切片的 CPU 冒烟:核前向/反向有限 + 两条验收线可算。

离线 CPU、DI 注入 mock 骨干,绕开网络/GPU,只验证接线与数值有限性(不验证收敛)。
对应设计:knowledge/rssm_sf_design.md。
"""
import os
import sys
import copy

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from net.config import RSSMConfig, BackboneConfig
from net.rssm import RSSM
from train.minecraft.train_rssm import (
    FrozenBackbonePerception, rssm_loss, hard_horizon_align_ratio,
    dose_response_corr, lambda_return_sf, empirical_discounted_future, _prep_batch)

ACT = 22


class MockDINOv2(nn.Module):
    """随机冻结卷积,模拟 DINOv2 patch token 输出 [B,M,E]。"""

    def __init__(self, d=16):
        super().__init__()
        self.embed_dim = d
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8), nn.ReLU(),
            nn.Conv2d(32, d, kernel_size=4, stride=4), nn.ReLU())

    def forward(self, x):
        f = self.net(x)
        B, d, h, w = f.shape
        return f.view(B, d, h * w).transpose(1, 2)


def _tiny_cfg(E=16):
    return RSSMConfig(embed_dim=E, act_dim=ACT, deter=32, hidden=32, d_rev=4,
                      inv_groups=2, inv_classes=3, sf_hidden=32, sf_dim=1, free_nats=0.5)


def test_rssm_core_forward_backward():
    B, T, E = 2, 6, 16
    cfg = _tiny_cfg(E)
    rssm = RSSM(cfg)
    sf_target = copy.deepcopy(rssm.sf_head)

    e = torch.randn(B, T, E)
    actions = torch.randn(B, T - 1, ACT)
    phi = torch.randint(0, 2, (B, T, 1)).float()
    phi_mask = torch.ones(B, T, 1)

    total, metrics, feats = rssm_loss(rssm, sf_target, e, actions, phi, phi_mask,
                                      gamma=0.97, lam=0.95, beta_ground=1.0)
    assert feats.shape == (B, T, cfg.feat_dim), feats.shape
    assert torch.isfinite(total), total
    for k in ("loss", "kl", "ground", "sf"):
        assert k in metrics and metrics[k] == metrics[k], (k, metrics)
    total.backward()
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in rssm.parameters())
    assert not has_nan, "梯度出现 NaN"


def test_observe_imagine_shapes():
    B, T, E = 3, 7, 16
    cfg = _tiny_cfg(E)
    rssm = RSSM(cfg)
    e = torch.randn(B, T, E)
    actions = torch.randn(B, T - 1, ACT)

    feats, post, prior, states = rssm.observe(e, actions)
    assert feats.shape == (B, T, cfg.feat_dim)
    assert states["h"].shape == (B, T, cfg.deter)
    assert post["inv_logits"].shape == (B, T, cfg.inv_groups, cfg.inv_classes)

    init = {"h": states["h"][:, 0], "z": states["z"][:, 0]}
    img = rssm.imagine(init, actions)
    assert img.shape == (B, T - 1, cfg.feat_dim), img.shape

    kl, kl_value = rssm.kl_loss(post, prior)
    assert torch.isfinite(kl) and torch.isfinite(kl_value)


def test_acceptance_lines_computable():
    B, T, E = 4, 8, 16
    cfg = _tiny_cfg(E)
    rssm = RSSM(cfg)
    e = torch.randn(B, T, E)
    actions = torch.randn(B, T - 1, ACT)
    phi = torch.randint(0, 2, (B, T, 1)).float()
    phi_mask = torch.ones(B, T, 1)

    ar, k = hard_horizon_align_ratio(rssm, e, actions)
    assert k == T // 2 and ar > 0 and ar == ar, (ar, k)
    dc = dose_response_corr(rssm, e, actions, phi, phi_mask, gamma=0.97)
    assert -1.0 <= dc <= 1.0, dc


def test_lambda_return_and_discount_shapes():
    B, T, F = 2, 5, 1
    phi = torch.rand(B, T, F)
    psi = torch.rand(B, T, F)
    tgt = lambda_return_sf(phi, psi, gamma=0.9, lam=0.9)
    assert tgt.shape == (B, T - 1, F), tgt.shape
    D = empirical_discounted_future(phi, gamma=0.9)
    assert D.shape == (B, T, F)
    # 末帧 D == phi(无未来项);折扣单调:D_t >= phi_t(非负 φ)
    assert torch.allclose(D[:, -1], phi[:, -1])
    assert (D >= phi - 1e-5).all()


def test_perception_and_prep_batch():
    B, T, E = 2, 6, 16
    perc = FrozenBackbonePerception(BackboneConfig(), injected=MockDINOv2(E))
    img = torch.rand(B, T, 3, 64, 64)
    e = perc.encode_seq(img)
    assert e.shape == (B, T, E) and not e.requires_grad

    batch = {
        "img": (torch.rand(B, T, 3, 64, 64) * 255).to(torch.uint8),
        "act_agg": torch.randn(B, T - 1, ACT),
        "has_item": torch.randint(0, 2, (B, T)).float(),
    }
    e2, actions, phi, phi_mask = _prep_batch(batch, perc, "cpu")
    assert e2.shape == (B, T, E)
    assert actions.shape == (B, T - 1, ACT)
    assert phi.shape == (B, T, 1) and phi_mask.shape == (B, T, 1)


if __name__ == "__main__":
    test_rssm_core_forward_backward()
    test_observe_imagine_shapes()
    test_acceptance_lines_computable()
    test_lambda_return_and_discount_shapes()
    test_perception_and_prep_batch()
    print("ok")
