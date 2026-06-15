"""集成冒烟:活的 MinecraftWorldModel(Δz-JEPA)前向 + 反向 + EMA,全程离线 CPU。

骨干用本文件内的 MockDINOv2(随机冻结卷积)经**依赖注入**喂给模型——按 AGENTS §2,
mock 只许在 tests/。绕开网络/GPU,只验证管线接线(import、token 拼接位置、shape、
无 NaN 梯度),不验证学习效果(那是训练指标的事)。替代了旧的 TaoNot42 + rhythm 冒烟。
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from net.world_model import MinecraftWorldModel
from net.config import ModelConfig, XiConfig, HeadsConfig

ACT_DIM = 22


class MockDINOv2(nn.Module):
    """随机冻结卷积,模拟 DINOv2 输出 patch token——仅供无网络管线冒烟,经依赖注入传入模型。

    输入 [B,3,H,W] → 输出 [B,M,d];.embed_dim 让模型自动取 enc_dim。
    """
    def __init__(self, d=64):
        super().__init__()
        self.embed_dim = d
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8), nn.ReLU(),
            nn.Conv2d(64, d, kernel_size=4, stride=4), nn.ReLU(),
        )

    def forward(self, x):
        feat = self.net(x)                                  # [B, d, h, w]
        B, d, h, w = feat.shape
        return feat.view(B, d, h * w).transpose(1, 2)       # [B, M, d]


def _tiny_model():
    cfg = ModelConfig(d=64, N=4, K=2, J=2, act_dim=ACT_DIM, ema_decay=0.99, max_skip=3,
                      xi=XiConfig(d_xi=8), heads=HeadsConfig(inv_dyn_ctx=True))
    return MinecraftWorldModel(cfg, backbone=MockDINOv2(64))


def test_world_model_forward_backward():
    B, device = 2, "cpu"
    model = _tiny_model().to(device).train()
    S = model.S

    img = torch.rand(B, 3, 64, 64, device=device)
    z_ref = model.encode_obs(img)                       # [B,N,d] 在线感知
    assert z_ref.shape == (B, model.N, model.d)

    h = torch.randn(B, 1, model.d, device=device)
    a_hist = torch.zeros(B, model.J, ACT_DIM, device=device)
    a_cur = torch.zeros(B, S, ACT_DIM, device=device)
    dt = torch.full((B,), float(S), device=device)
    t_vec = torch.zeros(B, device=device)

    out = model(z_ref, h, a_hist, a_cur, dt, t_vec)
    assert out["mu"].shape == (B, model.N, model.d)
    assert out["c"].shape == (B, model.N, 1)
    assert out["h_next"].shape == (B, 1, model.d)
    assert set(out["action_plan"]) >= {"mouse_logits", "keyboard", "onset", "duration", "exist"}

    # 逆动力学头:槽-Δz·c 残差 + patch 平均 Δz + ctx(h),验证三路接线
    with torch.no_grad():
        z_tg = model.encode_target(img)
    residual = (z_tg - z_ref) * out["c"]
    patch_dz = model.extract_feats(img).mean(dim=1)     # [B, enc_dim]
    mouse_logits, kb_prob, parts = model.inv_dyn(residual, patch_dz=patch_dz, ctx=h.squeeze(1))
    assert mouse_logits.shape == (B, 2, model.heads.n_cam_bins)
    assert kb_prob.shape == (B, ACT_DIM - 2)
    assert parts is not None                            # enc_dim 给定 ⇒ patch 旁路启用

    loss = out["mu"].pow(2).mean() + out["action_plan"]["onset"].mean() + mouse_logits.mean()
    loss.backward()

    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in model.parameters())
    assert not has_nan, "梯度出现 NaN"

    # EMA 目标编码器跟踪一步,不应抛错
    model.ema_update()


if __name__ == "__main__":
    test_world_model_forward_backward()
    print("ok")
