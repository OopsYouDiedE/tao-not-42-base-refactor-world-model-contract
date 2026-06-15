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


def test_control_remap_train_step():
    """in-context 实验的训练路径端到端冒烟:子集 ControlRemap + inv_dyn_ctx + 四项真实损失。

    验证「可直接训练」——前向(后验 ξ)→ dz_pred + inv_dyn(ctx=h)+ plan_bc + KL → 反向 →
    无 NaN 梯度。离线 mock 骨干、不触网络/GPU/数据。覆盖上轮新增的 remap 子集机制与
    base.yaml 默认开启的 inv_dyn_ctx(=true)在真实损失链路里的接线。"""
    from domains.minecraft.control_remap import ControlRemap, N_MOUSE
    from train.minecraft.losses import (dz_pred_loss, minecraft_inv_dyn_loss,
                                         plan_bc_loss, kl_diag_gauss)
    B, device = 2, "cpu"
    model = _tiny_model().to(device).train()                 # _tiny_model 已 inv_dyn_ctx=True
    S, T1, NK = model.S, 4, ACT_DIM - N_MOUSE

    img_t, img_t1 = torch.rand(B, 3, 64, 64), torch.rand(B, 3, 64, 64)

    # 动作:子集置换(高信号 w/a/s/d/space/attack),video(图像)不动。鼠标连续 + 键盘 0/1。
    def _raw(*lead):
        return torch.cat([torch.randn(*lead, N_MOUSE) * 2,
                          (torch.rand(*lead, NK) > 0.5).float()], dim=-1)
    remap = ControlRemap(remap_keys=True, n_holdout=8, seed=0, key_subset=[0, 1, 2, 3, 4, 7])
    spec = remap.sample(B)
    act_cur = remap.apply(_raw(B, S), spec)                  # a_cur [B,S,22]
    act_agg = remap.apply(_raw(B, T1), spec)                 # 聚合动作序列 [B,T1,22]

    z_obs = model.encode_obs(img_t)                          # [B,N,d] 带梯度
    with torch.no_grad():
        dz = model.encode_target(img_t1) - model.encode_target(img_t)
        z_tg_t1 = model.encode_target(img_t1)

    h = torch.randn(B, 1, model.d)
    a_hist = torch.zeros(B, model.J, ACT_DIM)
    dt_b = torch.full((B,), float(S))
    mu_q, lv_q = model.xi_posterior(z_obs, h, dt_b, dz)
    mu_p, lv_p = model.xi_prior(z_obs, h, dt_b)
    out = model(z_obs, h, a_hist, act_cur, dt_b, torch.zeros(B),
                task_emb=None, xi=model.xi_sample(mu_q, lv_q))

    pdz = (model.extract_feats(img_t1).mean(1) - model.extract_feats(img_t).mean(1)).float()
    ctx_h = h.squeeze(1).float()                             # pre-step h → FiLM 重绑定通路
    l_pred = dz_pred_loss(out["mu"].float(), dz)[0]
    l_inv = minecraft_inv_dyn_loss((z_tg_t1 - z_obs.float()), out["c"].float(),
                                   act_agg[:, 0], model.inv_dyn, move_w=4.0,
                                   patch_dz=pdz, ctx=ctx_h)[0]
    l_plan = plan_bc_loss(out["action_plan"], act_agg,
                          torch.full((B, T1), float(S)), 0, model.K)[0]
    l_kl = kl_diag_gauss(mu_q.float(), lv_q.float(), mu_p.float(), lv_p.float()).mean()
    loss = l_pred + l_inv + l_plan + l_kl
    loss.backward()

    assert torch.isfinite(loss), "训练步损失非有限"
    assert model.inv_dyn.use_ctx, "本测试须走 FiLM-on-h 重绑定通路(inv_dyn_ctx=True)"
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in model.parameters())
    assert not has_nan, "梯度出现 NaN"


if __name__ == "__main__":
    test_world_model_forward_backward()
    test_control_remap_train_step()
    print("ok")
