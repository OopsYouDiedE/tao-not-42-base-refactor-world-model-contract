"""集成冒烟:序列对齐 MinecraftWorldModel(时空 token 集合 → 未来帧潜对齐 + 因子化 + 反事实)。

离线 CPU,DI 注入 mock 骨干,绕开网络/GPU,只验证管线接线 + 反向无 NaN 梯度。
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from net.world_model import MinecraftWorldModel
from net.config import ModelConfig, DynamicsConfig, AdapterConfig, EffectConfig, PredictorConfig
from net.effect_tokenizer import EffectTokenizer
from train.minecraft.losses import (
    importance_from_effect, latent_align_loss, agreement_loss, event_ce, noop_loss)
from train.minecraft.train_minecraft import run_sequence
from train.minecraft.eval import evaluate
from blocks.regularization import SIGReg

ACT_DIM = 22


class MockDINOv2(nn.Module):
    """随机冻结卷积,模拟 DINOv2 输出 patch token。"""
    def __init__(self, d=64):
        super().__init__()
        self.embed_dim = d
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8), nn.ReLU(),
            nn.Conv2d(64, d, kernel_size=4, stride=4), nn.ReLU(),
        )

    def forward(self, x):
        feat = self.net(x)
        B, d, h, w = feat.shape
        return feat.view(B, d, h * w).transpose(1, 2)


def _tiny_cfg():
    return ModelConfig(d=64, d_rev=48, d_inv=16, K=2, J=2, act_dim=ACT_DIM, max_skip=3,
                       adapter=AdapterConfig(num_layers=1, nhead=4, ffn_mult=2),
                       dynamics=DynamicsConfig(num_layers=2, nhead=4, ffn_mult=2),
                       effect=EffectConfig(event_vocab_size=8, n_generators=4),
                       predictor=PredictorConfig(n_context_cutoffs=2))


def _tiny_model():
    cfg = _tiny_cfg()
    return MinecraftWorldModel(cfg, backbone=MockDINOv2(64)), cfg


def test_world_model_seq_align_forward_backward():
    B, T, device = 2, 5, "cpu"
    model, cfg = _tiny_model()
    model.to(device).train()
    etok = EffectTokenizer(d_inv=cfg.d_inv, event_vocab_size=cfg.effect.event_vocab_size)

    img = torch.rand(B, T, 3, 64, 64, device=device)
    feats = model.extract_feats(img.reshape(B * T, 3, 64, 64))
    z, kl = model.encode(feats)
    M = z.shape[-2]
    z = z.view(B, T, M, cfg.d)
    z_tgt = model.encode_target(feats).view(B, T, M, cfg.d)

    dt = torch.randint(1, 4, (B, T - 1), device=device).float()
    tf = torch.cat([torch.zeros(B, 1), dt.cumsum(1)], dim=1)
    target = T - 1
    act = torch.rand(B, target, ACT_DIM)
    out = model(z[:, :3], tf[:, :3], act, tf[:, :target], tf[:, target], null=False)
    out0 = model(z[:, :3], tf[:, :3], act, tf[:, :target], tf[:, target], null=True)
    assert out["z_hat"].shape == (B, M, cfg.d)
    assert out["event_logits"].shape == (B, M, cfg.effect.event_vocab_size)

    e = (out["z_hat_inv"] - out0["z_hat_inv"]).norm(dim=-1)
    w = importance_from_effect(e)
    al, _ = latent_align_loss(out["z_hat"], z_tgt[:, target], w)
    ev_idx, commit, _ = etok(z[:, 0, :, cfg.d_rev:], z[:, target, :, cfg.d_rev:])
    loss = (al + 0.1 * event_ce(out["event_logits"].mean(1), ev_idx)
            + noop_loss(out["e_norm_hat"], e) + commit.mean() + model.beta_kl * kl.mean())
    loss.backward()

    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in list(model.parameters()) + list(etok.parameters()))
    assert not has_nan, "梯度出现 NaN"
    model.update_ema()


def test_run_sequence_and_evaluate_smoke():
    """train.run_sequence 与 eval.evaluate 端到端冒烟(含闭环漂移 + 反捷径去相关指标)。"""
    B, T, device = 2, 5, "cpu"
    model, cfg = _tiny_model()
    etok = EffectTokenizer(d_inv=cfg.d_inv, event_vocab_size=cfg.effect.event_vocab_size)
    sigreg = SIGReg(knots=9, num_proj=64)

    batch = {
        "img": torch.rand(B, T, 3, 64, 64),
        "act_agg": torch.rand(B, T - 1, ACT_DIM),
        "dt": torch.randint(1, 4, (B, T - 1)).float(),
    }
    total, metrics = run_sequence(model.train(), etok, sigreg, batch, cfg,
                                  beta_sigreg=0.1, amp_dev="cpu", use_amp=False)
    total.backward()
    assert "align" in metrics and "agree" in metrics and "e_norm" in metrics

    ev = evaluate(model, etok, [batch], device, amp_dev="cpu", use_amp=False, cfg=cfg)
    for k in ("align", "agree", "rollout_drift", "corr_w_future", "corr_w_pixel"):
        assert k in ev


if __name__ == "__main__":
    test_world_model_seq_align_forward_backward()
    test_run_sequence_and_evaluate_smoke()
    print("ok")
