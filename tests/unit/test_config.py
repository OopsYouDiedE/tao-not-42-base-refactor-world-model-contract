"""net.config 单元测试:默认值守恒 + yaml→config→建模 shape。

无 pytest 依赖(与本仓其余测试一致,纯 assert + __main__);骨干用依赖注入的随机卷积
mock(AGENTS §2,只在 tests/),离线 CPU。默认值守恒确保「空 yaml 逐位复现今日模型」。
"""
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, _ROOT)

from net.config import ModelConfig
from net.world_model import MinecraftWorldModel
from utils.io import load_yaml


class _MockBackbone(nn.Module):
    """随机冻结卷积,模拟 patch token 骨干;.embed_dim 供模型取 enc_dim。"""
    def __init__(self, d):
        super().__init__()
        self.embed_dim = d
        self.conv = nn.Conv2d(3, d, kernel_size=8, stride=8)

    def forward(self, x):
        f = self.conv(x)                                # [B, d, h, w]
        B, d, h, w = f.shape
        return f.view(B, d, h * w).transpose(1, 2)      # [B, M, d]


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_defaults_match_legacy():
    """默认 ModelConfig 必须符合新预设值。"""
    c = ModelConfig()
    assert (c.d, c.K, c.J) == (384, 5, 8)
    assert (c.d_rev, c.d_inv) == (256, 128) and c.d_rev + c.d_inv == c.d
    assert c.act_dim == 22 and c.max_skip == 8
    assert c.state_dec_mult == 2
    assert (c.dynamics.kind, c.dynamics.num_layers, c.dynamics.nhead,
            c.dynamics.ffn_mult, c.dynamics.dropout) == ("transformer", 4, 8, 4, 0.0)
    assert c.adapter.z_inv_kind == "gaussian"
    assert c.effect.event_vocab_size == 64 and c.effect.n_generators == 8
    assert c.heads.n_cam_bins == 11
    assert c.backbone.kind == "dinov3" and c.backbone.weights is None


def test_factorization_dim_guard():
    """d_rev + d_inv != d 必须报错(__post_init__ 守卫)。"""
    assert _raises_value_error(lambda: ModelConfig(d=64, d_rev=40, d_inv=16))


def test_from_dict_merges_partial_and_rejects_unknown():
    """部分 dict 与默认合并;未知键报错。"""
    c = ModelConfig.from_dict({"d": 128, "d_rev": 96, "d_inv": 32, "dynamics": {"num_layers": 2}})
    assert c.d == 128 and c.dynamics.num_layers == 2
    assert c.dynamics.nhead == 8 and c.K == 5          # 缺键取默认
    assert _raises_value_error(lambda: ModelConfig.from_dict({"nope": 1}))
    assert _raises_value_error(lambda: ModelConfig.from_dict({"dynamics": {"nope": 1}}))


def test_yaml_to_model_smoke():
    """tiny.yaml → ModelConfig → 建模(DI mock 骨干),跑通 encode → 序列对齐 forward 的 shape。"""
    raw = load_yaml(os.path.join(_ROOT, "configs/minecraft/tiny.yaml"))
    cfg = ModelConfig.from_dict(raw["model"])
    model = MinecraftWorldModel(cfg, backbone=_MockBackbone(cfg.d)).eval()

    B, T = 2, 4
    img = torch.rand(B, T, 3, 64, 64)
    feats = model.extract_feats(img.reshape(B * T, 3, 64, 64))
    z, kl = model.encode(feats)
    M = z.shape[-2]                          # 64x64 经 MockBackbone(stride 8)→ 8x8=64 patch
    assert z.shape == (B * T, M, cfg.d) and kl.shape == (B * T,)
    z = z.view(B, T, M, cfg.d)

    dt = torch.ones(B, T - 1)
    tf = torch.cat([torch.zeros(B, 1), dt.cumsum(1)], dim=1)
    act = torch.zeros(B, T - 1, cfg.act_dim)
    out = model(z[:, :2], tf[:, :2], act, tf[:, :T - 1], tf[:, T - 1])
    assert out["z_hat"].shape == (B, M, cfg.d)
    assert out["event_logits"].shape == (B, M, cfg.effect.event_vocab_size)
    assert out["e_norm_hat"].shape == (B, M)


if __name__ == "__main__":
    test_defaults_match_legacy()
    test_factorization_dim_guard()
    test_from_dict_merges_partial_and_rejects_unknown()
    test_yaml_to_model_smoke()
    print("ok")
