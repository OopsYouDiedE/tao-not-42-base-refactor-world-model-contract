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
from utils.config_io import load_yaml


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
    """默认 ModelConfig 必须 == 重构前写死值(空/缺省 yaml 逐位复现今日模型)。"""
    c = ModelConfig()
    assert (c.d, c.N, c.K, c.J) == (384, 16, 5, 8)
    assert c.act_dim == 22 and c.ema_decay == 0.99 and c.max_skip == 8
    assert c.state_dec_mult == 2
    assert (c.dynamics.kind, c.dynamics.num_layers, c.dynamics.nhead,
            c.dynamics.ffn_mult, c.dynamics.dropout) == ("transformer", 4, 8, 4, 0.0)
    assert c.encoder.binder == "competitive" and c.encoder.binder_heads == 4
    assert c.heads.n_cam_bins == 11 and c.heads.inv_dyn_ctx is False
    assert c.xi.d_xi == 32 and c.xi.phi is None
    assert c.backbone.kind == "dinov3" and c.backbone.weights is None


def test_from_dict_merges_partial_and_rejects_unknown():
    """部分 dict 与默认合并;未知键(顶层 / 子配置)报错,防 yaml 漏配静默。"""
    c = ModelConfig.from_dict({"d": 128, "dynamics": {"num_layers": 2}})
    assert c.d == 128 and c.dynamics.num_layers == 2
    assert c.dynamics.nhead == 8 and c.N == 16          # 缺键取默认
    assert _raises_value_error(lambda: ModelConfig.from_dict({"nope": 1}))
    assert _raises_value_error(lambda: ModelConfig.from_dict({"dynamics": {"nope": 1}}))


def test_yaml_to_model_smoke():
    """tiny.yaml → ModelConfig → 建模(DI mock 骨干),跑通 encode_obs → forward 的 shape。"""
    raw = load_yaml(os.path.join(_ROOT, "configs/minecraft/tiny.yaml"))
    cfg = ModelConfig.from_dict(raw["model"])
    model = MinecraftWorldModel(cfg, backbone=_MockBackbone(cfg.d)).eval()

    B = 2
    img = torch.rand(B, 3, 64, 64)
    z = model.encode_obs(img)
    assert z.shape == (B, cfg.N, cfg.d)

    h = torch.zeros(B, 1, cfg.d)
    a_hist = torch.zeros(B, cfg.J, cfg.act_dim)
    a_cur = torch.zeros(B, model.S, cfg.act_dim)
    dt = torch.full((B,), float(model.S))
    t_vec = torch.zeros(B)
    out = model(z, h, a_hist, a_cur, dt, t_vec)
    assert out["mu"].shape == (B, cfg.N, cfg.d)
    assert out["c"].shape == (B, cfg.N, 1)


if __name__ == "__main__":
    test_defaults_match_legacy()
    test_from_dict_merges_partial_and_rejects_unknown()
    test_yaml_to_model_smoke()
    print("ok")
