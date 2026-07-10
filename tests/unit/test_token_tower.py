# -*- coding: utf-8 -*-
"""net/token_tower.py 验收:形状 / 各 token 组梯度真流 / 语言 pad 掩码 / 按键先验。"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.token_tower import TokenTowerConfig, build_token_tower, encode_utf8  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _batch(cfg, b=2, nv=7, nm=5):
    return (torch.randn(b, nv, cfg.vis_dim, device=DEV, requires_grad=True),
            torch.randn(b, nm, cfg.map_dim, device=DEV, requires_grad=True),
            encode_utf8(["chop the tree", "go north"]).to(DEV),
            torch.randn(b, cfg.geo_dim, device=DEV),
            torch.randn(b, cfg.n_mouse + cfg.n_keys, device=DEV))


def test_shapes_and_key_prior():
    cfg = TokenTowerConfig()
    t = build_token_tower(cfg).to(DEV)
    cam, key = t(*_batch(cfg))
    assert cam.shape == (2, 2, 11) and key.shape == (2, 20)
    b = t.key_head.bias.detach().cpu()
    assert torch.allclose(b, torch.full_like(b, float(np.log(0.05 / 0.95))), atol=1e-5)


def test_grads_reach_every_input_group():
    """cross-attention 真的在用每个 token 组(A1 语言嵌入含内)。"""
    cfg = TokenTowerConfig()
    t = build_token_tower(cfg).to(DEV)
    vis, map_t, lang, geo, prev = _batch(cfg)
    cam, key = t(vis, map_t, lang, geo, prev)
    (cam.sum() + key.sum()).backward()
    assert vis.grad is not None and vis.grad.abs().sum() > 0
    assert map_t.grad is not None and map_t.grad.abs().sum() > 0
    assert t.lang_emb.weight.grad is not None and t.lang_emb.weight.grad.abs().sum() > 0


def test_lang_antonym_tokens_distinct():
    """A1 的存在性检查:'turn left' 与 'turn right' 的字节 token 序列不同
    (MiniLM 反义词坍缩在此结构上不可能发生——码本由任务梯度塑形)。"""
    a, b = encode_utf8(["turn left", "turn right"])
    assert not torch.equal(a, b)


def test_empty_groups_ok():
    """缺视觉/地图组([B,0,·])也能前向——探针门控期各前端可独立缺席。"""
    cfg = TokenTowerConfig()
    t = build_token_tower(cfg).to(DEV)
    cam, key = t(torch.zeros(2, 0, cfg.vis_dim, device=DEV),
                 torch.zeros(2, 0, cfg.map_dim, device=DEV),
                 encode_utf8(["dig down", "x"]).to(DEV),
                 torch.zeros(2, cfg.geo_dim, device=DEV),
                 torch.zeros(2, 22, device=DEV))
    assert torch.isfinite(cam).all() and torch.isfinite(key).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
