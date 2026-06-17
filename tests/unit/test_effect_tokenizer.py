"""net.effect_tokenizer 单元测试:𝒟(对 Δz_inv 量化)+ 𝔤(生成元算子)。

离线 CPU,无网络/GPU。验证 token 来自**潜变化**而非动作、null 码语义、commitment 可反向、
生成元增量形状与可微。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from net.effect_tokenizer import EffectTokenizer, GeneratorBank


def test_effect_tokenizer_quantizes_latent_delta():
    """对 Δz_inv 量化得事件码 + commitment;输入是潜、不是动作。"""
    B, M, d_inv = 4, 9, 16
    tok = EffectTokenizer(d_inv=d_inv, event_vocab_size=8).train()
    z_t = torch.randn(B, M, d_inv)
    z_next = (z_t + 0.5 * torch.randn(B, M, d_inv)).requires_grad_(True)

    event_idx, loss, delta = tok(z_t, z_next)
    assert event_idx.shape == (B,) and event_idx.dtype == torch.int64
    assert loss.shape == () and delta.shape == (B, d_inv)
    assert tok.codebook.shape == (8, d_inv)
    assert 0 <= tok.null_code < 8

    loss.backward()                       # commitment 可反向(z 路径)
    assert z_next.grad is not None and not torch.isnan(z_next.grad).any()


def test_null_delta_routes_consistently():
    """Δz_inv≈0 的转移应稳定路由到同一个码(no-op 语义)。"""
    B, M, d_inv = 6, 4, 16
    tok = EffectTokenizer(d_inv=d_inv, event_vocab_size=8).eval()
    z = torch.randn(B, M, d_inv)
    idx, _, _ = tok(z, z + 1e-6)          # 几乎零变化
    assert (idx == idx[0]).all()          # 全部落到同一码


def test_generator_bank_shapes_and_grad():
    """𝔤 生成元:由锚点 z_rev 与系数 c 给出可逆增量,形状对、可微。"""
    B, M, d_rev, n_gen = 3, 7, 12, 5
    gen = GeneratorBank(d_rev=d_rev, n_generators=n_gen)
    z_rev = torch.randn(B, M, d_rev, requires_grad=True)
    c = torch.tanh(torch.randn(B, M, n_gen))
    delta = gen(z_rev, c)
    assert delta.shape == (B, M, d_rev)
    delta.pow(2).mean().backward()
    assert z_rev.grad is not None and not torch.isnan(z_rev.grad).any()


if __name__ == "__main__":
    test_effect_tokenizer_quantizes_latent_delta()
    test_null_delta_routes_consistently()
    test_generator_bank_shapes_and_grad()
    print("ok")
