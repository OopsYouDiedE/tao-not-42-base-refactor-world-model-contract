"""SpatialPosEmbed 单元测试(CPU 可跑)。

验收契约:
  - 输出形状 [B, d];同坐标确定性一致。
  - 不同位置 → 不同编码(脑子能区分方位);近位置比远位置更相似(局部连续)。
  - log(s) 前 clamp(I1):s→0 不产生 NaN/Inf。
  - 梯度可回传到投影层。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from blocks.primitives import SpatialPosEmbed


def _emb(mod, x, y, s):
    return mod(torch.tensor(x), torch.tensor(y), torch.tensor(s))


def test_shape_and_deterministic():
    torch.manual_seed(0)
    mod = SpatialPosEmbed(d=64).eval()
    B = 5
    x, y, s = torch.rand(B), torch.rand(B), torch.rand(B).clamp(min=0.05)
    a = mod(x, y, s)
    b = mod(x, y, s)
    assert a.shape == (B, 64), f"形状应为 [B,d],实测 {tuple(a.shape)}"
    assert torch.allclose(a, b), "同输入应确定性一致"
    assert torch.isfinite(a).all()


def test_distinct_positions_distinct_codes():
    torch.manual_seed(0)
    mod = SpatialPosEmbed(d=128).eval()

    def cos(p, q):
        e1 = _emb(mod, *p)[0]
        e2 = _emb(mod, *q)[0]
        return torch.cosine_similarity(e1, e2, dim=0).item()

    base = ([0.0], [0.0], [0.5])
    near = ([0.05], [0.05], [0.5])
    far = ([0.9], [-0.9], [0.5])

    sim_near = cos(base, near)
    sim_far = cos(base, far)
    print(f"sim_near={sim_near:.4f}  sim_far={sim_far:.4f}")
    assert sim_near > sim_far, "近位置应比远位置更相似(局部连续)"
    # 远位置应明显可区分(不等同)
    assert sim_far < 0.999, "不同位置应产生可区分编码"


def test_scale_clamp_no_nan():
    mod = SpatialPosEmbed(d=32).eval()
    # s 取极小值(含 0):log 前 clamp 应避免 NaN/Inf
    out = mod(torch.zeros(3), torch.zeros(3), torch.zeros(3))
    assert torch.isfinite(out).all(), "s→0 不应产生 NaN/Inf(I1 clamp)"


def test_grad_flows():
    mod = SpatialPosEmbed(d=32)
    x = torch.rand(4, requires_grad=True)
    y = torch.rand(4, requires_grad=True)
    s = torch.rand(4).clamp(min=0.1)
    loss = mod(x, y, s).pow(2).mean()
    loss.backward()
    assert mod.proj.weight.grad is not None and torch.isfinite(mod.proj.weight.grad).all()


if __name__ == "__main__":
    test_shape_and_deterministic()
    test_distinct_positions_distinct_codes()
    test_scale_clamp_no_nan()
    test_grad_flows()
    print("SpatialPosEmbed: all tests passed.")
