"""SIGReg 单元测试(CPU 可跑,无需 Mamba/GPU)。

验收契约:
  - 标准正态输入  → 统计量小(分布已是 N(0,1),无需正则)。
  - 坍缩/低秩输入 → 统计量大,且远大于正态情形(防坍缩信号有效)。
  - 输出为 0 维非负标量;梯度有限可回传。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from blocks.primitives import SIGReg


def _loss(x):
    # 无分组:外加一个长度 1 的 G 维 → [1, B, D]
    return SIGReg(num_proj=512)(x.unsqueeze(0))


def test_scalar_nonneg_and_grad():
    torch.manual_seed(0)
    x = torch.randn(2048, 32, requires_grad=True)
    loss = _loss(x)
    assert loss.dim() == 0, "应为 0 维标量"
    assert loss.item() >= 0.0, "统计量构造上非负"
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "梯度须有限可回传"


def test_normal_small_collapse_large():
    torch.manual_seed(0)
    B, D = 4096, 32
    normal = _loss(torch.randn(B, D)).item()

    # 完全坍缩:所有样本相同(常量),分布退化为点 → 强烈偏离 N(0,1)
    collapsed = _loss(torch.ones(B, D) * 0.3).item()

    # 低秩坍缩:仅一个方向有方差,其余为 0 → 各向同性检验应同样报警
    z = torch.zeros(B, D)
    z[:, 0] = torch.randn(B)
    lowrank = _loss(z).item()

    print(f"normal={normal:.4f}  collapsed={collapsed:.4f}  lowrank={lowrank:.4f}")
    # 统计量含 ×B(Epps-Pulley)标度:H0 下收敛到 O(1) 小常量(非 0),坍缩为 O(B)。判别力在比值。
    assert normal < 3.0, f"标准正态应为 O(1) 小常量,实测 {normal}"
    assert collapsed > 50.0 * normal, f"坍缩应远大于正态:{collapsed} vs {normal}"
    assert lowrank > 50.0 * normal, f"低秩应远大于正态:{lowrank} vs {normal}"


def test_scale_mismatch_flagged():
    # 各向同性但方差≠1(N(0, 4I)):特征函数与 N(0,1) 不符,应被惩罚 > 标准正态
    torch.manual_seed(0)
    B, D = 4096, 32
    normal = _loss(torch.randn(B, D)).item()
    scaled = _loss(torch.randn(B, D) * 2.0).item()
    print(f"normal={normal:.4f}  scaled(σ=2)={scaled:.4f}")
    assert scaled > normal, f"方差失配应被惩罚:{scaled} vs {normal}"


if __name__ == "__main__":
    test_scalar_nonneg_and_grad()
    test_normal_small_collapse_large()
    test_scale_mismatch_flagged()
    print("SIGReg: all tests passed.")
