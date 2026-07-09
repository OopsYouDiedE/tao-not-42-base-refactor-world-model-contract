#!/usr/bin/env python3
"""验证 Mamba 递归状态"不论序列多长都用固定存储",对照 Transformer KV=O(L)。

Mamba 自回归推理只保留固定大小的 (conv_state, ssm_state),与已处理长度 L 无关;
Transformer 必须缓存每步 KV,内存 ∝ L。这是"喂长帧流给模型"是否可承受的决定性依据。

测:①同一 Mamba2 层在 max_seqlen=1K vs 1M 下分配的状态字节完全相同;②步进数千步
峰值显存持平;③同规模 Transformer 的 KV 字节随 L 线性增长(公式+实测各一)。
"""
import argparse

import torch

from mamba_ssm import Mamba2


def state_bytes(cache):
    return sum(t.numel() * t.element_size() for t in cache)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)   # 对照 Transformer 层数
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()
    dev = "cuda"
    dt = torch.bfloat16
    headdim = args.d_model // args.heads

    m = Mamba2(d_model=args.d_model).to(dev).to(dt).eval()

    print("=" * 64)
    print("① Mamba 状态大小 vs 序列长度(allocate_inference_cache)")
    print(f"{'max_seqlen':>12} {'conv_state':>18} {'ssm_state':>18} {'总字节':>12}")
    for L in (1_000, 100_000, 1_000_000, 100_000_000):
        conv, ssm = m.allocate_inference_cache(1, L, dtype=dt)
        print(f"{L:>12,} {str(tuple(conv.shape)):>18} {str(tuple(ssm.shape)):>18} "
              f"{state_bytes((conv, ssm)):>12,}")
    print("→ 状态形状/字节与 max_seqlen 完全无关(固定存储)。")

    print("=" * 64)
    print("② Mamba 步进峰值显存 vs 已处理步数(状态复用)")
    conv, ssm = m.allocate_inference_cache(1, args.steps, dtype=dt)
    print(f"{'步进到':>10} {'峰值显存(MB)':>16}")
    with torch.no_grad():
        marks = {int(args.steps * f) for f in (0.1, 0.3, 0.6, 1.0)}
        torch.cuda.reset_peak_memory_stats()
        for s in range(1, args.steps + 1):
            x = torch.randn(1, 1, args.d_model, device=dev, dtype=dt)  # step 需 [B,1,D]
            m.step(x, conv, ssm)
            if s in marks:
                print(f"{s:>10,} {torch.cuda.max_memory_allocated() / 1e6:>16.1f}")
    print("→ 步进越多,峰值显存持平(单步 O(1),与已处理长度无关)。")

    print("=" * 64)
    print("③ 同规模 Transformer KV 缓存 vs 序列长度(公式 + 实测)")
    print(f"{'序列长度L':>12} {'KV字节(公式)':>18} {'实测allocated(MB)':>20}")
    for L in (1_000, 10_000, 100_000, 1_000_000):
        kv_bytes = 2 * args.layers * args.heads * L * headdim * 2   # K+V, bf16
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        try:
            kv = torch.zeros(2, args.layers, args.heads, L, headdim, device=dev, dtype=dt)
            meas = f"{(torch.cuda.memory_allocated() - base) / 1e6:.1f}"
            del kv
        except torch.OutOfMemoryError:
            meas = f"OOM(需{kv_bytes / 1e9:.1f}GB)"
        print(f"{L:>12,} {kv_bytes:>18,} {meas:>20}")
    print("→ KV 随 L 线性增长(每 token 每层每头都要存 K,V)。")
    print("=" * 64)
    print("结论:Mamba 固定存储(状态与L无关)⇒ 长帧流可承受;Transformer KV=O(L)⇒ 长流爆内存。")


if __name__ == "__main__":
    main()
