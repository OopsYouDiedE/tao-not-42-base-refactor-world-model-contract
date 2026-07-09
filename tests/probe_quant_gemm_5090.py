#!/usr/bin/env python3
"""同卡 bf16 vs FP8 vs NVFP4 GEMM 实测(RTX 5090 / sm_120)。

回答:"我们自己的世界模型该不该上 NVFP4 / FP8?"

为什么需要这个脚本:`knowledge/analysis_efficiency_levers.md §2` 那行
    NVFP4(租 5090) | 5-10× | FP4 算力 ~14× L4 bf16,带宽 1792 vs 300GB/s=6×
把**换卡**(L4→5090)与**换精度**(bf16→FP4)两个变量混在了一起。光换卡就有 ~3.4× 算力、
6× 带宽。本脚本只测**同一张 5090 上、同一个 shape**,精度换来的边际收益。

测的 shape 取自本仓真实维度(net/dreamerv3/config.py):
    deter=512, units=512, stoch=32x32 ⇒ feat_dim=1536
想象 rollout 的主 GEMM 形如 [B, 1536] x [1536, 512]。
另附一个大 shape(8192x8192)作对照,展示"张量核要吃饱才有 4bit 红利"。

用法:  python tests/probe_quant_gemm_5090.py
"""
from __future__ import annotations

import torch

from vllm import _custom_ops as ops

FP8 = torch.float8_e4m3fn
WARMUP, ITERS = 10, 50


def timeit(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS  # ms


def bench_bf16(m, k, n) -> float:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    return timeit(lambda: torch.mm(a, b))


def bench_fp8(m, k, n) -> float:
    a = torch.randn(m, k, device="cuda").to(FP8)
    b = torch.randn(n, k, device="cuda").to(FP8).t()   # column-major B
    sa = torch.tensor(1.0, device="cuda")
    sb = torch.tensor(1.0, device="cuda")
    return timeit(lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb,
                                           out_dtype=torch.bfloat16))


def bench_nvfp4(m, k, n, dynamic_act: bool) -> float | None:
    """NVFP4: E2M1 元素 + FP8(E4M3) 块缩放(gs=16) + FP32 全局缩放。

    dynamic_act=False: 激活也预量化 —— **这是作弊**,只在对比 kernel 峰值时有意义。
    dynamic_act=True : 权重离线量化一次(真实),激活每步现量化(真实)。

    关键量纲:激活量化开销 ∝ M*K(访存),GEMM 收益 ∝ M*K*N(算力)。
    ⇒ 量化开销只有在 **N 大** 时才被摊薄。Dreamer 的 N=512 很小,故收益远低于 LLM。
    """
    try:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        gsa = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        gsb = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        bq, bsf = ops.scaled_fp4_quant(b, gsb)          # 权重:离线一次
        alpha = (1.0 / (gsa * gsb)).to(torch.float32)
        if dynamic_act:
            def run():
                aq, asf = ops.scaled_fp4_quant(a, gsa)  # 激活:每步现量化
                return ops.cutlass_scaled_fp4_mm(aq, bq, asf, bsf, alpha, torch.bfloat16)
        else:
            aq, asf = ops.scaled_fp4_quant(a, gsa)
            def run():
                return ops.cutlass_scaled_fp4_mm(aq, bq, asf, bsf, alpha, torch.bfloat16)
        return timeit(run)
    except Exception as e:  # noqa: BLE001
        print(f"    nvfp4 unavailable for {(m, k, n)}: {type(e).__name__}: {str(e)[:60]}")
        return None


def tflops(m, k, n, ms) -> float:
    return 2 * m * k * n / (ms * 1e-3) / 1e12


def main() -> None:
    print(torch.cuda.get_device_name(0), "| sm_%d%d" % torch.cuda.get_device_capability())
    print()
    # (M, K, N, 说明).  M = 想象 rollout 的并行轨迹数(batch)
    SHAPES = [
        (1024, 1536, 512, "dreamer feat->units, batch 1k"),
        (4096, 1536, 512, "dreamer feat->units, batch 4k"),
        (16384, 1536, 512, "dreamer feat->units, batch 16k"),
        (65536, 1536, 512, "dreamer feat->units, batch 64k"),
        (16384, 512, 512, "dreamer units->units, batch 16k"),
        (16384, 8192, 8192, "LLM 级大 GEMM(对照)"),
    ]
    hdr = (f"{'shape (M,K,N)':>22} {'bf16':>8} {'fp8':>8} {'fp4 pre-q':>10} "
           f"{'fp4 real':>9} {'fp8 x':>6} {'fp4 x(real)':>12} {'bf16 TF':>8}")
    print(hdr)
    print("-" * len(hdr))
    print("  (fp4 pre-q = 激活也预量化,作弊基准;fp4 real = 激活每步现量化,可部署)")
    for m, k, n, note in SHAPES:
        t16 = bench_bf16(m, k, n)
        t8 = bench_fp8(m, k, n)
        t4p = bench_nvfp4(m, k, n, dynamic_act=False)
        t4r = bench_nvfp4(m, k, n, dynamic_act=True)
        s8 = t16 / t8 if t8 else float("nan")
        s4 = (t16 / t4r) if t4r else float("nan")
        print(f"{str((m, k, n)):>22} {t16:8.3f} {t8:8.3f} "
              f"{(t4p or 0):10.3f} {(t4r or 0):9.3f} {s8:6.2f} {s4:12.2f} "
              f"{tflops(m, k, n, t16):8.1f}   {note}")
    print()
    print("读法:激活量化开销 ∝ M*K,GEMM 收益 ∝ M*K*N ⇒ 只有 N 大时 4bit 才划算。")
    print("Dreamer 的 N=512 ⇒ batch 1k 时 fp4 反而更慢;LLM 的 N=8192 ⇒ 5x。")


if __name__ == "__main__":
    main()
