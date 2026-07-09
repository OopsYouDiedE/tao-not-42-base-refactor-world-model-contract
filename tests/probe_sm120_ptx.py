#!/usr/bin/env python3
"""sm_120(RTX 5090)+ 驱动 570 上,哪些注意力内核能用?

背景:vLLM 0.20.0 的 cu129 轮子里,自带的 FlashAttention-2 扩展(`_vllm_fa2_C`)对
sm_120 **只发 PTX、不发 native cubin**。PTX→SASS 的 JIT 编译器住在**驱动**里,
而 570 驱动最高只认 CUDA 12.8 的 PTX ⇒ 加载 CUDA 12.9 的 PTX 直接:

    torch.AcceleratorError: CUDA error: the provided PTX was compiled with
    an unsupported toolchain.   (cudaErrorUnsupportedPtxVersion)

这不是"显存不够"也不是"算力不支持",而是**工具链版本**问题。驱动是宿主注入的,
容器里改不了 ⇒ 只能绕开该内核(见 tests/serve_omni_nvfp4.sh 的 --mm-encoder-attn-backend)。

注意:Nemotron-Omni 的 **LLM 主干不受影响**——vLLM 给它选的是 FlashInfer(装了预编译
cubin 包 flashinfer-cubin);只有 C-RADIO 视觉编码器那条路默认硬走 FA2。

用法:  python tests/probe_sm120_ptx.py
"""
from __future__ import annotations

import torch


def main() -> None:
    cap = torch.cuda.get_device_capability()
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"capability  : sm_{cap[0]}{cap[1]}")
    print(f"torch       : {torch.__version__}")
    print(f"driver CUDA : {torch.version.cuda} (build)")
    print()

    results: dict[str, str] = {}

    # 1) torch 原生 SDPA —— torch 轮子带 sm_120 cubin,应当可用
    try:
        q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        results["TORCH_SDPA"] = "OK"
    except Exception as e:  # noqa: BLE001
        results["TORCH_SDPA"] = f"FAIL {type(e).__name__}: {str(e)[:70]}"

    # 2) vLLM 自带 FA2 varlen —— 视觉编码器默认用的就是它
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        qq = torch.randn(512, 8, 64, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 512], device="cuda", dtype=torch.int32)
        flash_attn_varlen_func(
            qq, qq, qq, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=512, max_seqlen_k=512, fa_version=2,
        )
        torch.cuda.synchronize()
        results["VLLM_FA2"] = "OK"
    except Exception as e:  # noqa: BLE001
        results["VLLM_FA2"] = f"FAIL {type(e).__name__}: {str(e)[:70]}"

    for k, v in results.items():
        print(f"{k:12s} {v}")

    if results["VLLM_FA2"].startswith("FAIL") and results["TORCH_SDPA"] == "OK":
        print("\n=> 本机需要 --mm-encoder-attn-backend TORCH_SDPA")


if __name__ == "__main__":
    main()
