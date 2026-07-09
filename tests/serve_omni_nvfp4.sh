#!/usr/bin/env bash
# 启动 Nemotron-3-Nano-Omni NVFP4 的 vLLM OpenAI 服务(单卡 RTX 5090 32GB)。
#
# 环境前提(见 knowledge/conclusion_omni_nvfp4_5090.md §1):
#   - 驱动 570.x ⇒ 最高 CUDA 12.8 ⇒ 必须用 cu129 轮子,PyPI 默认的 vllm/torch 是 cu13 构建,跑不起来
#       pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
#           --index-url https://download.pytorch.org/whl/cu129
#       pip install "vllm==0.20.0+cu129" \
#           --extra-index-url https://wheels.vllm.ai/0.20.0/cu129 \
#           --extra-index-url https://download.pytorch.org/whl/cu129
#       pip install librosa soundfile      # 任何音频输入都需要
#
# 与官方 model card 的差异(32GB 卡的必要收缩):
#   --max-model-len   131072 -> 32768   (权重 21GB 后 KV 预算不足)
#   --max-num-seqs    384    -> 8       (慢系统 worker 并发极低,不需要大 batch)
#
# 坑:model card 的 "RTX Pro: append --moe-backend triton" **对 NVFP4 权重无效**——
# triton 不在 NVFP4 MoE 的后端集合里,vLLM 直接 ValueError:
#   Expected one of ['cutlass','flashinfer_trtllm','flashinfer_cutlass',
#                    'flashinfer_cutedsl','marlin','emulation']
# 那条建议只适用于 BF16/FP8 权重。NVFP4 走默认 auto,让 oracle 自己挑(sm_120 实测选中
# FLASHINFER_CUTLASS)。MOE_BACKEND 留作覆盖口。
#
# 坑 2(驱动 570 / sm_120 必踩):C-RADIO 视觉编码器默认走 vLLM 自带的 FlashAttention-2
# (torch.ops._vllm_fa2_C.varlen_fwd),该内核对 sm_120 只发 PTX 不发 cubin,而 570 驱动的
# PTX JIT 吃不下 CUDA 12.9 的 PTX ⇒ 加载期 profile_run 直接
#   torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain
# LLM 主干不受影响(它选的是 FlashInfer,有预编译 cubin)。故必须把**编码器**注意力换掉。
# 最小复现:tests/probe_sm120_ptx.py
#
# 坑 3(同上环境):FlashInfer 需要为 sm_120 **JIT 编译** cutlass FP8 GEMM,但它的
# _normalize_cuda_arch() 硬编码 "SM 12.x requires CUDA >= 12.9";本机系统 nvcc 是 12.8
# (驱动上限也是 12.8)⇒ TARGET_CUDA_ARCHS 变空集 ⇒
#   RuntimeError: No supported CUDA architectures found for major versions [12].
# 实测 nvcc 12.8 **完全能编 compute_120a/sm_120a**(见下),所以那是个过严的版本守卫。
# FLASHINFER_CUDA_ARCH_LIST 带字母后缀时 flashinfer 原样采纳、跳过 normalize ⇒ 绕过。
#   $ nvcc -gencode=arch=compute_120a,code=sm_120a -cubin -o /dev/null t.cu   # OK
set -euo pipefail

# 见坑 3。首次启动会 JIT 编译 flashinfer 的 sm120 内核(数分钟),之后走 ~/.cache 命中。
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0a}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

MODEL="${MODEL:-/workspace/models/omni-nvfp4}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
MOE_FLAG=()
if [ "$MOE_BACKEND" != "auto" ]; then MOE_FLAG=(--moe-backend "$MOE_BACKEND"); fi

exec vllm serve "$MODEL" \
  --served-model-name nemotron_3_nano_omni \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 8 \
  --video-pruning-rate 0.5 \
  --allowed-local-media-path / \
  --limit-mm-per-prompt '{"video": 1, "image": 2, "audio": 1}' \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${MOE_FLAG[@]}" \
  --mm-encoder-attn-backend "${MM_ENC_ATTN:-TORCH_SDPA}" \
  --kv-cache-dtype fp8
