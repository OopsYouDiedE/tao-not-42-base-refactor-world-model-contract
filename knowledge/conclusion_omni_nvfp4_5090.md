---
name: conclusion-omni-nvfp4-5090
description: 实测结论:Nemotron-3-Nano-Omni NVFP4 在单卡 RTX 5090 上原生加载跑通;权重 21.5GiB、图像 TTFT 0.15s、ASR WER 0;"NVFP4 的 Mamba 混合模型"里 Mamba 本体其实是 BF16
metadata:
  type: conclusion
---

# 结论:Omni NVFP4 在单卡 5090 上原生可用(2026-07-09 实测)

> 环境:Vast.ai RTX 5090 32GB(**sm_120**,驱动 570.153.02 ⇒ driver_max_cuda 12.8),
> 无特权容器(跑不了 Docker)。vLLM 0.20.0+cu129 / torch 2.11.0+cu129 / Python 3.12。
> 权重:`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`(HF,21GB)。
> 复现:`tests/serve_omni_nvfp4.sh` + `tests/probe_omni_nvfp4.py`;
> 原始数据 `docs/results/omni_nvfp4_5090.json`。
> 排障流水见 `docs/activity_log.md` 2026-07-09 条目。

## 1. 裁决:能原生加载,无需任何反量化/格式转换

vLLM 日志同时挂上两条 ModelOpt 路径,直接吃官方 safetensors:

```
Resolved architecture: NemotronH_Nano_Omni_Reasoning_V3
Detected ModelOpt fp8 checkpoint (quant_algo=FP8)
Detected ModelOpt NVFP4 checkpoint
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
Model loading took 21.5 GiB memory and 6.64 seconds
GPU KV cache size: 289,408 tokens
```

| 指标 | 实测 |
|---|---|
| 权重显存 | **21.5 GiB** |
| 加载耗时 | 6.6 s |
| 服务后总显存 | 28.9 / 32.6 GB |
| KV cache(fp8) | 289,408 token ⇒ 131,072 上下文仍有 10.3× 并发 |
| 首次启动 | 525 s(几乎全是 FlashInfer JIT 编译,进 cache 后 ~90 s) |

**model card 的 "Minimum GPU (NVFP4): 1× RTX 5090 32GB" 属实。** 但四个问题必须先解决,
全部与 sm_120 + 570 驱动强相关,见 `docs/activity_log.md`;要点:

1. PyPI 默认 `torch 2.11.0` 是 **CUDA 13** 构建 ⇒ 5090(消费卡,无 forward-compat)跑不起来。必须 cu129 轮子。
2. model card 让 RTX Pro 加 `--moe-backend triton`,**对 NVFP4 权重是无效建议**(triton 不在 NVFP4 MoE 后端集)。
3. C-RADIO 视觉编码器默认走 vLLM 自带 FA2,该内核对 sm_120 **只发 PTX 不发 cubin**,570 驱动 JIT 不了 ⇒ `--mm-encoder-attn-backend TORCH_SDPA`。
4. FlashInfer 硬编码 `SM 12.x requires CUDA >= 12.9`,而本机 nvcc 12.8 **实测能编 `compute_120a`** ⇒ `FLASHINFER_CUDA_ARCH_LIST=12.0a` 绕过。

> **教训**:"驱动版本"与"nvcc 版本"是两件事。驱动决定能不能 JIT 某版本 PTX、能不能加载某 arch 的 cubin;
> nvcc 只决定能编出什么。问题 3 是驱动侧硬限制(只能换内核),问题 4 是编译器侧的**软性版本断言**(可绕)。

## 2. 更正:"NVFP4 的 Mamba 混合模型",Mamba 本体不是 NVFP4

`hf_quant_config.json` 的 `quant_algo` 是 **MIXED_PRECISION**。从 safetensors 张量层面核实
(dtype 直方图 + 逐张量检查):

| 模块 | 张量数 | dtype | bit/param |
|---|---|---|---|
| MoE 路由专家 up/down_proj | 5888 | **U8(NVFP4 packed)** + FP8 块缩放 + FP32 全局缩放 | 4.24 |
| Mamba mixer `in_proj`/`out_proj` | 46 | F8_E4M3 | — |
| Mamba `A_log`/`D`/`conv1d`/`dt_bias`/`norm` | — | **BF16** | — |
| MoE 共享专家 | 46 | F8_E4M3 | 8.00 |
| Attention proj | 6 | F8_E4M3 | — |
| C-RADIO 视觉 / Parakeet 音频 / projector / embed / lm_head | 1289 | BF16 | 16.00 |
| KV cache | — | FP8 | — |

NVFP4 格式实证(layer 1 expert 0):
```
U8      up_proj.weight        [1856, 1344]   1344 x 2 = 2688 = hidden_size   ✓ 每字节 2 个 E2M1
F8_E4M3 up_proj.weight_scale  [1856,  168]   2688 / 16 = 168                ✓ group_size = 16
F32     up_proj.weight_scale_2                                              ✓ 全局 scale
```

**MoE 专家约占 LLM 参数的 93%(≈29.4B / 31.6B)——62GB→21GB 全靠它。
Mamba 的选择性扫描核心(A_log/D/conv1d/dt_bias)原样保留 BF16,投影是 FP8,一点没到 4bit。**

⇒ 已据此更正 `docs/architectures/nemotron-3-nano-omni-architecture.md` 与
`docs/next_session.md`(后者原写"官方 NVFP4 前置 Mamba 留 **BF16**";实际投影是 **FP8**,
只有 SSM 参数是 BF16 —— 结论"NF4 不能碰 Mamba 投影"仍成立,但精度档位记错了)。

整机实测 **5.14 bit/param**(含 BF16 编码器与 embedding);架构文档引的 4.98 bit/权重
是**只算被量化的骨干**,两者不矛盾。

## 3. 效果:4bit 没有可观测的质量损失

BF16(62GB)/FP8(33GB) 都塞不进 32GB ⇒ **本机做不了同模型跨精度对照**。
故不依赖 baseline,改用**有标准答案的客观任务**:

| 探针 | 结果 |
|---|---|
| ASR(JFK 演讲,已知原文) | **WER = 0.0** 逐字正确 |
| OCR(自渲染字符串) | **精确匹配**,标点大小写全对 |
| Crafter 帧语义(本仓训练域) | "草地 + 右侧**三棵**树 + 玩家居中" —— 与 ground truth 完全一致 |
| 文本推理(sheep 陷阱题) | 正确 |
| 像素指点(见 `tests/probe_omni_pointing.py`) | 误差 **2.2–5.4 px** @640×360(<1%) |

## 4. 速度:满足慢系统预算且有余量

| 场景 | TTFT | 解码 |
|---|---|---|
| 纯文本 | 0.087–0.110 s | 257–262 tok/s |
| 图像 640×360 + 短输出 | **0.154 s**(热) | 全程 ~0.20 s |
| 图像 256×256 | 0.099 s(热) | — |

- **冷启动伪影**:首个请求 32.6 s(CUDA graph / FlashInfer autotune 预热)。热态 0.31 s。**不要把首请求计入基准。**
- thinking 模式代价 2–11 s,且会失控(实测 3000 token 输出空答案)⇒ **不能进控制环**,只适合 episode 级慢系统。

慢系统沿用的延迟预算是 0.5–2 s。**单卡 5090 的 NVFP4 Omni 满足该预算,
且有约 3–10× 余量。**

## 5. 反直觉发现:Mamba 的"无限上下文"在这套 serving 栈里不免费

引擎实际配置是 `enable_prefix_caching=False`,`Prefix cache hit rate: 0.0%`。
根因:`nemotron_h.py` 声明了 `SupportsMambaPrefixCaching`,但实际被调度的多模态包装
`nano_nemotron_vl.py` **没有声明** ⇒ vLLM 对整个 Omni 关掉 prefix caching。

后果:多轮对话每一轮都要把之前所有帧**重新 prefill**。实测(640×360,每帧 ~298 token):

| 历史帧数 | prompt token | TTFT |
|---|---|---|
| 1 | 308 | 0.11 s |
| 4 | 1,307 | 0.143 s |
| 8 | 2,639 | **0.240 s** |
| 16 | 5,309 | 0.389 s |
| 40 | 13,325 | 0.823 s |

每多一帧 ≈ **+20 ms**(prefill 吞吐 ≈16k tok/s),严格线性。
(`MM cache hit rate` 93% ⇒ ViT 的图像编码确实被缓存了,重算的是骨干 prefill。)

> **"Mamba 状态常数大小,所以历史可以直接灌"——在显存上成立,在延迟上不成立。**
> 4fps(250ms)实时约束下,无 prefix cache 时最多灌 ~8 帧。
> 想真正"直接灌",需要给 `nano_nemotron_vl` 补上 `SupportsMambaPrefixCaching`;补上后曲线应变平。
> 这是本项目 L2b(prefill-only 状态直读)方案的**前置工程项**,而非可假定的既有能力。

原始数据:`docs/results/omni_history_latency.json`。

## 6. 对本项目的意义

- 慢系统 VLM worker 的**生产候选成立**:单卡 5090 可自托管,延迟有余量,感知质量无损。
- 但**不能直接当控制器**用 —— 见 [conclusion_omni_pixel_control.md](conclusion_omni_pixel_control.md)。
- L2b 直读内部状态方案在动手前要先解决 §5 的 prefix caching 缺口。
