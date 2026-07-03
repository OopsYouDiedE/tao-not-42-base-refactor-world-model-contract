# 外部模型架构分析

本目录包含对外部模型的详细架构分析:腾讯混元视频生成系列 + NVIDIA Nemotron 新结构 LLM 系列。

## 新结构 LLM 坐标系(2025H2–2026-07 速览)

数学结构有实质新意的谱系(详见各文):
- **混合 SSM/线性注意力**:Nemotron-3(Mamba-2+GQA+MoE)、Qwen3-Next/3.5(Gated DeltaNet 3:1)、
  Kimi Linear(KDA 逐通道门控)、IBM Granite 4.0、Falcon-H——共识:~10:1 稀释比的少量全注意力兜检索;
- **稀疏/压缩注意力**:DeepSeek DSA(token 级稀疏)、GLM-5(MLA+DSA)、MiniMax 稀疏注意力;
  MLA(潜 KV 压缩)成为 MoE 大模型标配;
- **扩散 LM**:Nemotron-TwoTower(冻结 AR 塔上的块扩散改装)、LLaDA/Dream(从零掩码扩散)、
  Gemini Diffusion——块扩散(块间 AR + 块内并行)是当前收敛形态;
- **保守派对照**:MiniMax-M2.5 坚持全 MHA(以可靠性换吞吐)。

## 模型列表

### 1. [HunyuanVideo-1.5](./hunyuanvideo-15-architecture.md)
- **参数**: 8.3B DiT
- **特性**: SSTA 稀疏注意力
- **开源**: ✅ [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- **用途**: 通用文本到视频生成

### 2. [Hunyuan-GameCraft-1.0](./hunyuan-gamecraft-1-architecture.md)
- **参数**: ~8-10B DiT
- **特性**: CameraNet 相机编码（6通道）
- **开源**: ✅ [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0)
- **用途**: 动作条件游戏视频生成

### 3. [Hunyuan-GameCraft-2](./hunyuan-gamecraft-2-architecture.md)
- **参数**: 14B MoE
- **特性**: 文本驱动交互 + Qwen2 MLLM
- **开源**: ❌ 仅论文 (arXiv:2511.23429)
- **用途**: 语义级游戏交互生成

### 4. [Nemotron-3-Nano-Omni](./nemotron-3-nano-omni-architecture.md)
- **参数**: 31.6B 总 / 3.2B 激活(混合 Mamba-MoE)
- **特性**: C-RADIOv4 视觉 + Parakeet 音频,262K 上下文,NVFP4 ~20GB 自托管
- **开源**: ✅ CC BY 4.0 (arXiv:2604.24954)
- **用途**: 本项目慢系统 VLM worker 生产候选;含 L2b prefill 直读位置标注

### 5. [Nemotron-Labs-TwoTower](./nemotron-twotower-architecture.md)
- **参数**: ~60B 总(30B 冻结 AR 塔 + 30B 可训扩散塔),每塔激活 ~3B
- **特性**: 块扩散改装冻结 AR 骨干,98.7% 质量 / 2.42× 吞吐
- **开源**: ✅ 仅 Base (arXiv:2606.26493)
- **用途**: "冻结通用塔+可训增量塔"配方研究;移植 dreamer4 做换游戏快速适应的对应表

## 架构对比

| 特性 | HunyuanVideo-1.5 | GameCraft-1.0 | GameCraft-2 |
|------|------------------|---------------|-------------|
| 参数 | 8.3B | ~8-10B | 14B MoE |
| 双流/单流 | 20/40 | 19/38 | 未知 |
| 动作控制 | ❌ | ✅ | ✅ |
| 语义理解 | ❌ | ❌ | ✅ |
| 开源 | ✅ | ✅ | ❌ |

## 分析日期

2026-06-28
