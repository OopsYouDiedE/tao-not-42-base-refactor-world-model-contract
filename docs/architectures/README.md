# 外部模型架构分析

本目录包含对外部模型的架构分析(现存 Nemotron 系列两篇)。
腾讯混元视频系列三篇(HunyuanVideo-1.5 / GameCraft-1.0 / GameCraft-2)已随
视频世界模型线退役删除(2026-07-10),完整分析在 git 历史 commit `1a29855` 及之前。

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

### 1. [Nemotron-3-Nano-Omni](./nemotron-3-nano-omni-architecture.md)
- **参数**: 31.6B 总 / 3.2B 激活(混合 Mamba-MoE)
- **特性**: C-RADIOv4 视觉 + Parakeet 音频,262K 上下文,NVFP4 实测 **21.5GiB** 单卡 5090 自托管
- **开源**: ✅ **NVIDIA Open Model Agreement** (arXiv:2604.24954)(更正:非 CC BY 4.0)
- **用途**: 本项目慢系统 VLM worker 生产候选;含 L2b prefill 直读位置标注
- **实测**: knowledge/README.md(原生加载/量化结构/延迟);
  knowledge/README.md(零样本像素直控失败)

### 2. [Nemotron-Labs-TwoTower](./nemotron-twotower-architecture.md)
- **参数**: ~60B 总(30B 冻结 AR 塔 + 30B 可训扩散塔),每塔激活 ~3B
- **特性**: 块扩散改装冻结 AR 骨干,98.7% 质量 / 2.42× 吞吐
- **开源**: ✅ 仅 Base (arXiv:2606.26493)
- **用途**: "冻结通用塔+可训增量塔"配方研究(其 dreamer4 移植对应表随该线退役,仅存参考)

## 分析日期

2026-06-28 首建;2026-07-10 清理(混元三篇删除,见 git 历史)
