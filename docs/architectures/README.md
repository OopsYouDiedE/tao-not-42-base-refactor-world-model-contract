# 视频生成模型架构分析

本目录包含对腾讯混元视频生成模型系列的详细架构分析。

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
