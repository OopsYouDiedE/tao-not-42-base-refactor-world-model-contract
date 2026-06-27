---
name: hunyuanvideo-15-architecture
description: HunyuanVideo-1.5 详细架构：8.3B DiT + SSTA 稀疏注意力 + 3D Causal VAE
metadata: 
  node_type: memory
  type: reference
  originSessionId: 49002e33-dee1-4a07-9f37-40d6de00c647
---

# HunyuanVideo-1.5 架构详解（基于源码分析）

**仓库**: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5

## 核心参数

```python
# 默认配置（hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py）
hidden_size: 3072
heads_num: 24
head_dim: 128  # 3072 / 24
mm_double_blocks_depth: 20  # 双流块数量
mm_single_blocks_depth: 40  # 单流块数量
mlp_width_ratio: 4.0
patch_size: [1, 2, 2]  # 时间、高、宽
```

**总参数量**: 8.3B

## 架构设计

### 1. 双流到单流 Transformer (MM = MultiModal)

**阶段一：双流块 (MMDoubleStreamBlock) × 20**
- 图像流和文本流独立处理
- 各自的 Q/K/V 投影 + 自注意力
- 各自的 LayerNorm + MLP（GeLU_tanh）
- ModulateDiT 条件化（6-factor: shift/scale/gate × 2）

**阶段二：单流块 (MMSingleStreamBlock) × 40**
- 图像+文本 concat 后联合处理
- 共享 Q/K/V 投影
- RoPE 位置编码（仅对图像，dim=[16,56,56]）
- ModulateDiT 条件化（3-factor）

### 2. SSTA 稀疏注意力核心实现

**原理**: Selective and Sliding Tile Attention
- **STA (Spatio-Temporal Attention)**: 局部 3D 块注意力（kernel 窗口）
- **MOBA (Mixture of Block Attention)**: 基于相似度/重要性采样 topk 远程块
- **SSTA = STA ∪ MOBA**: 局部 + 稀疏全局

**关键算法**:

```python
# 1. 相似度采样（similarity_sampling）
gate = einsum("bhsd,bhkd->bhsk", q_block_avg, k_block_avg)
topk_indices = gate.topk(k=topk, dim=-1)

# 2. 重要性采样（importance_sampling）
similarity = einsum("bhsd,bhkd->bhsk", q, k)  # Q-K 相似度
redundancy = einsum("bhsd,bhkd->bhsk", k, k)  # K-K 冗余度
importance = λ * similarity - (1-λ) * redundancy
topk_indices = importance.topk(k=topk, dim=-1)

# 3. 块稀疏注意力（flex_block_attn）
block_mask = STA_mask | MOBA_mask  # 逻辑或合并
output = flex_block_attn_func(q, k, v, block_size, block_mask)
```

**加速效果**: 10秒 720p 视频，相比 FlashAttention-3 实现 **1.87× 端到端加速**

### 3. VAE (AutoencoderKLConv3D)

**类型**: 3D Causal 卷积 VAE  
**压缩比**:
- 空间: 16×（H/16, W/16）
- 时间: 4×（T/4）

**关键特性**:
- `PatchCausalConv3d`: 因果卷积（保证时序一致性）
- 大张量分块处理（>0.6GB 自动切分，避免 OOM）
- 支持 gradient checkpointing

### 4. 文本编码器

```python
ByT5Tokenizer (max_length=256)
  ↓
TextEncoder (custom, 非 CLIP/T5)
  ↓
VisionEncoder (729 semantic tokens, 1152-dim)
  ↓
SingleTokenRefiner (可选，LI-DiT 风格)
```

**Glyph-aware**: 使用 ByT5 字节级分词，增强中文/多语言理解

### 5. 蒸馏技术

**CFG 蒸馏**:
- Teacher: 50步, guidance_scale=6.0
- Student: 50步, guidance_scale=1.0
- 加速: ~2× (减少 CFG 计算)

**步数蒸馏** (480p I2V):
- Teacher: 50步
- Student: 8-12步
- 加速: ~6× (RTX 4090: 75秒/视频)

**稀疏注意力蒸馏** (720p):
- 需要 H 系列 GPU + flex-block-attn
- 结合 SSTA 进一步降低计算量

## 与 GameCraft-2 对比

| 维度 | HunyuanVideo-1.5 | GameCraft-2 |
|------|------------------|-------------|
| **参数** | 8.3B DiT | 14B MoE |
| **架构** | 双流→单流 Transformer | MoE Diffusion |
| **注意力** | SSTA (块稀疏) | Sink Token + Block Sparse |
| **交互控制** | ❌ 无 | ✅ 文本+相机(Plücker) |
| **开源状态** | ✅ GitHub 可用 | ❌ 仅论文 |
| **训练** | Muon 优化器 | 4阶段 (233k iters) |

**Why**: HunyuanVideo-1.5 是通用视频生成，GameCraft-2 专注游戏交互。

## 应用建议

- ✅ **数据增强**: 为 CraftGround 生成多样化场景
- ✅ **快速原型**: 8步推理适合快速迭代
- ⚠️ **世界模型**: 不是动作条件的，无法直接用于 RL 规划
- 🔬 **奖励塑形**: 可尝试用其视觉理解能力辅助稀疏奖励设计

**硬件需求**: 最低 14GB 显存（FP8 + CPU offload）