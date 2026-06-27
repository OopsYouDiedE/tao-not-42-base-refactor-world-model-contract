---
name: hunyuan-gamecraft-1-architecture
description: Hunyuan-GameCraft-1.0 详细架构（基于源码）：CameraNet 相机编码 + 双流单流 Transformer
metadata: 
  node_type: memory
  type: reference
  originSessionId: 49002e33-dee1-4a07-9f37-40d6de00c647
---

# Hunyuan-GameCraft-1.0 架构详解（基于源码分析）

**仓库**: https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0  
**论文**: arXiv:2506.17201  
**官网**: https://hunyuan-gamecraft.github.io/

## 核心参数（从源码提取）

```python
# hymm_sp/modules/models.py - HYVideoDiffusionTransformer
hidden_size: 3072
num_heads: 24
head_dim: 128  # 3072 / 24
depth_double_blocks: 19  # 双流块
depth_single_blocks: 38  # 单流块
mlp_width_ratio: 4.0
mlp_act_type: "gelu_tanh"
patch_size: [1, 2, 2]  # T, H, W
rope_dim_list: [16, 56, 56]  # RoPE 维度分配（时间+空间）
qk_norm: True  # RMSNorm
```

**总参数量**: 未明确标注（估计 8-10B，接近 HunyuanVideo）

## 架构设计

### 与 HunyuanVideo-1.5 的关系

**核心发现**: GameCraft-1.0 = HunyuanVideo-1.5 基础架构 + **CameraNet 动作编码模块**

| 组件 | GameCraft-1.0 | HunyuanVideo-1.5 |
|------|---------------|------------------|
| 双流块 | 19 | 20 |
| 单流块 | 38 | 40 |
| hidden_size | 3072 | 3072 |
| num_heads | 24 | 24 |
| 动作控制 | ✅ CameraNet | ❌ |
| 稀疏注意力 | ❌ | ✅ SSTA |

### 1. CameraNet（动作编码核心）

**输入**: 相机状态张量 `(batch, frames, 6, H, W)`
- **6 通道**: 相机参数（可能是：位置xyz + 旋转roll/pitch/yaw，或其他6-DoF表示）

**架构流程**:

```python
# hymm_sp/modules/cameranet.py
Input: (B, F, 6, 704, 1280)
  ↓
PixelUnshuffle(downscale=8)  # 空间下采样 8×
  → (B×F, 384, 88, 160)  # 6×8²=384 channels
  ↓
Conv2d + GroupNorm + ReLU (384 → 192)
  ↓
compress_time(2×)  # 时间压缩：33帧 → 17帧
  ↓
Conv2d + GroupNorm + ReLU (192 → 96)
  ↓
compress_time(2×)  # 17帧 → 9帧
  ↓
Conv2d(96 → 16)  # 零初始化
  ↓
PatchEmbed([1,2,2], 16 → 3072)
  ↓
Output × learnable_scale
```

**时间压缩策略** (`compress_time`):

```python
# 特殊处理确保关键帧保留
if frames == 66 or frames == 34:
    # 分两段：各保留首帧 + 池化其余
    segment1: keep_first + avg_pool(rest, kernel=2)
    segment2: keep_first + avg_pool(rest, kernel=2)
    concat(segment1, segment2)
elif frames % 2 == 1:  # 奇数帧
    keep_first_frame + avg_pool(rest, kernel=2, stride=2)
else:  # 偶数帧
    avg_pool(all, kernel=2, stride=2)
```

**初始化策略**:
- 前两层卷积：He 初始化（正态分布，std=√(2/fan_in)）
- 最终投影：**零初始化**（residual 友好，训练初期无影响）
- Scale 参数：初始化为 1（可学习）

### 2. Transformer 主干

**双流块 (DoubleStreamBlock)** × 19:
- 图像流和文本流独立 Q/K/V 投影
- RoPE 位置编码（仅图像，dim=[16,56,56]）
- Flash Attention 或 Sequence Parallel Attention
- ModulateDiT 条件化（6-factor）

**单流块 (SingleStreamBlock)** × 38:
- 图像+文本 concat 后联合处理
- 共享 Q/K/V + MLP
- ModulateDiT 条件化（3-factor）

### 3. 动作注入方式

```python
# 推测的注入点（从代码结构）
img_tokens = PatchEmbed(image)  # 图像 token
camera_tokens = CameraNet(camera_states)  # 相机 token

# 可能是加法注入（类似 ControlNet）
img_tokens = img_tokens + camera_tokens

# 或者拼接后进入 Transformer
tokens = concat([img_tokens, txt_tokens, camera_tokens])
```

**注意**: 源码中 CameraNet 输出被 scale 参数缩放，初始≈0 影响，逐步学习增强。

## 训练数据

- **100万+ 游戏录像**
- **100+ 款 3A 游戏**
- 相机参数通过 VIPE 等工具从游戏视频重建

## 推理流程

```python
# 用户提供
reference_image: PIL.Image
prompt: str
action_list: ["w", "s", "d", "a"]  # 键盘动作序列
action_speed: [0-3, ...]  # 每个动作的速度

# 内部流程
1. 动作 → 相机参数（6-DoF轨迹生成）
2. 相机参数 → CameraNet → camera_tokens
3. image + text + camera_tokens → Transformer
4. 50步 Diffusion 采样（CFG=2.0）
5. VAE 解码 → 视频输出（704×1216, 33帧）
```

## 蒸馏版本

**8步推理模型**:
- CFG 蒸馏 + 步数蒸馏
- 推理速度提升约 6×
- 与 HunyuanVideo-1.5 使用相同的蒸馏技术

## 与 GameCraft-2 的对比

| 维度 | GameCraft-1.0 | GameCraft-2 |
|------|---------------|-------------|
| **交互方式** | 键盘/鼠标 → 相机参数 | 文本 + 键盘/鼠标 |
| **语义理解** | ❌ 无 | ✅ Qwen2 MLLM |
| **相机编码** | CameraNet (6通道) | Plücker 嵌入 |
| **架构规模** | ~8-10B DiT | 14B MoE |
| **训练数据** | 100万游戏录像 | 同样规模 |
| **开源状态** | ✅ GitHub | ❌ 仅论文 |

**进化路径**: 
```
GameCraft-1.0 (固定动作) → GameCraft-2 (语义控制)
  相机参数编码              文本驱动交互 + MoE
```

## 对 CraftGround 项目的适用性

### 优势
- ✅ **开源可用**: 代码、权重完整
- ✅ **动作条件**: 支持键盘/鼠标输入
- ✅ **相机可控**: 直接控制视角运动
- ✅ **蒸馏版本**: 8步推理速度快

### 局限
- ⚠️ **非 RL 世界模型**: 是生成模型，不能用于在线规划
- ⚠️ **离散动作映射**: WASD → 6-DoF 映射对 Crafter 可能不适配
- ⚠️ **显存需求**: 需要 24GB+ （虽然比 GameCraft-2 小）

### 可行方向

**1. 动作条件数据增强**:
```python
# 用 GameCraft-1.0 生成条件化数据
for state in replay_buffer:
    action_seq = policy.get_actions(state)
    augmented_video = GameCraft.generate(
        reference_image=state.obs,
        action_list=action_seq,
        prompt=f"Crafter environment, {action_desc}"
    )
    # 用于增强训练数据多样性
```

**2. 逆向奖励模型**:
```python
# 训练：给定(state, action) → 预测未来帧
# 奖励：生成帧与真实帧的接近度
reward = -||GameCraft(s, a) - real_next_frame||
```

**3. 想象规划**:
```python
# Model-based RL 的"想象"部分
for candidate_action in action_space:
    imagined_future = GameCraft(current_state, candidate_action)
    value = value_network(imagined_future)
# 选择最优 action
```

### 不可行
- ❌ 直接替换 DreamerV3（训练目标不同）
- ❌ 实时在线决策（推理速度仍不够）

## 实现细节备注

**CPU Offload 支持**:
```python
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", 0)
# 在前向传播中多次调用 torch.cuda.empty_cache()
```

**Sequence Parallelism**:
```python
DISABLE_SP = os.environ.get("DISABLE_SP", 0)
# 支持跨GPU的序列并行（长视频生成）
```

**SageAttention 优化**:
- 可选替代 FlashAttention
- 进一步加速（与 HunyuanVideo-1.5 共享）

## 关联 Memory

- [[hunyuanvideo-15-architecture]] - 基础架构相同
- [[hunyuan-gamecraft-2-architecture]] - 下一代版本

**总结**: GameCraft-1.0 是第一个开源的动作条件游戏视频生成模型，对 RL 研究有潜在价值，但不能直接作为世界模型使用。最适合用于**数据增强**和**辅助奖励设计**。