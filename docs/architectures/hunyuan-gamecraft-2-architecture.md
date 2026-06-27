---
name: hunyuan-gamecraft-2-architecture
description: Hunyuan-GameCraft-2 详细架构：14B MoE + 文本驱动交互注入 + Sink Token + ReCache
metadata: 
  node_type: memory
  type: reference
  originSessionId: 49002e33-dee1-4a07-9f37-40d6de00c647
---

# Hunyuan-GameCraft-2 架构详解（基于论文分析）

**GameCraft-2 论文**: arXiv:2511.23429  
**GameCraft-2 官网**: https://hunyuan-gamecraft-2.github.io/  
**GameCraft-2 代码**: ❌ 暂未开源

**GameCraft-1.0 代码**: ✅ https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0

## 核心参数

- **规模**: 14B 参数
- **架构**: Image-to-Video Mixture-of-Experts (MoE) Diffusion Model
- **专家分离**: 高噪声专家 + 低噪声专家（不同学习率）

## 核心创新

### 1. 文本驱动交互注入机制

**突破**: 从固定键盘输入 → 自然语言语义控制

```
输入方式:
├─ 自然语言: "open the door", "trigger explosion"
├─ 键盘: W/A/S/D/Space
└─ 鼠标: ↑/←/↓/→

交互能力:
├─ Camera Motion (相机运动)
├─ Character Behavior (角色行为)
└─ Environment Dynamics (环境动态)
```

### 2. 网络组件

#### 相机控制注入

```python
离散动作 (W/A/S/D/...)
  ↓
6-DoF 相机参数
  ↓
Plücker 嵌入编码
  ↓
Token 加法注入到 MoE Transformer
```

#### 交互提示处理器

- **基于 Qwen2 的多模态 LLM**
- 提取并推理交互信息
- 区分场景描述 vs. 精细交互行为

### 3. 注意力机制

#### Sink Token 策略

```python
# 初始帧永久保留为 Sink Token
KV_cache[0] = initial_frame  # 提供坐标系原点
# 防止质量退化，保证相机参数对齐
```

#### 块稀疏注意力

```
结构: [Sink tokens] + [Local window N frames] + [Target block]
Target block attends to:
  - Sink tokens (永久)
  - 前 N 帧局部窗口
```

#### Context Parallelism

- 训练时使用 CP=4（后期阶段）
- 分布式处理长视频序列

### 4. KV Cache 管理

#### 固定长度缓存 + 滚动更新

```python
Cache 结构:
├─ Sink tokens (固定不变)
└─ Local window (size K, rolling update)

总长度: L = len(sink) + K
```

#### ReCache 机制（多轮交互关键）

```python
def recache(new_interaction_prompt):
    # 收到新交互提示时
    recompute(last_autoregressive_block)  # 仅重算最后一块
    update(self_attn_cache)
    update(cross_attn_cache)
    # 开销极小，保证多轮精确交互
```

## 训练流程（4 阶段）

### 阶段 1: 动作注入训练 (100k iters)

```yaml
课程学习: 45 → 81 → 149 frames @ 480p
数据: 随机长/短 caption + 交互 caption
目标: Flow-matching objective
学习: Camera encoder + MoE experts
```

### 阶段 2: 指令导向 SFT (20k iters)

```yaml
数据: 150K 样本 (游戏录像 + 合成数据)
冻结: Camera encoder
微调: MoE experts only
目标: 增强指令遵循能力
```

### 阶段 3: 自回归蒸馏 (10k iters)

```yaml
方法: Self-Forcing (扩展到 14B MoE)
对齐: Distributional Moment Distance (DMD)
策略: 交替 self-forcing 和 teacher-forcing
目标: 提升长视频生成质量
```

### 阶段 4: 随机长视频调优 (3k iters)

```yaml
采样: 从 N 帧 rollout 中均匀采样 T 帧窗口
扩展: 随机长度 K → N_max
Student: 使用预测的 V[i-1]
Teacher: 使用真实历史
```

**损失函数**:

```python
L = DMD(
    T_fake(x_t(W), t, c_student),  # Student 条件于预测历史
    T_real(x_t(W), t, c_teacher)   # Teacher 条件于真实历史
)
```

## 数据构建

### 合成数据生成

1. **起止帧策略**: VLM 引导图像编辑 → 显式状态转换（如开门）
2. **首帧驱动**: 从初始帧自由生成 → 动态相机运动

### 游戏数据处理管道

```
原始游戏录像 (100+ 款 3A 游戏)
  ↓ PySceneDetect (6秒片段分割)
场景片段
  ↓ RAFT 光流 (动作边界定位)
动作标注
  ↓ VIPE (6-DoF 相机轨迹重建)
相机参数 + 原始帧
  ↓ 双重 Captioning
最终数据: [C_t, I_{t→t+1}]

交互计算: I_{t→t+1} = Δ(Φ(C_{t+1}), Φ(C_t))
```

## 推理优化（实时生成）

```
FP8 量化
+ 并行 VAE 解码
+ SageAttention (替代 FlashAttention)
+ Sequence Parallelism (多 GPU)
= 16 FPS 实时生成
```

## 多轮交互生成

```python
Loop:
  1. Generate block (N frames)
  2. Update KV cache (rolling)
  3. Receive new interaction prompt
  4. ReCache last block  # 关键：仅重算最后一块
  5. Continue generation with new prompt
```

## 与 Genie 2 对比

| 特性 | GameCraft-2 | Genie 2 |
|------|-------------|---------|
| **架构** | 14B MoE Diffusion | 未公开 |
| **交互方式** | 文本+键盘+鼠标 | 键盘+鼠标 |
| **语义理解** | ✅ Qwen2 MLLM | ❌ |
| **开源** | ❌ 仅论文 | ❌ |
| **实时性** | 16 FPS | 未知 |

**关键优势**: 语义级控制（"开门"、"爆炸"）vs. 低级动作（W/A/S/D）

## 对 CraftGround 项目的意义

### 可行方向

1. **奖励模型训练**: 利用语义理解能力设计稀疏奖励
2. **数据增强**: 生成多样化游戏场景（需等开源）
3. **想象规划**: 作为"想象"未来轨迹的模块

### 不可行

- ❌ 直接替换 DreamerV3（这是生成模型，非世界模型）
- ❌ 在线决策（推理速度虽快但仍非实时 RL 要求）

### 待验证

- 能否理解 Crafter 类游戏的物理规则？
- 生成数据的物理一致性如何？
- 能否微调到特定环境？

**结论**: GameCraft-2 是游戏视频生成的重大进展，但对 RL 训练的直接帮助有限。更适合作为辅助工具（数据增强、奖励塑形）而非核心组件。

**关联**: [[hunyuanvideo-15-architecture]]（同源技术栈）