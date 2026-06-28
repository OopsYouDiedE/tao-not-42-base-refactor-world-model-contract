# OpenAI VPT vs DreamerV3 在 Minecraft 上的对比调查报告

**日期:** 2026-06-28  
**目的:** 为蒸馏实验准备 VPT 和 DreamerV3 模型

---

## 1. OpenAI VPT (Video Pre-Training)

### 1.1 核心架构

- **论文:** "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos" (OpenAI, 2022)
- **方法:** 两阶段训练范式
  1. **逆动态模型 (IDM):** 在 2K 小时标注视频上训练，学习从连续帧预测动作
  2. **行为克隆 (BC):** 用 IDM 自动标注 70K 小时 YouTube 无标注视频，进行大规模行为克隆
- **模型规模:** 最大 1.2B 参数 (foundation model)
- **架构:** ResNet 视觉编码器 + Transformer 策略网络

### 1.2 Minecraft 性能表现

#### 基准任务成功率：
- **获取钻石 (Diamond):** VPT 是首个从像素成功获取钻石的模型（需完成复杂技能链）
- **制作钻石镐:** 微调后成功率从基准 0% 提升到 **2.5%**
- **MineRL BASALT 竞赛:** 达到接近人类偏好的排名

#### 训练数据规模：
- 标注数据: 2,000 小时（键鼠标注）
- 无标注数据: 70,000 小时（YouTube 视频）

### 1.3 模型变体

VPT 提供多种规模和训练方式的模型：

#### BC 模型（纯行为克隆）
- `foundation-model-1x`: 基础规模，通用型（167MB）
- `foundation-model-2x`: 2倍宽度
- `foundation-model-3x`: 3倍宽度，最强通用能力
- `bc-house-3x`: 在建筑数据上微调
- `bc-early-game-2x/3x`: 在早期游戏数据上微调

#### RL 强化模型（BC + RL）
- `rl-from-foundation-2x`: **从 foundation 模型 RL 微调，针对钻石获取优化**（108MB）
- `rl-from-house-2x`: 从建筑模型 RL 微调
- `rl-from-early-game-2x`: 从早期游戏模型 RL 微调

**已下载模型:**
- ✅ `foundation-model-1x.model` (2KB) + `foundation-model-1x.weights` (167MB)
- ✅ `2x.model` (4KB) + `rl-from-foundation-2x.weights` (108MB)

---

## 2. DreamerV3 on Minecraft

### 2.1 核心架构

- **论文:** "Mastering Diverse Domains through World Models" (Hafner et al., Nature 2025)
- **方法:** 世界模型 + 想象中训练 (Model-based RL)
  1. 学习潜在世界模型：编码观察 → 预测未来状态和奖励
  2. 在想象的轨迹中训练 Actor-Critic 策略
  3. 完全端到端，无需人类演示数据
- **核心优势:** 统一超参数跨多领域（Atari、DMC、Crafter、Minecraft 等）

### 2.2 Minecraft 实现细节

DreamerV3 代码库包含完整的 Minecraft 支持：

#### 配置 (`configs.yaml`)
```yaml
minecraft: 
  size: [64, 64]           # 观察图像分辨率
  break_speed: 100.0       # 破坏方块速度加速
  logs: False              # 日志记录
  length: 36000            # episode 最大长度（10小时游戏时间）
  task: minecraft_diamond  # 默认任务
```

#### 支持的任务 (`embodied/envs/minecraft_flat.py`)

1. **Wood (收集木头)**
   - 动作空间: 基础动作（移动、跳跃、攻击、转向、放置泥土）
   - 奖励: 每收集 1 个原木 + 健康惩罚

2. **Climb (攀爬高度)**
   - 动作空间: 基础动作
   - 奖励: 高度增量 + 健康惩罚

3. **Diamond (获取钻石)** ⭐
   - 动作空间: 基础动作 + **合成/放置/装备动作**（共 23 个动作）
     - 合成: 木板、木棍、工作台、镐子（木→石→铁）
     - 放置: 工作台、熔炉
     - 装备: 各级镐子
     - 冶炼: 铁锭
   - 奖励: **稀疏里程碑奖励**（首次获得每个物品 +1）
     - 原木 → 木板 → 木棍 → 工作台 → 木镐 → 圆石 → 石镐 → 铁矿 → 熔炉 → 铁锭 → 铁镐 → **钻石**
   - 总共 12 个里程碑，最高 +12 奖励

### 2.3 性能对比

DreamerV3 论文中**没有直接报告 Minecraft Diamond 任务的性能数据**，主要测试环境为：
- Atari 55 (100k 步)
- DMC Proprio/Vision
- **Crafter** (Minecraft 风格的 2D 环境): 14.5 分（DreamerV2: 10.0）

**推测性能:**
- DreamerV3 在 Crafter（包含类似 Minecraft 的合成/探索机制）上表现优异
- 但 Minecraft 3D 环境复杂度远超 Crafter，需要更长训练时间
- **论文未公开 Minecraft 钻石任务的成功率数据**

### 2.4 模型权重

**状态:** ❌ **DreamerV3 官方仓库不提供预训练检查点**

- 官方 repo 仅提供训练代码，需从头训练
- 训练成本较高（需要长时间 GPU 训练）
- 可能需要自行训练或寻找社区分享的权重

---

## 3. 关键对比总结

| 维度 | OpenAI VPT | DreamerV3 |
|------|-----------|-----------|
| **学习范式** | 行为克隆 (BC) + RL | 世界模型 + 想象中训练 (MBRL) |
| **数据需求** | 需要大量人类演示视频 | 纯强化学习，无需演示 |
| **训练数据** | 70K 小时 YouTube 视频 | 环境交互（样本效率更高） |
| **模型规模** | 1.2B 参数 (foundation-3x) | 未明确，但通常较小（~50M-200M） |
| **Minecraft 成功率** | 钻石镐制作 2.5% | **未公开** |
| **预训练权重** | ✅ 公开多个变体 | ❌ 需自行训练 |
| **泛化能力** | 强（学习自人类通用行为） | 中等（需针对任务训练） |
| **样本效率** | 低（需海量数据） | 高（世界模型复用经验） |

---

## 4. 蒸馏实验建议

### 4.1 推荐蒸馏方向

**VPT → 小模型** (推荐优先级: ⭐⭐⭐⭐⭐)
- **优势:**
  - VPT 已有成熟权重，可立即开始蒸馏
  - `rl-from-foundation-2x` (108MB) 已针对钻石任务优化
  - 可提取 VPT 的通用 Minecraft 先验知识
- **方法:**
  - 知识蒸馏: 小模型模仿 VPT 的动作分布
  - 特征蒸馏: 对齐视觉编码器的表征
  - 数据集: 可用 VPT 玩游戏生成轨迹作为演示数据

### 4.2 DreamerV3 作为学生模型

**从 VPT 蒸馏到 DreamerV3 架构** (推荐优先级: ⭐⭐⭐)
- **优势:**
  - DreamerV3 的世界模型可提供更好的样本效率
  - 结合 VPT 的先验 + DreamerV3 的规划能力
- **方法:**
  - 用 VPT 生成高质量轨迹
  - 在这些轨迹上预训练 DreamerV3 的世界模型
  - 继续 RL 微调

### 4.3 评估方案

**基准任务:**
1. **Wood 收集** (简单): 5 分钟内收集木头数量
2. **石镐制作** (中等): 10 分钟内是否制作出石镐
3. **钻石获取** (困难): 60 分钟内是否获取钻石

**对比基线:**
- VPT `rl-from-foundation-2x` (教师模型)
- VPT `foundation-model-1x` (BC 基线)
- 随机策略
- 从头训练的 DreamerV3 (如果时间允许)

**评估指标:**
- 任务成功率 (Success Rate)
- 平均奖励 (Average Reward)
- 样本效率 (Sample Efficiency)
- 推理速度 (FPS)
- 模型大小 (Model Size)

---

## 5. 下一步行动

### 立即可做：
1. ✅ 安装 VPT 依赖并运行 `foundation-model-1x` 测试基线性能
2. ✅ 运行 `rl-from-foundation-2x` 评估教师模型能力
3. ⏳ 设计蒸馏损失函数（动作匹配 + 特征对齐）
4. ⏳ 收集 VPT 生成的演示轨迹作为蒸馏数据

### 中期目标：
5. ⏳ 训练小型学生模型（目标: 10-50MB）
6. ⏳ 对比蒸馏模型 vs VPT 原模型的性能
7. ⏳ 如有必要，从头训练 DreamerV3 在 Minecraft 上作为对比

### 长期探索：
8. ⏳ 研究多任务蒸馏（Wood + Climb + Diamond）
9. ⏳ 探索 VPT + DreamerV3 混合架构
10. ⏳ 在 CraftGround 环境中测试蒸馏模型

---

## 6. 文件位置

```
models/
├── vpt/
│   ├── Video-Pre-Training/          # VPT 官方代码
│   └── weights/
│       ├── foundation-model-1x.model       (2KB)
│       ├── foundation-model-1x.weights     (167MB)
│       ├── 2x.model                        (4KB)
│       └── rl-from-foundation-2x.weights   (108MB) ⭐ 推荐用于蒸馏
└── dreamerv3/
    ├── dreamerv3/                   # DreamerV3 官方代码
    │   ├── embodied/envs/minecraft.py
    │   └── embodied/envs/minecraft_flat.py
    └── checkpoints/                 # (空) 需自行训练
```

**VPT 运行命令:**
```bash
cd models/vpt/Video-Pre-Training
python run_agent.py \
  --model ../weights/2x.model \
  --weights ../weights/rl-from-foundation-2x.weights
```

**DreamerV3 训练命令 (Minecraft Diamond):**
```bash
cd models/dreamerv3/dreamerv3
python dreamerv3/main.py \
  --logdir ~/logdir/minecraft_diamond/{timestamp} \
  --configs minecraft \
  --task minecraft_diamond \
  --run.train_ratio 32
```

---

**报告完成。建议优先评估 VPT 模型性能，再设计蒸馏方案。**
