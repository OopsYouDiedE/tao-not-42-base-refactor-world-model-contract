---
name: nemotron-3-nano-omni-architecture
description: Nemotron-3-Nano-Omni 架构与数学分析:C-RADIOv4 视觉 + Parakeet 音频进混合 Mamba-MoE 骨干;含 L2b prefill 直读位置标注
metadata:
  type: reference
---

# Nemotron-3-Nano-Omni 架构分析(慢系统生产候选)

**技术报告**: arXiv:2604.24954(2026-04)
**权重**: HF `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-{BF16,FP8,NVFP4}`,
许可 **NVIDIA Open Model Agreement**(非 CC BY 4.0;2026-07-09 核对 model card)
**定位(本项目)**: 双系统慢塔 VLM worker 的生产候选；现行接口见 `knowledge/design_bitter_lesson_map_integration.md`。
**实测**: 单卡 RTX 5090 原生加载跑通,见 knowledge/conclusion_omni_nvfp4_5090.md;
零样本像素直控失败,见 knowledge/conclusion_omni_pixel_control.md

## 1. 骨干:52 层混合 Mamba-MoE(与文本版 Nemotron-3-Nano 同构,无结构改动)

23 层 Mamba-2 + 6 层 GQA 注意力(32Q/2KV 头)+ 23 层 MoE(128 路由专家 top-6 + 1 共享),
d_model=2688,总参 31.6B / 激活 ~3.2B,上下文 262,144 token。多模态**不改骨干**,
全部适配发生在输入侧(编码器 + 投影器)。

### 1.1 Mamba-2 / SSD 数学

每层每头维护固定大小矩阵状态 H ∈ R^{N×P}(N=状态维,P=头维),逐 token 线性更新:

```
H_t = a_t · H_{t-1} + B_t x_t^T        (a_t ∈ (0,1) 输入依赖的标量衰减,选择性门)
y_t = C_t^T H_t
```

- **选择性**:a_t、B_t、C_t 均由当前输入投影而来——模型逐 token 决定"写入/衰减/读出"。
- **为什么可并行训练(SSD 定理)**:整个序列变换等价于一个 1-半可分矩阵
  M_ij = C_i^T (∏_{k=j+1..i} a_k) B_j,可分解为"块内稠密矩阵乘(tensor core 友好)+
  块间低秩状态传递(扫描)",即分块并行扫描——训练吞吐与注意力同级,推理退回 O(1)/token 递归。
- **状态容量的信息论边界**:每层状态是常数大小(23 层 × 头数 × N × P 个标量),
  与上下文长度无关 ⇒ 对任意长历史必然有损压缩。精确随机访问(大海捞针)超出状态容量
  即失败——由 6 层 GQA 注意力兜底(KV cache 只在这 6 层增长)。

### 1.2 MoE 路由

router 对 128 专家打分取 top-6 + 恒选共享专家;负载均衡辅助损失防塌缩。
逐 token 知识变换,不参与序列混合。量化部署:路由专家 NVFP4、共享专家/注意力 FP8
(整机 4.98 bit/权重,中位精度损失 <1%)。

> **2026-07-09 张量级核实(conclusion_omni_nvfp4_5090.md §2)**:此处描述正确,但要补一句关键的——
> **Mamba 本体没有被 4bit 化**。`in_proj`/`out_proj` 是 FP8,而 `A_log`/`D`/`conv1d`/`dt_bias`
> **保持 BF16**。NVFP4 只落在 5888 个 MoE 专家张量上(U8 打包 E2M1 + FP8 块缩放 gs=16 + FP32 全局)。
> 专家占 LLM 参数 ~93%,62GB→21GB 全靠它。实测整机 5.14 bit/param(含 BF16 编码器与 embedding);
> 4.98 bit 那个数只算被量化的骨干,两者不矛盾。

## 2. 多模态输入路径(全部在 token 化层,时序交错拼接进同一序列)

### 2.1 视觉:C-RADIOv4-H

- 16×16 patch,保长宽比动态分辨率(弃 tiling),每图 1,024–13,312 个视觉 patch
  (方图对应 512²–1840²);
- ViT 后 **pixel shuffle 4× 下采样**再过 MLP 投影进 LLM——空间信息换通道深度;
- 视频:**Conv3D tubelet** 每 2 相邻帧融合为 1(token 减半),训练采样至 256 帧;
- EVS(Efficient Video Sampling):ViT 后按跨帧余弦不相似度剪枝空间 token,
  50% 剪枝率下 ~70% token 削减,精度损失极小——**静止画面几乎不产生新 token**。

### 2.2 音频:Parakeet-TDT-0.6B-v2(FastConformer)

16kHz 单声道 → log-mel(10ms hop)→ 3 层 stride-2 卷积,合计 ~8× 时间下采样
→ **≈12.5 token/秒**(80ms/token),30 秒一段(~375 token),上下文可容 5 小时音频流。

### 2.3 融合方式

无统一嵌入空间设计——视觉/音频/文本 token **按时间顺序交错拼进同一序列**,
由骨干的 Mamba(流式压缩)+ 6 层注意力(跨模态精确对齐)完成融合。

## 3. 训练:7 阶段渐进课程 + MPO 强化

Stage 0 视觉投影器热身 → 1 视觉+LLM SFT → 2 音频投影器热身 → 3 音频编码器+投影器
→ 4 首次全模态联合 SFT → 5 扩上下文 49K → 6 扩 262K(长文档,冻音频)。
合计 SFT 434M 样本 / 467B token。后训练 MPO(DPO+BCO)+ 分阶段 RL
(文本→图像→全模态→文本)。**每引入一个模态先只训投影器再解冻编码器**——
与本仓"冻结骨干、只训读头"是同一纪律。

## 4. 推理效率(慢系统预算依据)

- B200 单流 500+ token/s(Qwen3-Omni 的 ~3×);多文档 TTFT ≈1.3s;
- 高并发 5000 token/s,iso-interactivity 下 9× Qwen3-Omni 吞吐;
- NVFP4 量化 ~20GB 级显存可自托管。

## 5. 对本项目的接口含义(L2b prefill-only 状态直读位置标注)

L2b 方案 = 帧喂入、零解码、直读内部状态过 adapter(见记忆 vlm-dual-system-integration)。
可读位置按信息性质分三档:

| 读点 | 形状(概念) | 含什么 | 适用 |
|---|---|---|---|
| **23 层 Mamba-2 终态** | 23 × [头数, N, P](常数大小,与会话长度无关) | 整段游玩史的流式压缩摘要("它对这局的全部理解") | 指导 latent 主通道:大小恒定、递交成本不随会话增长 |
| **末层最终位置 hidden** | [2688] | 下一步生成意图的浓缩(最"接近说话"的表示) | 轻量档:单向量,adapter 最小 |
| **末注意力层的逐 token 输出** | [T, 2688](随长度增长) | 保留 token 级定位(哪帧哪个区域) | 需要空间接地的指导(瞄点/区域提示),按需截取 |

工程要点:
1. **骨干冻结 ⇒ 状态分布平稳**,adapter 监督用 L1 文本通路的 hindsight 配对做课程化蒸馏
   (先回归 MiniLM 嵌入,再端到端)——非平稳性问题被冻结钉死;
2. EVS + Conv3D 意味着**静止游戏画面的增量 prefill 极便宜**——指导 tick 的实际成本
   随画面变化率自适应,利于 0.5s 级节拍;
3. 12.5 token/s 的音频通道对游戏音效(脚步/枪声/提示音)是免费的额外传感器,
   我们的 gaming500 数据没有音轨,但真实游戏闭环(路线 Phase 3+)有;
4. 精确读点层号/头数以 HF config 为准(报告未披露逐层细节),接线前先 dump config 核对。

## 6. Agentic / computer-use 能力(2026-07 核实,补 §3 遗漏)

§3 只写了"理解"导向的训练课程,漏了关键一段:**Omni 是真 computer-use agent,不是只看懂**
(据 HF model card `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-*` + NVIDIA 技术博客):
- **Computer Use(OSWorld)= 47.4%**(前代 11.1%,+76.58%)——开放桌面 GUI 操作基准;
- **结构化工具调用**:Qwen3 兼容解析器(`qwen3_coder` + auto-tool-choice,OpenAI 兼容 API);
- **视觉 grounding** 是其 RL 后训练 5 任务之一(grounding/图表文档/STEM/视频/ASR);
- 官方用例明列 browser agent / email agent / GUI 自动化。
关键读数是 OSWorld **47.4%**:开放桌面上不到一半——**MC GUI 窄(布局固定)会更高,但非 100%**,
而合成是确定性(错一格即废)。

### 6.1 GUI 操纵:两条路径并存(非二选一),按熟练度分工

只有慢塔能**读** GUI(像素→槽位语义);**操纵**走两条路:
- **路径 A(慢塔直吐操作序列)**:慢塔用自身 computer-use 能力读屏+grounding+直接吐操作。
  用于**陌生/精细**的 GUI(没见过的容器、需现场定位)。
- **路径 B(慢塔告知→快塔执行)**:慢塔给操作意图,快塔作**肌肉记忆**执行。熟练例程
  (如"合成木板")在固定布局上是**记忆化动作序列,快塔执行不需 GUI 感知**(布局是常量,
  执行阶段无需 GUI 感知)——快、省慢塔算力。二者是同一技能的"深思"档与"直觉"档。

### 6.2 反拐杖原则(不把结构化 MC 工具做成承重架构)

不把"MC GUI 结构化工具 + snap-to-网格"当永久架构:**硬编码的固定工具会把能力
上限固定在工具设计上,模型/快塔学不出真实的 GUI 操纵**。Omni 已有 computer-use
基座,目标是**域内微调让它(路径A)/快塔(路径B)学会操纵**;任何确定性 snap/纠偏只作
**可移除的训练期辅助**(DAgger 式纠偏 / 声明脚手架),部署期撤除——与 raycast 闩锁
"A1 收口时移除"同纪律。

## 7. 现状核对:地图结构未接入实训模型(2026-07-09)

自我中心地图(EgoMapNorthLoc,`net/fovea_twotower/ego_map.py`)目前**只有模块 + 探针**
(`map_probe.py` / `map_loc_probe.py`)+ 装配测试(`assembly_a1.py`),**未接入实际在训的
快塔**:TrackNavTower 输入仍是 `(tokens, goal, prev_action)`,GRPO worker 只喂
`goal_relative(tokens)`,**都不含地图**。地图在 m3 阶段被探到地形部署上限 ~0.401 后转二期
(见记忆 fovea-m3-ego-map-design),接线是二期待办,非现状。

### 7.1 地图效果核对(2026-07-09):孤立探针有效,端任务未转化

- **孤立探针有效**:自定位(map_loc_probe,模拟)末端误差 4.7→**1.53** 且有界
  (G-loc1/G-loc2 PASS);远场特征保持(map_probe)8-16m 朴素 0.365 vs 北锚定 0.859、
  32-64m clip 0.594——旋转不变+近精远粗成立。
- **端任务未转化**:采木 approach 无重训消融(map_approach_ablation.py)map 1/6 vs
  nomap 0/6 = 噪声+混淆(唯一成功那局 mapsteer=0,非地图所致)。探针亦暴露主因:
  **nearest 方位准确率仅 0.62**(G-query FAIL,门 ≥0.8)→ 拿地图方位驱动导航不可靠。
- **三点卡口**:①方位查询精度弱(0.62,需真实感知域内标定);②地图未接进在训策略、
  未在真实感知上标定(探针用干净模拟地标);③approach 瓶颈本也不全在地图
  (走不到 + 准星压不住细树干)。**结论:地图现阶段不是导航的即时杠杆**,要兑现需先
  拉高方位查询精度再接进 GRPO 带位姿重训,非 no-retrain 脚手架能给。
