# 当前系统结构的数学原理

> 性质：本文描述**当前在跑的结构**的数学契约与其可推导后果，不写失败史、不写决策时间线（那些进 git log 与别的专题）。
> SSOT：代码是唯一事实来源；本文只写宏观数学与物理含义。每个数字追到 `文件:行号` 或 `runs/*.json`，追不到的标"待核"。
> 单位约定：角度=度、时间=tick/帧、空间=格（block）。Minecraft 20 tick = 1 秒。
> 现状与路线：`net/pixel_tower.py` 的 ConvStem + FiLM 是**路线 1**，是目前唯一在跑的快塔实现；
> 快塔输入契约的目标形态是**路线 2**（冻结 YOLOE-26 pf 提案 + 类别无关嵌入），本文把它写成待实现的接口契约（§2、§6.6）。

---

## 1. 系统与两个时间尺度

两塔异步，锁步在 CraftGround 的 tick 上（`train/craftground/grpo_pixel.py:60`，`SLOW_EVERY=20`）：

- **CraftGround 锁步**：1 `env.step()` = 1 tick（`grpo_pixel.py:251`）。观测 `obs["rgb"]` 是 `[360,640,3] uint8`，
  下采样到快塔输入 `[90,160,3]`（`grpo_pixel.py:234`，`IMG_HW=(90,160)`）。
- **快塔 20 Hz**：每 tick 前向一次，输出该 tick 动作（`grpo_pixel.py:226-251` 的逐 tick 循环）。
- **慢塔 1 Hz**：每 20 tick 刷新一次（`grpo_pixel.py:228`，`t % SLOW_EVERY == 0`）；20 tick = 1 秒 = 1 Hz。
  慢塔 = Omni（NVFP4，本地 vLLM，`grpo_pixel.py:97-139`，`MODEL="nemotron_3_nano_omni"`）。

### 1.1 慢塔输出与 goal 通道构造

慢塔读一帧 RGB，输出一行 JSON（`grpo_pixel.py:71-82` 的 `SLOW_SYSTEM`）：

```
{"subgoal": "<=6 词祈使句>", "aim": [X, Y]}
```

- `subgoal`：截断到 40 字符（`grpo_pixel.py:127`）。
- `aim`：归一化图像坐标 `[X,Y] ∈ [0,1000]²`，(0,0)=左上，(1000,1000)=右下，中心 (500,500)（`grpo_pixel.py:78-79`）；
  越界 clip 到 `[0,1000]`（`grpo_pixel.py:128`），慢塔失灵时降级为 `""` + `[500,500]`（`grpo_pixel.py:129-131`）。

goal 向量按下式拼成（`grpo_pixel.py:135-139`）：

```
goal = [ MiniLM(subgoal)  (384 维, L2 归一句向量) ,  aim[0]/1000 , aim[1]/1000 ]
```

- `MiniLM` = `sentence-transformers/all-MiniLM-L6-v2`（`grpo_pixel.py:328`），`normalize_embeddings=True`（`grpo_pixel.py:329`）
  ⇒ 384 维、L2 范数为 1、`float32`。
- 拼接后 goal 的 **Shape = [386], Dtype = float32**（384 语义 ⊕ 2 归一化 aim；`grpo_pixel.py:137-138`）。
- 实例化时 `goal_dim = 384 + 2 = 386`（`grpo_pixel.py:331`）覆盖 `PixelTowerConfig.goal_dim` 默认 384（`net/pixel_tower.py:53`）。

**零阶保持**：goal 在两次慢塔输出之间保持不变，跨 20 tick（`grpo_pixel.py:228-229`，只有 `t % 20 == 0` 才重算 `goal`）。

### 1.2 goal 换基到 512d 的待实现契约（路线 2）

路线 1 下 goal（386 维）经 FiLM 注入快塔（§6.1）。路线 2 要求 goal 与提案单位嵌入 `e_j`（512 维，见 §2）
**同空间**，作 query 对 N 个提案槽做 cross-attention 学"该看谁"。因此需新增一个投影：

```
goal_512 = W_g · goal_386        W_g: [386 → 512]（待实现，当前代码不存在）
```

`e_j` 是 L2 单位向量（`net/fovea_twotower/yolo_unified.py:128`），故 `goal_512` 作 query 前也应 L2 归一，
使 `⟨goal_512, e_j⟩ ∈ [-1,1]` 与提案嵌入的余弦几何一致（`yolo_unified.py:18-20` 的余弦口径）。这是接口契约，非现行代码。

---

## 2. YOLOE 打分数学与"cls 是信息塌缩"

本节数学从 `knowledge/design_fovea_yolo_fasttower.md §4.5` 与 `net/fovea_twotower/yolo_unified.py` 抄准。

### 2.1 native 打分式

YOLOE 的 promptable 对比头（BNContrastiveHead）逐锚打分（`yolo_unified.py:17`，`design_fovea_yolo_fasttower.md:114`）：

```
score_i = BN_i(cv3_i(feat)) · L2norm(reprta(pe)) × exp(logit_scale_i) + bias_i
```

- `BN_i(cv3_i(feat))`：第 i 尺度嵌入图（post-BN），逐锚 512 维特征。
- `L2norm(reprta(pe))`：类原型向量（= 类的"prompt 向量"），单位化。
- `exp(logit_scale_i)`：逐尺度温度（正标量），`bias_i`：逐尺度偏置。
- 数学形态：**512 维特征与类原型的一次内积，乘温度、加偏置**。

`set_classes` 存入的向量**原样进 cv4**，`reprta` 只作用于推理期新传的 tpe，不再套在库向量上
（`yolo_unified.py:77-78`）。本模块的可比口径用余弦：`ê = BN(emb)/‖·‖`，`score_cos = ê · bank^T ∈ [-1,1]`
（`yolo_unified.py:18-20`），跨 P3/P4/P5 三尺度可比。

**取嵌入的陷阱**：`predict()` 会把文本向量融进卷积（`is_fused`），融合后 cv3 尾层变类卷积、嵌入通道消失。
取嵌入必须直调底层 nn 前向、不触发融合，模块内有护栏 `assert not self.head.is_fused`（`yolo_unified.py:86-92`）。

### 2.2 为何 cls / n_cls 是有损压缩

一个提案的单位嵌入 `e_j`（512 维，`yolo_unified.py:104-128` 的 `proposal_embed`：掩膜内逐格单位嵌入均值再归一）
携带的是该区域的完整纹理语义。`cls`（类别分数）是把它投到 C 个类原型上：

```
cls_j = softmax 或 argmax_c ( ⟨e_j, prototype_c⟩ × exp(scale) + bias_c )   c ∈ {1..C}
```

即 `512 维 → C 维`的一次线性投影再取 max/argmax。信息论上这是**有损压缩**：512 维只保留在 C 个原型方向上的投影，
其余（512−C）维正交补被丢弃。当 `C=4`（木材词表）时，保留的世界结构约等于零。

**实测依据**（`docs/activity_log.md:456-466`，`tests/probe_yoloe_coverage.py`，8 帧黑橡木森林，yoloe-11l-seg{,-pf}）：

| 通路 | 平均提案/帧 | 像素覆盖 | 准星被覆盖的帧 |
|---|---|---|---|
| pf（无词表，`max_det=256`） | 48.8 | 90.8% | 88% |
| pf 截断 top-K=8 | 8 | 87.8% | 62% |
| prompt（4 类词表，路线 1 现行） | 0.5 | 1.3% | 0% |

两个独立瓶颈：(a) 词表瓶颈（主）——4 类打分把观测塌成"命名过的东西"，森林里每帧 0.5 框、覆盖 1.3% 像素、准星前方块 0% 被看见；
(b) top-K=8 截断（次）——准星覆盖 88%→62%。（限定：prompt 臂用原始文本 PE，真实系统会用域内校准原型，故 1.3% 是**下界**；`activity_log.md:474-475`。）
（待核：上述覆盖实测在 yoloe-**11l**-seg 上跑，路线 2 契约指定 YOLOE-**26**；结构性结论不因骨干版本改变，但绝对数字未在 26l 上复测。）

### 2.3 路线 2 的提案 token（删 cls / 删 n_cls）

结论：删掉 `cls / n_cls` 这一步塌缩，直接把 512 维嵌入喂给快塔，让快塔自行用 goal 做 cross-attention 挑选。
路线 2 单个提案 token（**518 维**）：

```
token_j = [ e_j (512 维单位嵌入) , cx , cy , w , h , conf , area ]        (512 + 6 = 518)
```

- `e_j`：`yolo_unified.py:104-128` 的 `proposal_embed` 输出，`[512] float32`，L2 单位向量。
- 几何 6 维 `[cx,cy,w,h,conf,area]`：`yolo_unified.py:161-165`，均已归一（cx,cy,w,h ∈ [0,1]，conf ∈ [0,1]，area = 框面积/全图面积）。
- **无 cls、无 n_cls**。当前 `yolo_unified.py:156,166` 的 `forward` 输出是 `[geo6 + cos(C)]`（几何 6 + C 类余弦分数），
  路线 2 是把 `cos(C)` 替换成 `e_j(512)`。
- 提案数：路线 2 目标 ≤256 个类别无关提案（对齐 §2.2 表中 `max_det=256`）；当前 `yolo_unified.py:53` 默认 `max_det=64`（待改）。

---

## 3. 相机动作头：mu-law 分箱

相机头输出 `cam_logits`，**Shape = [B,T,k,n_mouse,camera_bins]**，`n_mouse=2`（yaw, pitch），
`camera_bins=11`，`k=chunk_k=1`（`net/pixel_tower.py:123,136`；`grpo_pixel.py:68` `CAM_BINS=11`）。

### 3.1 bin → 度 的解码

采样得 bin 索引后，解码到度（`grpo_pixel.py:205-210` `bins_to_deg`）：

```
x = (bin / (CAM_BINS-1)) · 2 − 1            # bin 线性铺到 [-1,1]
v = sign(x) · ((1+μ)^|x| − 1) / μ           # mu-law 扩张, μ = 8
deg = v · CAM_MAX_DEG                        # CAM_MAX_DEG = 18°/tick
```

- `μ = 8.0`（`grpo_pixel.py:208`），`CAM_MAX_DEG = 18.0`（`grpo_pixel.py:69`，每 tick 相机增量上限，单位=度/tick）。
- 11 个 bin 为奇数 ⇒ 中心 bin（index 5）恰对应 `x=0 → v=0 → deg=0`（无转动）。
- mu-law 扩张使刻度**非均匀**：`|x|` 小处 `v` 增长慢（小角度分辨率高），`|x|` 大处指数增长（大角度粗量化）。
  这正是把"小角度精细、大转身粗略"编码进 bin 结构。
- 单位链：bin（无量纲）→ `v ∈ [-1,1]`（无量纲）→ `deg ∈ [-18,18]`（度/tick），写入 `a["camera_yaw"], a["camera_pitch"]`（`grpo_pixel.py:247`）。

### 3.2 为何分箱 + CE 而非回归

论证的原始出处在 `train/minecraft/vpt_action.py:10-13`：

> 相机 dx/dy 分布 = 尖峰在 0 + 重尾大转身（零均值重尾）。MSE 回归下"恒预测 0"是近似最优的**平凡解**
> （对零均值分布，常数 0 使 MSE 逼近方差下界，学不到方向）。改成 mu-law 分箱分类后，0 只是 11 个类之一，
> 基率解的 CE 等于边缘熵，任何真实信号都能压过它。

VPT 原版同样用 mu-law 离散相机（`vpt_action.py:13`）。故快塔用 `Categorical` 采样 bin + CE 监督（`grpo_pixel.py:242,291`），
不做连续回归。

---

## 4. 按键头

按键头输出 `key_logits`，**Shape = [B,T,k,n_keys]**，`n_keys=20`，`k=1`（`net/pixel_tower.py:124,137`）。
20 个键（`grpo_pixel.py:65-67` `V2_KEYS`，CraftGround V2 口径）：

```
forward, back, left, right, jump, sneak, sprint, attack, use, drop, inventory,
hotbar.1, hotbar.2, hotbar.3, hotbar.4, hotbar.5, hotbar.6, hotbar.7, hotbar.8, hotbar.9
```

每键各自独立 Bernoulli：采样 `key ~ Bernoulli(sigmoid(logit))`（`grpo_pixel.py:243`），监督用逐键 BCE（`grpo_pixel.py:292`）。
按下的键写入动作字典 `a[k]=True`（`grpo_pixel.py:248-250`）。

### 4.1 随机初始化的数学后果与先验注入

`key_head = nn.Linear(d, k·n_keys)`（`net/pixel_tower.py:124`），默认 `nn.Linear` bias ≈ 0，故 `sigmoid(0)=0.5`。
后果：随机初始化下每 tick 期望按下 `20 × 0.5 = 10` 个键（背包 `inventory` 反复开合、`hotbar.1..9` 全按、瘫痪式乱按）。

**修法（先验注入，非启发式补丁）**：把 `key_head.bias` 初始化到人类按键率的 logit：

```
bias_k = logit(p_k) = ln( p_k / (1 − p_k) )
```

若取人类按键率 `p ≈ 0.05`，则 `bias ≈ ln(0.05/0.95) ≈ −2.94`，起点期望按下 `20 × 0.05 = 1` 个键。
这是把"人类多数 tick 只按少数键"的先验写进初始分布，等价于给 REINFORCE 一个信息量正确的起点，
而非在采样后手工屏蔽。（待核：当前 `net/pixel_tower.py` **未**做此初始化，`key_head` 用 `nn.Linear` 默认 bias；这是待注入的先验。）

---

## 5. 训练目标：组内相对优势的 REINFORCE

### 5.1 损失与梯度推导

每条 rollout 的更新损失（`grpo_pixel.py:291-293`）：

```
loss = adv · ( CE(cam_logits, 采样到的 bin) + BCE(key_logits, 采样到的 key) )
```

- CE 打在**采样到的**类别上，恒等于 `−log π_cam(a)`（softmax 交叉熵在 one-hot 目标上的定义）。
- BCE 打在**采样到的**二值 key 上，恒等于 `−log π_key(a)`（逐键 Bernoulli 对数似然之和取负）。
- 故 `CE + BCE = −log π(a)`，`loss = −adv · log π(a)`，
  `∇loss = −adv · ∇log π(a)`。最小化 loss ⇔ 最大化 `adv · log π(a)` ——这是 REINFORCE（`grpo_pixel.py:273-277` 的推导注释一致）。

### 5.2 优势的构造

判官（Haiku）读联络表图 + 行为统计文本，从好到差排名（`grpo_pixel.py:175-200`，rubric `grpo_pixel.py:85-92`）：

```
名次 rank_j （1=最好）  →  score_j = −rank_j        （名次取负当分数，grpo_pixel.py:195）
adv = (s − mean(s)) / max(std(s), 1e-3)             （group_advantage，train/fovea_twotower/grpo_harness.py:52-55）
```

std 地板 `1e-3`（`grpo_harness.py:52` `std_floor=1e-3`）。判官两轮解析失败时回退里程碑机器分 `len(inv_events)`（`grpo_pixel.py:197-199`），并记 `fallback=True`。

### 5.3 该优势的两条数学性质

- **(a) 只携带序数信息**：`adv` 由名次 `{−1,−2,...,−n}` 经 z 归一得到。对固定组大小 n，无论 n 条轨迹真实差距是大是小，
  名次集合恒为 `{−1,−2,...,−n}`，z 归一后 `adv` 是**同一组固定数**。举例 n=4 时 `adv` 恒为对称的四个值（与轨迹间真实差距无关）。
  优势的取值分布与轨迹质量的绝对差无关，只反映排序。
- **(b) 判官在噪声上强行排序 ⇒ 梯度是纯噪声而非零**：因名次必然产生非零方差（`adv_var > 0` 恒成立，除非判官判全并列），
  即使判官在无信息的噪声上随机排序，`adv` 仍非零、梯度仍非零。因此 **`adv_var > 0` 不构成"学到东西"的证据**
  （`grpo_pixel.py:364` 记录 `adv_var`，其为正是结构必然，非训练进展）。

### 5.4 明确：这不是完整 GRPO

当前只有"组内基线"这一件。**缺**：

- importance ratio `π_new(a)/π_old(a)`；
- PPO clip（`clip(ratio, 1±ε)`）；
- KL to reference（对参考策略的散度约束）。

`grpo_harness.py:69` 的 `grpo_update` 骨架带 `clip=0.2` 参数但未实现 ratio；实际在跑的 `grpo_pixel.py:272-298` 的 `update` 无 ratio、无 clip、无 KL。
**数学后果**：若一批数据要做多次梯度步（§6 指出当前每组做 24 次），采样分布 `π_old` 与被更新分布 `π_new` 已偏移，
`−adv·log π_new(a)` 不再是无偏 policy-gradient 估计；无 ratio 修正时 off-policy 偏差无界。

---

## 6. 当前实现里让 `log π(a)` 算错的数学不一致

> ✅ **2026-07-10 已全部修复**(commit 见 git log;验收单测 `tests/unit/test_grpo_pixel_fixes.py`):
> §6.1 双侧 T=1+帧堆叠 S=4;§6.2 eval 采样+dropout=0;§6.3 损失同除温度;
> §6.4 goal 逐 tick 落盘(含 aim)回放;§6.5 组内累积单步(严格 on-policy,尾段不丢);
> §4.1 按键先验 bias=logit(0.05) 已注入(`net/pixel_tower.py` `key_prior`)。
> 本节原文保留:它是"为什么必须这么修"的数学论证。

REINFORCE 的正确性要求**采样时的 π** 与**求梯度时的 π** 是同一个分布。以下各处独立破坏了这一点。

### 6.1 序列长度：采样 T=1，更新 T=seq

- 采样：`img` 构造为 `[1,1,3,90,160]`（`grpo_pixel.py:236`，`[None,None]` ⇒ B=1,T=1），前向 `tower(img, goal[None], pv)`（`grpo_pixel.py:239`）。
  T=1 时位置编码只取第 0 槽 `pos[:, :1]`（`net/pixel_tower.py:131`），因果注意力对单 token 是恒等（无历史可看）。
  ⇒ 采样时快塔实际是**无时序上下文的单帧策略**。
- 更新：`img` 构造为 `[1,seq,...]`（`grpo_pixel.py:286`，`seq` 默认 64，`grpo_pixel.py:307`；smoke=32，`grpo_pixel.py:317`），
  前向吃 T=seq，位置编码取 `pos[:, :64]`、因果注意力对 64 步生效。
- ⇒ 采样时的 `π`（单帧、pos[0]）≠ 更新时的 `π`（64 步序列、pos[0..63]）。同一 tick 的动作在两个不同条件分布下被算概率。

### 6.2 dropout：采样时开着，且两次 mask 不同

- 模型建成后全程无 `.eval()`（`grpo_pixel.py:333` 建成即 `.to(device)`，无 `.eval()`）；`dropout=0.1`（`net/pixel_tower.py:52`）在 train 模式下生效。
- `torch.no_grad()`（`grpo_pixel.py:238`）只关梯度、**不关 dropout**。故采样时 dropout 以 0.1 概率随机置零激活。
- 更新时同样在 train 模式（dropout 开），但那是**另一次独立的随机 mask**。
- ⇒ 采样 `π`（dropout mask A）≠ 更新 `π`（dropout mask B）。`log π(a)` 算在与采样不同的网络实例上。

### 6.3 温度：采样除 1.3，梯度打在未除温度的 logits 上

- 采样：`cam_l = cam_l[0,-1,0] / temp`、`key_l = key_l[0,-1,0] / temp`（`grpo_pixel.py:240-241`，`temp=1.3`，`grpo_pixel.py:309`），
  即从 `π_T(a) ∝ softmax(logit/1.3)` 采样。
- 更新：CE/BCE 打在**未除温度**的 `cam_l/key_l` 上（`grpo_pixel.py:291-292`），即对 `π_1(a) ∝ softmax(logit)` 求梯度。
- ⇒ 采样自 `π_{1.3}`，却最大化 `log π_1(a)`。温度改变分布形状，两者不是同一 `π`。

### 6.4 goal 重放：整条 400 步轨迹用单一 goal_last

- rollout 中 goal 每 20 tick 换一次（§1.1）。但采集侧**只存了 `goal_last`**（`grpo_pixel.py:269`，= 最后一 tick 的 goal），
  未逐 tick 保存 goal。
- 更新时整条轨迹用 `goal = r["goal_last"]`（`grpo_pixel.py:283`）重放全部 seq 段。
- ⇒ 前 380 tick 的动作在采样时的条件 goal ≠ 更新时喂的 `goal_last`。这不仅是重放选择错误，更是**采集侧数据缺口**：
  逐 tick goal 未落盘，更新即使想对齐也无数据。（用户转述标 `grpo_pixel.py:285`，实际赋值在 `:283`。）

### 6.5 一组 24 次 opt.step 复用同一批 adv

- `update` 对每条 roll 按 `seq` 切段：`T=400, seq=64` ⇒ `range(0, 336, 64)` = 6 段/roll（`grpo_pixel.py:284`）；
  `per_group=4`（`grpo_pixel.py:305`）⇒ 一组 4×6 = 24 次 `opt.step()`（`grpo_pixel.py:296`，扣除 `|adv|<1e-6` 被跳过的，`grpo_pixel.py:280`）。
- 24 次梯度步复用同一批 `adv`（判官对该组的一次排序），且无 §5.4 的 ratio 修正。
- ⇒ 首步之后网络已偏离采样分布，后续 23 步是无 ratio 的 off-policy 更新，偏差无界（承 §5.4）。
  （附：`range(0,336,64)` 末段起点 320、切到 383，尾部 tick 384-399 未进任何段。）

### 6.6 路线 1 vs 路线 2 的条件注入形态

当前（路线 1）goal 经 **FiLM** 注入（`net/pixel_tower.py:130`）：

```
x = x · (1 + goal_q(goal)) + goal_bias(goal)          # 逐通道仿射调制
```

`goal_q, goal_bias: [386 → 256]`（`net/pixel_tower.py:113-114`，`d=256`）。这是全局逐通道仿射，非按提案选择。
路线 2 要求 goal 换基到 512d、作 cross-attention query 对 N 个提案槽选择（§1.2、§2.3），是不同的条件机制，待实现。

---

## 附：读代码时发现的、与转述不符处（以代码为准）

1. **mu-law 的 μ 在两文件不一致**：相机头 `grpo_pixel.py:208` 用 `μ=8.0`（转述"μ=8"对，与在跑代码一致）；
   但被引为"11-bin 相机头口径来源"的 `train/minecraft/vpt_action.py:30` 用 `CAMERA_MU=10.0`。
   两者只共享 **bin 结构**（11 bin、mu-law、奇数中心=0）与"分箱+CE 而非回归"的论证，**μ 常数不共享**。

2. **相机的物理量纲也不同**：`grpo_pixel.py` 工作在**度**（`bins_to_deg` 末乘 `CAM_MAX_DEG=18°/tick`，`:210,:69`）；
   `vpt_action.py` 工作在**归一化像素**（`CAMERA_SCALE=10.0` px/帧，`:28`，`bin_to_camera` 不乘角度）。
   只有 bin 编码方式迁移，物理标定（度 vs 像素）不迁移。

3. **goal_last 赋值行号**：转述标 `grpo_pixel.py:285`，实际在 `:283`。更关键的是采集侧**根本未逐 tick 存 goal**
   （只 `:269` 存 `goal_last`），§6.4 的重放错误因此是数据缺口而非可简单修正的重放选择。

4. **live 运行的 goal_dim = 386 非 384**：`grpo_pixel.py:331` 用 `goal_dim=384+2` 覆盖 `PixelTowerConfig.goal_dim` 默认 384
   （`net/pixel_tower.py:53`）。`pixel_tower.py` docstring 写"MiniLM=384"只指 subgoal 语义分量，未含 2 维 aim。

5. **按键先验尚未落地**：§4.1 的 `key_head.bias ← logit(p)` 是**待注入**的先验；当前 `net/pixel_tower.py:124` 的 `key_head`
   用 `nn.Linear` 默认 bias（≈0，`sigmoid(0)=0.5`），期望每 tick 按下 10 键的问题现存。

6. **键命名两套**：在跑的快塔用 `V2_KEYS`（CraftGround 名：`forward/back/.../hotbar.1..9`，`grpo_pixel.py:65-67`，与转述一致）；
   `vpt_action.py:19-24` 的 `VPT_KEYS` 是另一套（`key_w/key_a/...`）。两者都 20 键，但命名与 attack/use 的语义位置不同
   （craftground 里 attack/use 是 20 键中的布尔位，非鼠标按钮）。

7. **覆盖率实测骨干是 11l 非 26l**：§2.2 的 48.8/90.8%/88% 与 0.5/1.3%/0%（`docs/activity_log.md:456-461`）在 **yoloe-11l-seg** 上跑；
   路线 2 契约指定 **YOLOE-26**。结构性结论（词表塌缩）不因版本改变，但绝对数字未在 26l 复测（标"待核"）。

8. **max_det 当前 64 非 256**：路线 2 目标 ≤256 提案；`yolo_unified.py:53` 默认 `max_det=64`。§2.2 表中 pf 全量行的 `max_det=256`
   是覆盖率探针的设置（`activity_log.md:459`），非 `yolo_unified` 现行默认。

9. **`grpo_harness.py` 的 group_advantage 已实证于假 rollout**：`--smoke`（`grpo_harness.py:92-119`）用 Step0 摘要合成，
   与 `grpo_pixel.judge` 复用同一 `group_advantage`（`grpo_pixel.py:57` import）。§5.4 的 clip/ratio 缺失在两处 `update` 均成立。
