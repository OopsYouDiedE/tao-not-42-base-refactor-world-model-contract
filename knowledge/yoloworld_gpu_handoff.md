# YOLO-World-Dreamer · GPU 接力手册(给下一会话)

> 本文是跨会话交接 runbook:GPU 怎么起、怎么调、盯哪些信号、下一步做什么。
> 设计与数学见 [yoloworld.md](yoloworld.md);代码在 `net/yoloworld/` 与 `train/crafter/`。
> 分支 `feat/yoloworld`。

## 0. 现状(CPU 阶段结论)

- **结构全通**:世界模型线 + 256 候选小头 + 双头行为线,前向/反向/递归 policy/slot 多样性
  集成测试全过(`tests/integration/test_yoloworld_build.py`)。
- **一次设计返工已完成**:长跑暴露"选择线无信号 + 策略坍缩",已按 DreamerV3 口径重写——
  **主信号 = λ-return 策略梯度 + RewardEMA 归一 + 熵**,`cls/align/div/load` 降为小权重二级。
- **CPU 验证(tiny,~20k 步)**:`ent` 稳在 2.79 **不坍缩**、`ach/ep` 稳在 ~2.2、`ep_rew` ~1.3
  **不再退化**(旧版同期一路崩到 0.3 / 1.0)。但**奖励未抬升**:策略近均匀随机(`ent≈log17`),
  因塑形回报小 + 世界模型没训够 → 优势≈噪声。**这是尺度问题**:CPU tiny 只能证明"不崩",
  证明不了"会学"。出分必须 GPU + `crafter` 预设 + ~10⁶ 步。

## 1. GPU 启动

```bash
# 标准训练(crafter 预设 = DreamerV3 已验证容量 ~1.7e7 参数:deter512/32×32/units512/K256/H16)
python -m train.crafter.train_yoloworld \
    --size crafter --device cuda --total-steps 1000000 \
    --n-envs 16 --batch-size 32 --seq-len 32 --n-start 0 \
    --her-ratio 0.5 --updates-per 1 --task-encoder minilm \
    --run-dir runs/crafter_yoloworld
```

- `--n-start 0`:行为线用**全部 B·L 起点**(= batch×seq),GPU 上最大化批维利用率。
- 结构超参(K/H/n_rollout/各损失权重/熵)在 `net/yoloworld/config.py` 与
  `train/crafter/train_yoloworld.py:SIZE_PRESETS`;只有 `--actor-entropy` 暴露成 CLI(最常调)。
- 训练入口已自动开 **TF32 + cudnn.benchmark**(`_enable_fast_math`);CPU 才设 `set_num_threads`。

## 2. GPU 利用率 / 吞吐

- **重计算在 rollout**:`N·R·H` 次 `img_step`,`N=n_start(或 B·L)`、`R=n_rollout+n_explore`
  (crafter 预设 R=40)、`H=16`。已矢量化成 `[N·R]` 单批沿 H 一次 scan,批维越大 GPU 越满 →
  调大 `--batch-size`/`--n-envs`/不设 `--n-start`。
- **已知瓶颈:Crafter 环境是 CPU 顺序步进**(`VecCrafterEnv` python for 循环,无多进程)。
  采集阶段 GPU 会闲,更新阶段才打满。若 SPS 受限于采集而非更新,提高 `--updates-per`
  (每次采集多做几步更新,抬高 train-ratio)比加 GPU 更有效。异步/多进程环境是后续优化项(超出当前范围)。
- **AMP(可选,未默认开)**:可在两条线的前向外包 `torch.autocast('cuda', dtype=bfloat16)` 提吞吐。
  **注意数值不变量 I1–I8**(AGENTS §6):除法/normalize/exp/two-hot/Sinkhorn 等危险算子须留 fp32
  (`get_dist`/`DiscDist`/`reward_ema` 已是 fp32)。bf16 比 fp16 安全;开前先小步验证 loss 仍有限。

## 3. 调试清单

| 症状 | 处理 |
|---|---|
| **OOM** | 依次降:`--n-start`(设几百)→ `--batch-size` → `SIZE_PRESETS.crafter.n_rollout`/`n_explore` → `--seq-len` |
| **loss = NaN/Inf** | 已有 grad clip(WM 1000 / 行为 100)。查 `wm`/`actor`/`value` 哪项先炸;`reward_ema` 尺度下界 1;若 KL 爆调 `kl_free`。AMP 开着先关 |
| **SPS 很低** | 多半是 Crafter 采集瓶颈(见 §2);确认 `--device cuda` 真生效(开头打印 device);`nvidia-smi` 看更新阶段是否打满 |
| **策略钉在最大熵(`ent≈log17≈2.83`)= 随机不学** | 优势太弱。调小 `--actor-entropy`(如 3e-3→1e-3→3e-4);或增 `--updates-per`、加 `plan_horizon` 让计划结果拉开差异;确认 WM 已训出 action-敏感的奖励 |
| **熵坍缩到 ~0 + `ach/ep` 下滑** | 优势在放大噪声。调大 `--actor-entropy`;别再引入 per-batch std 归一(已移除,RewardEMA 才是对的) |
| **`cls` 长期 = `log(R)` 不动** | 预期早期如此(二级信号,无害)。WM 把候选回报训出差异后应下行;若到中后期(>200k 步)仍不动,说明候选回报仍无差,考虑加 `plan_horizon` 或检查 `ψ`(成就头)是否对动作敏感 |

## 4. 该盯哪些信号(判定设计是否成立)

按重要性排序,期望轨迹:

1. **`ent` 全程在 (0, log17) 之间、随训练缓降但不归零** —— 既不随机也不坍缩 = 策略在学。
2. **`ach/ep` 与 `ep_rew` 中后期上行**(~10⁵ 步起) —— 真正"会玩"的硬指标。
   DreamerV3 原版 crafter 档 ~10⁶ 步到 reward≈14 / 成就数显著上升,作对照尺。
3. **`wm`/`img` 持续降、`ach`(成就头 BCE)降** —— 世界模型 + ψ 在学(ρ^g 的根基)。
4. **`ret_scale`(RewardEMA)稳定不发散**、`val` 收敛 —— critic 健康。
5. **`cls` 后期下行、`algn` 低位** —— 选择线"自然激活"(二级信号开始起作用,可控性/泛化的前提)。
6. **`Rbest`(候选最优回报)与 `ach/ep` 同向上行** —— 候选群体在变好。

> checkpoint 落 `runs/<run-dir>/checkpoints/`(`--save-interval`),`state_dict` 含 WM+小头+critic。
> 续训/评测加载:`build_yoloworld(...); agent.set_ach_embed(E); agent.load_state_dict(ckpt['model_state'])`。

## 5. 下一步(出能力之后)

把"泛化强于 DreamerV3"从主张变证据——**目标跟随评测协议**(待建,见 yoloworld.md):

1. **目标跟随率**:P(达成 g | 条件于 g) vs P(达成 g | 条件于随机 g')。前者显著高 = 目标条件生效。
2. **留出措辞**:训练用句集 A,测试用近义改写 B 的目标跟随率(DreamerV3 无此机制 → 可控性基线 0)。
3. **零样本切目标**:同模型评测中途换目标看行为是否随之变。

诚实边界:裸单任务 Crafter 分数上我们不一定赢 DreamerV3(多出的目标条件是额外复杂度);
我们的价值在**可控性 / 语言泛化**这条不同的轴。

## 6. 关键文件

```
net/yoloworld/config.py      结构超参(K/H/n_rollout/各损失权重/熵/温度)
net/yoloworld/world_model.py RSSM + 成就头 ψ(pos_weight BCE)
net/yoloworld/heads.py       256 query 稀疏头 + select_score 点乘
net/yoloworld/behavior.py    λ-return PG 主信号 + RewardEMA + 二级蒸馏(cls/align/div/load)
net/yoloworld/agent.py       装配 + 采集 policy(点乘选序列,环内无 rollout)
train/crafter/crafter_tasks.py    22 成就语言句 + E + 逐 env 目标采样
train/crafter/yoloworld_buffer.py 含成就/目标回放(HER 事后重标)
train/crafter/train_yoloworld.py  两线分优化器主程序
tests/integration/test_yoloworld_build.py  CPU 集成测试
colab_yoloworld.ipynb        CPU Colab(自检 + 冒烟 + GPU 扩展说明)
```
