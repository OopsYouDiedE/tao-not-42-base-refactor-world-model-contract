# MineStudio 轨迹训练题生成报告

> 本报告由出题流程自动生成。图片与参考动作来自真实 MineStudio 轨迹。
> 参考轨迹是一种人类示范，不是唯一正确答案。`answer_key.jsonl` 不应交给做题模型。

## 汇总

| 项目 | 数量 |
|---|---:|
| 候选题目 | 400 |
| 结构审核完成 | 0 |
| 结构审核通过 | 0 |

## demonstration_optimization_000000

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player593-f153ac423f61-20211115-143233` |
| 图片帧 | `[4591, 4595, 4599, 4603]` |
| 目标动作区间 | `[4591, 4607]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4591**

![demonstration_optimization_000000 frame 4591](images/demonstration_optimization_000000_00.jpg)

**图 2，帧 4595**

![demonstration_optimization_000000 frame 4595](images/demonstration_optimization_000000_01.jpg)

**图 3，帧 4599**

![demonstration_optimization_000000 frame 4599](images/demonstration_optimization_000000_02.jpg)

**图 4，帧 4603**

![demonstration_optimization_000000 frame 4603](images/demonstration_optimization_000000_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```
动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse 1 4 MouseLeft ; Mouse 0 1 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 0 MouseLeft ; MouseLeft ; MouseLeft ; Mouse 20 8 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse 1 4 MouseLeft ; Mouse 0 1 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 0 MouseLeft ; MouseLeft ; MouseLeft ; Mouse 20 8 MouseLeft <|action_end|>
```

## demonstration_optimization_000001

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `pokey-cyan-spitz-f153ac423f61-20220112-131429` |
| 图片帧 | `[3662, 3666, 3670, 3674]` |
| 目标动作区间 | `[3662, 3678]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3662**

![demonstration_optimization_000001 frame 3662](images/demonstration_optimization_000001_00.jpg)

**图 2，帧 3666**

![demonstration_optimization_000001 frame 3666](images/demonstration_optimization_000001_01.jpg)

**图 3，帧 3670**

![demonstration_optimization_000001 frame 3670](images/demonstration_optimization_000001_02.jpg)

**图 4，帧 3674**

![demonstration_optimization_000001 frame 3674](images/demonstration_optimization_000001_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -70 14 W ; Mouse -28 12 ; Mouse -53 14 ; Mouse -2 0 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 MouseRight ; MouseRight ; MouseRight ; Mouse -20 -14 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -34 -22 ; Mouse -19 -13 ; Mouse -38 -20 ; Mouse -9 -6 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -7 -2 ; Mouse -2 -1 ; Mouse -1 0 ; MouseRight <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -70 14 W ; Mouse -28 12 ; Mouse -53 14 ; Mouse -2 0 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 MouseRight ; MouseRight ; MouseRight ; Mouse -20 -14 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -34 -22 ; Mouse -19 -13 ; Mouse -38 -20 ; Mouse -9 -6 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -7 -2 ; Mouse -2 -1 ; Mouse -1 0 ; MouseRight <|action_end|>
```

## demonstration_optimization_000002

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220304-203855` |
| 图片帧 | `[17430, 17434, 17438, 17442]` |
| 目标动作区间 | `[17430, 17446]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 17430**

![demonstration_optimization_000002 frame 17430](images/demonstration_optimization_000002_00.jpg)

**图 2，帧 17434**

![demonstration_optimization_000002 frame 17434](images/demonstration_optimization_000002_01.jpg)

**图 3，帧 17438**

![demonstration_optimization_000002 frame 17438](images/demonstration_optimization_000002_02.jpg)

**图 4，帧 17442**

![demonstration_optimization_000002 frame 17442](images/demonstration_optimization_000002_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 -2 ; Mouse -1 -1 ;  ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ;  ; Mouse -2 5 ; Mouse -10 20 ; Mouse -10 20 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -3 6 ;  ; Mouse -2 0 ; Mouse -3 0 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -4 2 ; Mouse -2 1 MouseLeft ;  ;  <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 -2 ; Mouse -1 -1 ;  ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ;  ; Mouse -2 5 ; Mouse -10 20 ; Mouse -10 20 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -3 6 ;  ; Mouse -2 0 ; Mouse -3 0 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -4 2 ; Mouse -2 1 MouseLeft ;  ;  <|action_end|>
```

## demonstration_optimization_000003

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player445-7b77b6e4e459-20211210-205506` |
| 图片帧 | `[3109, 3113, 3117, 3121]` |
| 目标动作区间 | `[3109, 3125]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3109**

![demonstration_optimization_000003 frame 3109](images/demonstration_optimization_000003_00.jpg)

**图 2，帧 3113**

![demonstration_optimization_000003 frame 3113](images/demonstration_optimization_000003_01.jpg)

**图 3，帧 3117**

![demonstration_optimization_000003 frame 3117](images/demonstration_optimization_000003_02.jpg)

**图 4，帧 3121**

![demonstration_optimization_000003 frame 3121](images/demonstration_optimization_000003_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000004

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-5d9ab504739c-20220213-022859` |
| 图片帧 | `[15940, 15944, 15948, 15952]` |
| 目标动作区间 | `[15940, 15956]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 15940**

![demonstration_optimization_000004 frame 15940](images/demonstration_optimization_000004_00.jpg)

**图 2，帧 15944**

![demonstration_optimization_000004 frame 15944](images/demonstration_optimization_000004_01.jpg)

**图 3，帧 15948**

![demonstration_optimization_000004 frame 15948](images/demonstration_optimization_000004_02.jpg)

**图 4，帧 15952**

![demonstration_optimization_000004 frame 15952](images/demonstration_optimization_000004_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 8 -10 D MouseRight ; Mouse 5 -6 D MouseRight ; Mouse 5 -4 D MouseRight ; Mouse 4 -2 D MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 0 D MouseRight ; D MouseRight ; MouseRight ; Mouse 2 11 A MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 2 4 A MouseRight ; Mouse 2 4 A MouseRight ; Mouse 0 6 S A MouseRight ; S A MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 2 S A MouseRight ; Mouse 1 1 A MouseRight ; A MouseRight ; A MouseRight <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 8 -10 D MouseRight ; Mouse 5 -6 D MouseRight ; Mouse 5 -4 D MouseRight ; Mouse 4 -2 D MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 0 D MouseRight ; D MouseRight ; MouseRight ; Mouse 2 11 A MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 2 4 A MouseRight ; Mouse 2 4 A MouseRight ; Mouse 0 6 S A MouseRight ; S A MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 2 S A MouseRight ; Mouse 1 1 A MouseRight ; A MouseRight ; A MouseRight <|action_end|>
```

## demonstration_optimization_000005

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220214-135504` |
| 图片帧 | `[732, 736, 740, 744]` |
| 目标动作区间 | `[732, 748]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 732**

![demonstration_optimization_000005 frame 732](images/demonstration_optimization_000005_00.jpg)

**图 2，帧 736**

![demonstration_optimization_000005 frame 736](images/demonstration_optimization_000005_01.jpg)

**图 3，帧 740**

![demonstration_optimization_000005 frame 740](images/demonstration_optimization_000005_02.jpg)

**图 4，帧 744**

![demonstration_optimization_000005 frame 744](images/demonstration_optimization_000005_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; Mouse 6 -3 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 8 -8 W space ctrl ; Mouse 1 -1 W space ctrl ; Mouse 0 -2 W space ctrl MouseRight ; W space ctrl MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl MouseRight ; Mouse -4 -1 W space ctrl ; Mouse -6 -1 W space ctrl ; Mouse -6 -3 W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -9 -1 W space ctrl ; Mouse -5 -1 W space ctrl ; Mouse -7 -1 W space ctrl ; Mouse -5 -2 W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; Mouse 6 -3 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 8 -8 W space ctrl ; Mouse 1 -1 W space ctrl ; Mouse 0 -2 W space ctrl MouseRight ; W space ctrl MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl MouseRight ; Mouse -4 -1 W space ctrl ; Mouse -6 -1 W space ctrl ; Mouse -6 -3 W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -9 -1 W space ctrl ; Mouse -5 -1 W space ctrl ; Mouse -7 -1 W space ctrl ; Mouse -5 -2 W space ctrl <|action_end|>
```

## demonstration_optimization_000006

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220220-094710` |
| 图片帧 | `[5811, 5815, 5819, 5823]` |
| 目标动作区间 | `[5811, 5827]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5811**

![demonstration_optimization_000006 frame 5811](images/demonstration_optimization_000006_00.jpg)

**图 2，帧 5815**

![demonstration_optimization_000006 frame 5815](images/demonstration_optimization_000006_01.jpg)

**图 3，帧 5819**

![demonstration_optimization_000006 frame 5819](images/demonstration_optimization_000006_02.jpg)

**图 4，帧 5823**

![demonstration_optimization_000006 frame 5823](images/demonstration_optimization_000006_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 20 13 ; Mouse 13 10 ; Mouse 7 2 ; Mouse 2 1 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 ; Mouse -50 1 ; Mouse -124 -23 ; Mouse -111 -36 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -21 -24 ; Mouse 6 -16 ; Mouse 12 -10 ; Mouse 14 -8 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 -1 ; Mouse -8 3 ; Mouse -35 5 ; Mouse -24 0 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 20 13 ; Mouse 13 10 ; Mouse 7 2 ; Mouse 2 1 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 ; Mouse -50 1 ; Mouse -124 -23 ; Mouse -111 -36 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -21 -24 ; Mouse 6 -16 ; Mouse 12 -10 ; Mouse 14 -8 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 -1 ; Mouse -8 3 ; Mouse -35 5 ; Mouse -24 0 <|action_end|>
```

## demonstration_optimization_000007

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `pokey-cyan-spitz-f153ac423f61-20220308-223507` |
| 图片帧 | `[7276, 7280, 7284, 7288]` |
| 目标动作区间 | `[7276, 7292]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7276**

![demonstration_optimization_000007 frame 7276](images/demonstration_optimization_000007_00.jpg)

**图 2，帧 7280**

![demonstration_optimization_000007 frame 7280](images/demonstration_optimization_000007_01.jpg)

**图 3，帧 7284**

![demonstration_optimization_000007 frame 7284](images/demonstration_optimization_000007_02.jpg)

**图 4，帧 7288**

![demonstration_optimization_000007 frame 7288](images/demonstration_optimization_000007_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseRight ; Mouse 4 -1 ; Mouse 8 -2 ;  <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 0 ; Mouse 2 -1 ;  ;  <|action_end|>
```

动作块 3：

```text
<|action_start|> ;  ;  ;  ; Mouse 0 3 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 24 ; Mouse 12 37 ; Mouse 2 16 ; Mouse 2 2 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ; Mouse 4 -1 ; Mouse 8 -2 ;  <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 0 ; Mouse 2 -1 ;  ;  <|action_end|>
```

动作块 3：

```text
<|action_start|> ;  ;  ;  ; Mouse 0 3 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 24 ; Mouse 12 37 ; Mouse 2 16 ; Mouse 2 2 <|action_end|>
```

## demonstration_optimization_000008

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220106-000238` |
| 图片帧 | `[9320, 9324, 9328, 9332]` |
| 目标动作区间 | `[9320, 9336]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9320**

![demonstration_optimization_000008 frame 9320](images/demonstration_optimization_000008_00.jpg)

**图 2，帧 9324**

![demonstration_optimization_000008 frame 9324](images/demonstration_optimization_000008_01.jpg)

**图 3，帧 9328**

![demonstration_optimization_000008 frame 9328](images/demonstration_optimization_000008_02.jpg)

**图 4，帧 9332**

![demonstration_optimization_000008 frame 9332](images/demonstration_optimization_000008_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 3 ; Mouse -10 8 ; Mouse -9 3 W ; Mouse -5 2 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 2 W space ; W space ; Mouse -9 0 W ; Mouse -19 6 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -10 3 W ; Mouse -18 3 W ; Mouse -11 5 W ; Mouse -5 0 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -12 3 W ; Mouse -30 3 W ; Mouse -71 9 W ; Mouse -44 6 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 3 ; Mouse -10 8 ; Mouse -9 3 W ; Mouse -5 2 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 2 W space ; W space ; Mouse -9 0 W ; Mouse -19 6 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -10 3 W ; Mouse -18 3 W ; Mouse -11 5 W ; Mouse -5 0 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -12 3 W ; Mouse -30 3 W ; Mouse -71 9 W ; Mouse -44 6 W <|action_end|>
```

## demonstration_optimization_000009

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220225-154320` |
| 图片帧 | `[2199, 2203, 2207, 2211]` |
| 目标动作区间 | `[2199, 2215]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2199**

![demonstration_optimization_000009 frame 2199](images/demonstration_optimization_000009_00.jpg)

**图 2，帧 2203**

![demonstration_optimization_000009 frame 2203](images/demonstration_optimization_000009_01.jpg)

**图 3，帧 2207**

![demonstration_optimization_000009 frame 2207](images/demonstration_optimization_000009_02.jpg)

**图 4，帧 2211**

![demonstration_optimization_000009 frame 2211](images/demonstration_optimization_000009_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ctrl MouseRight ; Mouse 2 0 W ctrl MouseRight ; Mouse 1 0 W ctrl MouseRight ; W ctrl MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 -2 W ctrl MouseRight ; Mouse 2 -3 W ctrl MouseRight ; W ctrl MouseRight ; Mouse 1 -1 W ctrl MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 2 0 W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl MouseRight ; Mouse 2 0 W ctrl MouseRight ; Mouse 1 0 W ctrl MouseRight ; W ctrl MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 -2 W ctrl MouseRight ; Mouse 2 -3 W ctrl MouseRight ; W ctrl MouseRight ; Mouse 1 -1 W ctrl MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 2 0 W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight ; W ctrl MouseRight <|action_end|>
```

## demonstration_optimization_000010

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-28b3f886dc8d-20220228-144641` |
| 图片帧 | `[15086, 15090, 15094, 15098]` |
| 目标动作区间 | `[15086, 15102]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 15086**

![demonstration_optimization_000010 frame 15086](images/demonstration_optimization_000010_00.jpg)

**图 2，帧 15090**

![demonstration_optimization_000010 frame 15090](images/demonstration_optimization_000010_01.jpg)

**图 3，帧 15094**

![demonstration_optimization_000010 frame 15094](images/demonstration_optimization_000010_02.jpg)

**图 4，帧 15098**

![demonstration_optimization_000010 frame 15098](images/demonstration_optimization_000010_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 173 -54 W ; Mouse 46 -73 W ; Mouse 11 -29 W ; Mouse 1 -1 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W A ; W A ; W A ; Mouse 42 -6 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 49 -7 W ; Mouse 13 -3 W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; Mouse 1 0 W ; Mouse 25 1 W ; Mouse 41 7 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 173 -54 W ; Mouse 46 -73 W ; Mouse 11 -29 W ; Mouse 1 -1 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W A ; W A ; W A ; Mouse 42 -6 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 49 -7 W ; Mouse 13 -3 W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; Mouse 1 0 W ; Mouse 25 1 W ; Mouse 41 7 W <|action_end|>
```

## demonstration_optimization_000011

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-9698955bae7d-20220122-034522` |
| 图片帧 | `[4425, 4429, 4433, 4437]` |
| 目标动作区间 | `[4425, 4441]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4425**

![demonstration_optimization_000011 frame 4425](images/demonstration_optimization_000011_00.jpg)

**图 2，帧 4429**

![demonstration_optimization_000011 frame 4429](images/demonstration_optimization_000011_01.jpg)

**图 3，帧 4433**

![demonstration_optimization_000011 frame 4433](images/demonstration_optimization_000011_02.jpg)

**图 4，帧 4437**

![demonstration_optimization_000011 frame 4437](images/demonstration_optimization_000011_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 3 1 W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; Mouse -2 8 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -9 25 W space ctrl ; Mouse -4 26 W space ctrl ; Mouse 0 9 W space ctrl ; Mouse 2 10 W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 10 27 W space ctrl ; Mouse 2 4 W space ctrl ; Mouse 69 42 W space ctrl ; Mouse 95 17 W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 3 1 W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; Mouse -2 8 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -9 25 W space ctrl ; Mouse -4 26 W space ctrl ; Mouse 0 9 W space ctrl ; Mouse 2 10 W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 10 27 W space ctrl ; Mouse 2 4 W space ctrl ; Mouse 69 42 W space ctrl ; Mouse 95 17 W space ctrl <|action_end|>
```

## demonstration_optimization_000012

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `scaly-fuchsia-wasp-809d68bd8eea-20220215-163254` |
| 图片帧 | `[610, 614, 618, 622]` |
| 目标动作区间 | `[610, 626]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 610**

![demonstration_optimization_000012 frame 610](images/demonstration_optimization_000012_00.jpg)

**图 2，帧 614**

![demonstration_optimization_000012 frame 614](images/demonstration_optimization_000012_01.jpg)

**图 3，帧 618**

![demonstration_optimization_000012 frame 618](images/demonstration_optimization_000012_02.jpg)

**图 4，帧 622**

![demonstration_optimization_000012 frame 622](images/demonstration_optimization_000012_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W D ; W D ; Mouse -5 11 ; Mouse -25 50 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -41 72 ; Mouse -30 31 ; Mouse -34 10 ; Mouse -5 0 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; A ; A ; A ; A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; A ; A ; Mouse -14 -4 ; Mouse -61 -39 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D ; W D ; Mouse -5 11 ; Mouse -25 50 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -41 72 ; Mouse -30 31 ; Mouse -34 10 ; Mouse -5 0 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; A ; A ; A ; A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; A ; A ; Mouse -14 -4 ; Mouse -61 -39 W <|action_end|>
```

## demonstration_optimization_000013

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220214-144020` |
| 图片帧 | `[3750, 3754, 3758, 3762]` |
| 目标动作区间 | `[3750, 3766]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3750**

![demonstration_optimization_000013 frame 3750](images/demonstration_optimization_000013_00.jpg)

**图 2，帧 3754**

![demonstration_optimization_000013 frame 3754](images/demonstration_optimization_000013_01.jpg)

**图 3，帧 3758**

![demonstration_optimization_000013 frame 3758](images/demonstration_optimization_000013_02.jpg)

**图 4，帧 3762**

![demonstration_optimization_000013 frame 3762](images/demonstration_optimization_000013_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ; W space ; Mouse -6 0 W space ; W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W A space ; W A space ; W A space ; W A space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W A space ; W A space ; W A space ; W A space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A space ; W A space ; W A space ; W A space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; Mouse -6 0 W space ; W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W A space ; W A space ; W A space ; W A space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W A space ; W A space ; W A space ; W A space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A space ; W A space ; W A space ; W A space <|action_end|>
```

## demonstration_optimization_000014

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-64e29ec9526d-20220304-192635` |
| 图片帧 | `[93, 97, 101, 105]` |
| 目标动作区间 | `[93, 109]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 93**

![demonstration_optimization_000014 frame 93](images/demonstration_optimization_000014_00.jpg)

**图 2，帧 97**

![demonstration_optimization_000014 frame 97](images/demonstration_optimization_000014_01.jpg)

**图 3，帧 101**

![demonstration_optimization_000014 frame 101](images/demonstration_optimization_000014_02.jpg)

**图 4，帧 105**

![demonstration_optimization_000014 frame 105](images/demonstration_optimization_000014_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; S ; Mouse 2 0 S ; Mouse 4 0 S ; Mouse 12 3 S <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 8 0 S ; Mouse 11 -3 S ; Mouse 1 0 S ; S A MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; S A MouseLeft ; S A MouseLeft ; S A ; Mouse 5 0 S A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 23 0 S A ; Mouse 24 -2 S A ; Mouse 11 -1 S A ; Mouse 20 -2 S <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; S ; Mouse 2 0 S ; Mouse 4 0 S ; Mouse 12 3 S <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 8 0 S ; Mouse 11 -3 S ; Mouse 1 0 S ; S A MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; S A MouseLeft ; S A MouseLeft ; S A ; Mouse 5 0 S A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 23 0 S A ; Mouse 24 -2 S A ; Mouse 11 -1 S A ; Mouse 20 -2 S <|action_end|>
```

## demonstration_optimization_000015

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220305-172335` |
| 图片帧 | `[33430, 33434, 33438, 33442]` |
| 目标动作区间 | `[33430, 33446]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 33430**

![demonstration_optimization_000015 frame 33430](images/demonstration_optimization_000015_00.jpg)

**图 2，帧 33434**

![demonstration_optimization_000015 frame 33434](images/demonstration_optimization_000015_01.jpg)

**图 3，帧 33438**

![demonstration_optimization_000015 frame 33438](images/demonstration_optimization_000015_02.jpg)

**图 4，帧 33442**

![demonstration_optimization_000015 frame 33442](images/demonstration_optimization_000015_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse 1 1 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse 1 1 MouseLeft <|action_end|>
```

## demonstration_optimization_000016

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-2d9b2e9fbbea-20220323-214755` |
| 图片帧 | `[8306, 8310, 8314, 8318]` |
| 目标动作区间 | `[8306, 8322]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8306**

![demonstration_optimization_000016 frame 8306](images/demonstration_optimization_000016_00.jpg)

**图 2，帧 8310**

![demonstration_optimization_000016 frame 8310](images/demonstration_optimization_000016_01.jpg)

**图 3，帧 8314**

![demonstration_optimization_000016 frame 8314](images/demonstration_optimization_000016_02.jpg)

**图 4，帧 8318**

![demonstration_optimization_000016 frame 8318](images/demonstration_optimization_000016_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -118 -22 W ctrl ; Mouse -92 -10 W ; Mouse -45 0 W ; Mouse -14 3 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 2 W ; Mouse -3 1 W ; Mouse -4 2 W ; Mouse -9 3 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -15 4 W ; Mouse -10 2 W D ; Mouse -11 2 W D ; Mouse -17 4 W D <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -22 4 W D ; Mouse -22 4 W D ; Mouse -27 6 W D ; Mouse -25 5 W D <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -118 -22 W ctrl ; Mouse -92 -10 W ; Mouse -45 0 W ; Mouse -14 3 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 2 W ; Mouse -3 1 W ; Mouse -4 2 W ; Mouse -9 3 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -15 4 W ; Mouse -10 2 W D ; Mouse -11 2 W D ; Mouse -17 4 W D <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -22 4 W D ; Mouse -22 4 W D ; Mouse -27 6 W D ; Mouse -25 5 W D <|action_end|>
```

## demonstration_optimization_000017

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-972909e183ca-20220114-095446` |
| 图片帧 | `[149, 153, 157, 161]` |
| 目标动作区间 | `[149, 165]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 149**

![demonstration_optimization_000017 frame 149](images/demonstration_optimization_000017_00.jpg)

**图 2，帧 153**

![demonstration_optimization_000017 frame 153](images/demonstration_optimization_000017_01.jpg)

**图 3，帧 157**

![demonstration_optimization_000017 frame 157](images/demonstration_optimization_000017_02.jpg)

**图 4，帧 161**

![demonstration_optimization_000017 frame 161](images/demonstration_optimization_000017_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 3 0 W ctrl ; Mouse 1 1 W ; W ; Mouse 0 1 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 2 W ; Mouse 3 1 W ; W ; Mouse 0 3 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W space ; Mouse 1 1 W space ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 1 W space ; Mouse 2 1 W space ; W space ; W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 3 0 W ctrl ; Mouse 1 1 W ; W ; Mouse 0 1 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 2 W ; Mouse 3 1 W ; W ; Mouse 0 3 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W space ; Mouse 1 1 W space ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 1 W space ; Mouse 2 1 W space ; W space ; W space <|action_end|>
```

## demonstration_optimization_000018

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220306-183903` |
| 图片帧 | `[9, 13, 17, 21]` |
| 目标动作区间 | `[9, 25]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9**

![demonstration_optimization_000018 frame 9](images/demonstration_optimization_000018_00.jpg)

**图 2，帧 13**

![demonstration_optimization_000018 frame 13](images/demonstration_optimization_000018_01.jpg)

**图 3，帧 17**

![demonstration_optimization_000018 frame 17](images/demonstration_optimization_000018_02.jpg)

**图 4，帧 21**

![demonstration_optimization_000018 frame 21](images/demonstration_optimization_000018_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -206 -11 ; Mouse -115 -3 ; Mouse 513 48 ; Mouse 357 60 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 4 0 ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 314 -74 W ; Mouse 426 -28 ; Mouse 229 -4 ; Mouse 26 0 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -206 -11 ; Mouse -115 -3 ; Mouse 513 48 ; Mouse 357 60 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 4 0 ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 314 -74 W ; Mouse 426 -28 ; Mouse 229 -4 ; Mouse 26 0 <|action_end|>
```

## demonstration_optimization_000019

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220202-154444` |
| 图片帧 | `[296, 300, 304, 308]` |
| 目标动作区间 | `[296, 312]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 296**

![demonstration_optimization_000019 frame 296](images/demonstration_optimization_000019_00.jpg)

**图 2，帧 300**

![demonstration_optimization_000019 frame 300](images/demonstration_optimization_000019_01.jpg)

**图 3，帧 304**

![demonstration_optimization_000019 frame 304](images/demonstration_optimization_000019_02.jpg)

**图 4，帧 308**

![demonstration_optimization_000019 frame 308](images/demonstration_optimization_000019_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ; Mouse 23 8 W space ; Mouse 12 10 W space ; Mouse 10 0 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 0 W space ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; Mouse 21 4 W space ctrl ; Mouse 21 9 W space ; Mouse 4 0 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 4 2 W space ; Mouse 17 9 W space ; Mouse 29 12 W space ; Mouse 38 12 W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; Mouse 23 8 W space ; Mouse 12 10 W space ; Mouse 10 0 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 0 W space ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; Mouse 21 4 W space ctrl ; Mouse 21 9 W space ; Mouse 4 0 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 4 2 W space ; Mouse 17 9 W space ; Mouse 29 12 W space ; Mouse 38 12 W space <|action_end|>
```

## demonstration_optimization_000020

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player770-f153ac423f61-20211220-073120` |
| 图片帧 | `[718, 722, 726, 730]` |
| 目标动作区间 | `[718, 734]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 718**

![demonstration_optimization_000020 frame 718](images/demonstration_optimization_000020_00.jpg)

**图 2，帧 722**

![demonstration_optimization_000020 frame 722](images/demonstration_optimization_000020_01.jpg)

**图 3，帧 726**

![demonstration_optimization_000020 frame 726](images/demonstration_optimization_000020_02.jpg)

**图 4，帧 730**

![demonstration_optimization_000020 frame 730](images/demonstration_optimization_000020_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## demonstration_optimization_000021

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220217-202443` |
| 图片帧 | `[6064, 6068, 6072, 6076]` |
| 目标动作区间 | `[6064, 6080]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6064**

![demonstration_optimization_000021 frame 6064](images/demonstration_optimization_000021_00.jpg)

**图 2，帧 6068**

![demonstration_optimization_000021 frame 6068](images/demonstration_optimization_000021_01.jpg)

**图 3，帧 6072**

![demonstration_optimization_000021 frame 6072](images/demonstration_optimization_000021_02.jpg)

**图 4，帧 6076**

![demonstration_optimization_000021 frame 6076](images/demonstration_optimization_000021_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; Mouse -11 -1 W ; Mouse -136 -7 W ; Mouse -435 -21 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -362 6 W ; Mouse 0 -2 W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ; W A ; W A ; Mouse -75 3 W A <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; Mouse -11 -1 W ; Mouse -136 -7 W ; Mouse -435 -21 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -362 6 W ; Mouse 0 -2 W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ; W A ; W A ; Mouse -75 3 W A <|action_end|>
```

## demonstration_optimization_000022

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-21605bba2708-20220220-025451` |
| 图片帧 | `[2156, 2160, 2164, 2168]` |
| 目标动作区间 | `[2156, 2172]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2156**

![demonstration_optimization_000022 frame 2156](images/demonstration_optimization_000022_00.jpg)

**图 2，帧 2160**

![demonstration_optimization_000022 frame 2160](images/demonstration_optimization_000022_01.jpg)

**图 3，帧 2164**

![demonstration_optimization_000022 frame 2164](images/demonstration_optimization_000022_02.jpg)

**图 4，帧 2168**

![demonstration_optimization_000022 frame 2168](images/demonstration_optimization_000022_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; Mouse 1 0 W ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ctrl ; Mouse 43 -29 ; Mouse 9 -139 ; Mouse -46 -50 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; Mouse 1 0 W ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ctrl ; Mouse 43 -29 ; Mouse 9 -139 ; Mouse -46 -50 <|action_end|>
```

## demonstration_optimization_000023

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20220122-110827` |
| 图片帧 | `[899, 903, 907, 911]` |
| 目标动作区间 | `[899, 915]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 899**

![demonstration_optimization_000023 frame 899](images/demonstration_optimization_000023_00.jpg)

**图 2，帧 903**

![demonstration_optimization_000023 frame 903](images/demonstration_optimization_000023_01.jpg)

**图 3，帧 907**

![demonstration_optimization_000023 frame 907](images/demonstration_optimization_000023_02.jpg)

**图 4，帧 911**

![demonstration_optimization_000023 frame 911](images/demonstration_optimization_000023_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseRight ; MouseRight ; MouseRight ; MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; space ; space ; space ; space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; space ; space ; space ; MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseRight ; Mouse -3 1 MouseRight ; MouseRight ; space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ; MouseRight ; MouseRight ; MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; space ; space ; space ; space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; space ; space ; space ; MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseRight ; Mouse -3 1 MouseRight ; MouseRight ; space <|action_end|>
```

## demonstration_optimization_000024

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220125-030216` |
| 图片帧 | `[31897, 31901, 31905, 31909]` |
| 目标动作区间 | `[31897, 31913]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 31897**

![demonstration_optimization_000024 frame 31897](images/demonstration_optimization_000024_00.jpg)

**图 2，帧 31901**

![demonstration_optimization_000024 frame 31901](images/demonstration_optimization_000024_01.jpg)

**图 3，帧 31905**

![demonstration_optimization_000024 frame 31905](images/demonstration_optimization_000024_02.jpg)

**图 4，帧 31909**

![demonstration_optimization_000024 frame 31909](images/demonstration_optimization_000024_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W D shift ; W D shift ; W D shift ; D shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; D shift ; D shift ; Mouse -49 -13 A shift ; Mouse -193 -41 A shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -89 -43 A shift ; Mouse -36 -30 A shift ; Mouse -36 -15 W A shift ; Mouse -76 -20 W A shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -57 -10 W A shift ; Mouse -21 -5 W A shift ; W A shift ; W A shift <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D shift ; W D shift ; W D shift ; D shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; D shift ; D shift ; Mouse -49 -13 A shift ; Mouse -193 -41 A shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -89 -43 A shift ; Mouse -36 -30 A shift ; Mouse -36 -15 W A shift ; Mouse -76 -20 W A shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -57 -10 W A shift ; Mouse -21 -5 W A shift ; W A shift ; W A shift <|action_end|>
```

## demonstration_optimization_000025

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220215-212605` |
| 图片帧 | `[4603, 4607, 4611, 4615]` |
| 目标动作区间 | `[4603, 4619]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4603**

![demonstration_optimization_000025 frame 4603](images/demonstration_optimization_000025_00.jpg)

**图 2，帧 4607**

![demonstration_optimization_000025 frame 4607](images/demonstration_optimization_000025_01.jpg)

**图 3，帧 4611**

![demonstration_optimization_000025 frame 4611](images/demonstration_optimization_000025_02.jpg)

**图 4，帧 4615**

![demonstration_optimization_000025 frame 4615](images/demonstration_optimization_000025_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; Mouse -1 3 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 W space ; W space ; W space ; Mouse 0 1 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W space ; Mouse -1 0 W space ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ; W space ; Mouse -2 3 W space ; Mouse -4 3 W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; Mouse -1 3 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 W space ; W space ; W space ; Mouse 0 1 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W space ; Mouse -1 0 W space ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ; W space ; Mouse -2 3 W space ; Mouse -4 3 W space <|action_end|>
```

## demonstration_optimization_000026

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-8331f4a58ff9-20220130-070359` |
| 图片帧 | `[25, 29, 33, 37]` |
| 目标动作区间 | `[25, 41]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 25**

![demonstration_optimization_000026 frame 25](images/demonstration_optimization_000026_00.jpg)

**图 2，帧 29**

![demonstration_optimization_000026 frame 29](images/demonstration_optimization_000026_01.jpg)

**图 3，帧 33**

![demonstration_optimization_000026 frame 33](images/demonstration_optimization_000026_02.jpg)

**图 4，帧 37**

![demonstration_optimization_000026 frame 37](images/demonstration_optimization_000026_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 1 0 W space ctrl ; Mouse 6 0 W space ctrl ; Mouse 46 6 W space ctrl ; Mouse 48 21 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -28 4 W space ctrl ; Mouse -2 -1 W space ctrl ; W ; Mouse -13 9 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 53 2 ; Mouse 162 16 ; Mouse 161 9 ; Mouse 104 18 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 37 8 ; Mouse 1 0 ; Mouse -2 1 ; Mouse -38 6 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 W space ctrl ; Mouse 6 0 W space ctrl ; Mouse 46 6 W space ctrl ; Mouse 48 21 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -28 4 W space ctrl ; Mouse -2 -1 W space ctrl ; W ; Mouse -13 9 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 53 2 ; Mouse 162 16 ; Mouse 161 9 ; Mouse 104 18 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 37 8 ; Mouse 1 0 ; Mouse -2 1 ; Mouse -38 6 <|action_end|>
```

## demonstration_optimization_000027

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220217-010123` |
| 图片帧 | `[1519, 1523, 1527, 1531]` |
| 目标动作区间 | `[1519, 1535]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1519**

![demonstration_optimization_000027 frame 1519](images/demonstration_optimization_000027_00.jpg)

**图 2，帧 1523**

![demonstration_optimization_000027 frame 1523](images/demonstration_optimization_000027_01.jpg)

**图 3，帧 1527**

![demonstration_optimization_000027 frame 1527](images/demonstration_optimization_000027_02.jpg)

**图 4，帧 1531**

![demonstration_optimization_000027 frame 1531](images/demonstration_optimization_000027_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 2 ; Mouse -1 0 ; MouseLeft ;  <|action_end|>
```

动作块 2：

```text
<|action_start|> ;  ;  ; Mouse 1 2 ; Mouse 10 2 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 40 -4 ; Mouse 57 -22 ; Mouse 6 -10 ; Mouse 0 -8 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -2 -1 ; Mouse 0 -2 ; Mouse -9 -1 ; Mouse -3 0 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 2 ; Mouse -1 0 ; MouseLeft ;  <|action_end|>
```

动作块 2：

```text
<|action_start|> ;  ;  ; Mouse 1 2 ; Mouse 10 2 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 40 -4 ; Mouse 57 -22 ; Mouse 6 -10 ; Mouse 0 -8 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -2 -1 ; Mouse 0 -2 ; Mouse -9 -1 ; Mouse -3 0 <|action_end|>
```

## demonstration_optimization_000028

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player565-f153ac423f61-20220204-212225` |
| 图片帧 | `[220, 224, 228, 232]` |
| 目标动作区间 | `[220, 236]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 220**

![demonstration_optimization_000028 frame 220](images/demonstration_optimization_000028_00.jpg)

**图 2，帧 224**

![demonstration_optimization_000028 frame 224](images/demonstration_optimization_000028_01.jpg)

**图 3，帧 228**

![demonstration_optimization_000028 frame 228](images/demonstration_optimization_000028_02.jpg)

**图 4，帧 232**

![demonstration_optimization_000028 frame 232](images/demonstration_optimization_000028_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; Mouse -2 6 shift ; Mouse -4 15 shift ; Mouse -1 2 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -3 0 shift ; Mouse -4 -4 shift ; Mouse 0 -3 shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift MouseLeft ; shift ; Mouse 2 0 shift ; Mouse 9 8 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 26 32 shift ; Mouse 57 28 shift ; Mouse 26 6 shift ; Mouse -2 -2 shift <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; Mouse -2 6 shift ; Mouse -4 15 shift ; Mouse -1 2 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -3 0 shift ; Mouse -4 -4 shift ; Mouse 0 -3 shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift MouseLeft ; shift ; Mouse 2 0 shift ; Mouse 9 8 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 26 32 shift ; Mouse 57 28 shift ; Mouse 26 6 shift ; Mouse -2 -2 shift <|action_end|>
```

## demonstration_optimization_000029

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `squeaky-ultramarine-chihuahua-f153ac423f61-20220304-003328` |
| 图片帧 | `[108, 112, 116, 120]` |
| 目标动作区间 | `[108, 124]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 108**

![demonstration_optimization_000029 frame 108](images/demonstration_optimization_000029_00.jpg)

**图 2，帧 112**

![demonstration_optimization_000029 frame 112](images/demonstration_optimization_000029_01.jpg)

**图 3，帧 116**

![demonstration_optimization_000029 frame 116](images/demonstration_optimization_000029_02.jpg)

**图 4，帧 120**

![demonstration_optimization_000029 frame 120](images/demonstration_optimization_000029_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## demonstration_optimization_000030

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20220126-232602` |
| 图片帧 | `[12692, 12696, 12700, 12704]` |
| 目标动作区间 | `[12692, 12708]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12692**

![demonstration_optimization_000030 frame 12692](images/demonstration_optimization_000030_00.jpg)

**图 2，帧 12696**

![demonstration_optimization_000030 frame 12696](images/demonstration_optimization_000030_01.jpg)

**图 3，帧 12700**

![demonstration_optimization_000030 frame 12700](images/demonstration_optimization_000030_02.jpg)

**图 4，帧 12704**

![demonstration_optimization_000030 frame 12704](images/demonstration_optimization_000030_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -16 0 W ; Mouse -27 0 W ; Mouse -5 0 ; Mouse -1 -1 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -5 0 W ; Mouse -5 0 W ; W ; Mouse 24 1 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 95 0 W ; Mouse 33 0 W ; Mouse 42 0 W ; Mouse 104 0 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 138 0 W ; Mouse 49 0 W space ; Mouse 74 0 W space ; Mouse 66 0 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -16 0 W ; Mouse -27 0 W ; Mouse -5 0 ; Mouse -1 -1 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -5 0 W ; Mouse -5 0 W ; W ; Mouse 24 1 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 95 0 W ; Mouse 33 0 W ; Mouse 42 0 W ; Mouse 104 0 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 138 0 W ; Mouse 49 0 W space ; Mouse 74 0 W space ; Mouse 66 0 W <|action_end|>
```

## demonstration_optimization_000031

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-01e4953da19b-20220301-063613` |
| 图片帧 | `[1514, 1518, 1522, 1526]` |
| 目标动作区间 | `[1514, 1530]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1514**

![demonstration_optimization_000031 frame 1514](images/demonstration_optimization_000031_00.jpg)

**图 2，帧 1518**

![demonstration_optimization_000031 frame 1518](images/demonstration_optimization_000031_01.jpg)

**图 3，帧 1522**

![demonstration_optimization_000031 frame 1522](images/demonstration_optimization_000031_02.jpg)

**图 4，帧 1526**

![demonstration_optimization_000031 frame 1526](images/demonstration_optimization_000031_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -72 0 shift ; Mouse -19 -2 shift ; Mouse -9 -13 shift ; Mouse 5 -6 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 28 -10 shift ; Mouse 8 -4 shift ; Mouse 0 -2 shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -3 -2 shift ; Mouse -20 -2 shift ; Mouse -23 2 shift ; Mouse -8 0 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -6 0 shift ; shift ; shift MouseLeft ; shift <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -72 0 shift ; Mouse -19 -2 shift ; Mouse -9 -13 shift ; Mouse 5 -6 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 28 -10 shift ; Mouse 8 -4 shift ; Mouse 0 -2 shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -3 -2 shift ; Mouse -20 -2 shift ; Mouse -23 2 shift ; Mouse -8 0 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -6 0 shift ; shift ; shift MouseLeft ; shift <|action_end|>
```

## demonstration_optimization_000032

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-78c527f56667-20220115-091750` |
| 图片帧 | `[16385, 16389, 16393, 16397]` |
| 目标动作区间 | `[16385, 16401]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 16385**

![demonstration_optimization_000032 frame 16385](images/demonstration_optimization_000032_00.jpg)

**图 2，帧 16389**

![demonstration_optimization_000032 frame 16389](images/demonstration_optimization_000032_01.jpg)

**图 3，帧 16393**

![demonstration_optimization_000032 frame 16393](images/demonstration_optimization_000032_02.jpg)

**图 4，帧 16397**

![demonstration_optimization_000032 frame 16397](images/demonstration_optimization_000032_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse 1 0 W ctrl ; Mouse 50 16 W space ctrl ; Mouse 40 16 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 1 W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; Mouse -24 8 W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse 1 0 W ctrl ; Mouse 50 16 W space ctrl ; Mouse 40 16 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 1 W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; Mouse -24 8 W space ctrl <|action_end|>
```

## demonstration_optimization_000033

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220112-103458` |
| 图片帧 | `[14173, 14177, 14181, 14185]` |
| 目标动作区间 | `[14173, 14189]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 14173**

![demonstration_optimization_000033 frame 14173](images/demonstration_optimization_000033_00.jpg)

**图 2，帧 14177**

![demonstration_optimization_000033 frame 14177](images/demonstration_optimization_000033_01.jpg)

**图 3，帧 14181**

![demonstration_optimization_000033 frame 14181](images/demonstration_optimization_000033_02.jpg)

**图 4，帧 14185**

![demonstration_optimization_000033 frame 14185](images/demonstration_optimization_000033_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 3 -4 ; Mouse 1 -4 MouseRight ; Mouse 0 -16 MouseRight ; Mouse 3 -14 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 -15 ; Mouse 0 -31 ; Mouse 0 -17 ; Mouse 0 -7 MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseRight ; MouseRight ; Mouse 1 -3 ; MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseRight ; Mouse -5 8 ; Mouse -70 48 ; Mouse -15 11 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 3 -4 ; Mouse 1 -4 MouseRight ; Mouse 0 -16 MouseRight ; Mouse 3 -14 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 -15 ; Mouse 0 -31 ; Mouse 0 -17 ; Mouse 0 -7 MouseRight <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseRight ; MouseRight ; Mouse 1 -3 ; MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseRight ; Mouse -5 8 ; Mouse -70 48 ; Mouse -15 11 <|action_end|>
```

## demonstration_optimization_000034

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player445-7b77b6e4e459-20211210-205506` |
| 图片帧 | `[2445, 2449, 2453, 2457]` |
| 目标动作区间 | `[2445, 2461]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2445**

![demonstration_optimization_000034 frame 2445](images/demonstration_optimization_000034_00.jpg)

**图 2，帧 2449**

![demonstration_optimization_000034 frame 2449](images/demonstration_optimization_000034_01.jpg)

**图 3，帧 2453**

![demonstration_optimization_000034 frame 2453](images/demonstration_optimization_000034_02.jpg)

**图 4，帧 2457**

![demonstration_optimization_000034 frame 2457](images/demonstration_optimization_000034_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000035

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220208-214114` |
| 图片帧 | `[780, 784, 788, 792]` |
| 目标动作区间 | `[780, 796]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 780**

![demonstration_optimization_000035 frame 780](images/demonstration_optimization_000035_00.jpg)

**图 2，帧 784**

![demonstration_optimization_000035 frame 784](images/demonstration_optimization_000035_01.jpg)

**图 3，帧 788**

![demonstration_optimization_000035 frame 788](images/demonstration_optimization_000035_02.jpg)

**图 4，帧 792**

![demonstration_optimization_000035 frame 792](images/demonstration_optimization_000035_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; Mouse -1 -4 MouseLeft ; Mouse -2 -5 MouseLeft ; Mouse 0 -3 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; Mouse -1 -4 MouseLeft ; Mouse -2 -5 MouseLeft ; Mouse 0 -3 MouseLeft <|action_end|>
```

## demonstration_optimization_000036

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-f80804d55877-20211226-173217` |
| 图片帧 | `[19443, 19447, 19451, 19455]` |
| 目标动作区间 | `[19443, 19459]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 19443**

![demonstration_optimization_000036 frame 19443](images/demonstration_optimization_000036_00.jpg)

**图 2，帧 19447**

![demonstration_optimization_000036 frame 19447](images/demonstration_optimization_000036_01.jpg)

**图 3，帧 19451**

![demonstration_optimization_000036 frame 19451](images/demonstration_optimization_000036_02.jpg)

**图 4，帧 19455**

![demonstration_optimization_000036 frame 19455](images/demonstration_optimization_000036_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000037

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220202-142316` |
| 图片帧 | `[3954, 3958, 3962, 3966]` |
| 目标动作区间 | `[3954, 3970]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3954**

![demonstration_optimization_000037 frame 3954](images/demonstration_optimization_000037_00.jpg)

**图 2，帧 3958**

![demonstration_optimization_000037 frame 3958](images/demonstration_optimization_000037_01.jpg)

**图 3，帧 3962**

![demonstration_optimization_000037 frame 3962](images/demonstration_optimization_000037_02.jpg)

**图 4，帧 3966**

![demonstration_optimization_000037 frame 3966](images/demonstration_optimization_000037_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000038

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220202-145930` |
| 图片帧 | `[2127, 2131, 2135, 2139]` |
| 目标动作区间 | `[2127, 2143]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2127**

![demonstration_optimization_000038 frame 2127](images/demonstration_optimization_000038_00.jpg)

**图 2，帧 2131**

![demonstration_optimization_000038 frame 2131](images/demonstration_optimization_000038_01.jpg)

**图 3，帧 2135**

![demonstration_optimization_000038 frame 2135](images/demonstration_optimization_000038_02.jpg)

**图 4，帧 2139**

![demonstration_optimization_000038 frame 2139](images/demonstration_optimization_000038_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ; W space ; Mouse 1 1 W space ; Mouse 9 0 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 8 0 W space ; Mouse 6 0 W space ; Mouse 2 0 W space ctrl ; Mouse 2 -1 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; Mouse 2 -2 W space ctrl ; Mouse 10 -2 W space ; Mouse 24 -1 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 37 0 W space ; Mouse 27 0 W space ; Mouse 20 -3 W space ; Mouse 23 0 W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; Mouse 1 1 W space ; Mouse 9 0 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 8 0 W space ; Mouse 6 0 W space ; Mouse 2 0 W space ctrl ; Mouse 2 -1 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; Mouse 2 -2 W space ctrl ; Mouse 10 -2 W space ; Mouse 24 -1 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 37 0 W space ; Mouse 27 0 W space ; Mouse 20 -3 W space ; Mouse 23 0 W space <|action_end|>
```

## demonstration_optimization_000039

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `wiggy-aquamarine-tapir-f153ac423f61-20220302-200419` |
| 图片帧 | `[1456, 1460, 1464, 1468]` |
| 目标动作区间 | `[1456, 1472]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1456**

![demonstration_optimization_000039 frame 1456](images/demonstration_optimization_000039_00.jpg)

**图 2，帧 1460**

![demonstration_optimization_000039 frame 1460](images/demonstration_optimization_000039_01.jpg)

**图 3，帧 1464**

![demonstration_optimization_000039 frame 1464](images/demonstration_optimization_000039_02.jpg)

**图 4，帧 1468**

![demonstration_optimization_000039 frame 1468](images/demonstration_optimization_000039_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; Mouse 10 4 MouseLeft ; Mouse 35 17 A MouseLeft ; Mouse 26 9 W A MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 1 W A MouseLeft ; W A MouseLeft ; Mouse 10 13 W A MouseLeft ; Mouse 30 41 W A MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 27 36 A MouseLeft ; Mouse 48 33 A MouseLeft ; Mouse 66 21 MouseLeft ; Mouse 71 6 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; Mouse 10 4 MouseLeft ; Mouse 35 17 A MouseLeft ; Mouse 26 9 W A MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 1 W A MouseLeft ; W A MouseLeft ; Mouse 10 13 W A MouseLeft ; Mouse 30 41 W A MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 27 36 A MouseLeft ; Mouse 48 33 A MouseLeft ; Mouse 66 21 MouseLeft ; Mouse 71 6 MouseLeft <|action_end|>
```

## demonstration_optimization_000040

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20220129-184553` |
| 图片帧 | `[6641, 6645, 6649, 6653]` |
| 目标动作区间 | `[6641, 6657]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6641**

![demonstration_optimization_000040 frame 6641](images/demonstration_optimization_000040_00.jpg)

**图 2，帧 6645**

![demonstration_optimization_000040 frame 6645](images/demonstration_optimization_000040_01.jpg)

**图 3，帧 6649**

![demonstration_optimization_000040 frame 6649](images/demonstration_optimization_000040_02.jpg)

**图 4，帧 6653**

![demonstration_optimization_000040 frame 6653](images/demonstration_optimization_000040_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 1 0 W ; W ; Mouse -8 8 W D ; Mouse -20 11 W D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -60 18 ; Mouse -129 -1 ; Mouse -183 -25 ; Mouse -237 -53 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -123 -27 W ; Mouse -86 -16 W ; Mouse -45 -14 W ; Mouse -28 -7 W D <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -12 -4 W D ; Mouse -16 -1 W ; Mouse -11 0 W ; Mouse -15 1 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 W ; W ; Mouse -8 8 W D ; Mouse -20 11 W D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -60 18 ; Mouse -129 -1 ; Mouse -183 -25 ; Mouse -237 -53 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -123 -27 W ; Mouse -86 -16 W ; Mouse -45 -14 W ; Mouse -28 -7 W D <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -12 -4 W D ; Mouse -16 -1 W ; Mouse -11 0 W ; Mouse -15 1 W <|action_end|>
```

## demonstration_optimization_000041

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-54e75e98eb61-20220129-020420` |
| 图片帧 | `[4242, 4246, 4250, 4254]` |
| 目标动作区间 | `[4242, 4258]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4242**

![demonstration_optimization_000041 frame 4242](images/demonstration_optimization_000041_00.jpg)

**图 2，帧 4246**

![demonstration_optimization_000041 frame 4246](images/demonstration_optimization_000041_01.jpg)

**图 3，帧 4250**

![demonstration_optimization_000041 frame 4250](images/demonstration_optimization_000041_02.jpg)

**图 4，帧 4254**

![demonstration_optimization_000041 frame 4254](images/demonstration_optimization_000041_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 71 -1 A ; Mouse 75 -3 A ; Mouse 49 -7 A ; Mouse 19 -5 A <|action_end|>
```

动作块 2：

```text
<|action_start|> ; A ; A ; A ; A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 0 -1 A ; Mouse -12 -6 A ; Mouse -42 -15 A ; Mouse -109 -26 A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -79 -18 A ; Mouse -72 -29 A ; Mouse -33 -19 A ; Mouse -7 -8 W A <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 71 -1 A ; Mouse 75 -3 A ; Mouse 49 -7 A ; Mouse 19 -5 A <|action_end|>
```

动作块 2：

```text
<|action_start|> ; A ; A ; A ; A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 0 -1 A ; Mouse -12 -6 A ; Mouse -42 -15 A ; Mouse -109 -26 A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -79 -18 A ; Mouse -72 -29 A ; Mouse -33 -19 A ; Mouse -7 -8 W A <|action_end|>
```

## demonstration_optimization_000042

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `wiggy-aquamarine-tapir-167c21c6a7b9-20220116-235053` |
| 图片帧 | `[848, 852, 856, 860]` |
| 目标动作区间 | `[848, 864]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 848**

![demonstration_optimization_000042 frame 848](images/demonstration_optimization_000042_00.jpg)

**图 2，帧 852**

![demonstration_optimization_000042 frame 852](images/demonstration_optimization_000042_01.jpg)

**图 3，帧 856**

![demonstration_optimization_000042 frame 856](images/demonstration_optimization_000042_02.jpg)

**图 4，帧 860**

![demonstration_optimization_000042 frame 860](images/demonstration_optimization_000042_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 MouseLeft ; Mouse -4 11 MouseLeft ; Mouse 0 1 MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; Mouse 2 1 MouseLeft ; Mouse 2 -1 MouseLeft ; Mouse 0 -1 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 -4 MouseLeft ; Mouse 0 -4 MouseLeft ; Mouse 0 -5 MouseLeft ; Mouse -1 -5 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 MouseLeft ; Mouse -4 11 MouseLeft ; Mouse 0 1 MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; Mouse 2 1 MouseLeft ; Mouse 2 -1 MouseLeft ; Mouse 0 -1 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 -4 MouseLeft ; Mouse 0 -4 MouseLeft ; Mouse 0 -5 MouseLeft ; Mouse -1 -5 MouseLeft <|action_end|>
```

## demonstration_optimization_000043

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-e6dad09ade3b-20220213-001127` |
| 图片帧 | `[8720, 8724, 8728, 8732]` |
| 目标动作区间 | `[8720, 8736]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8720**

![demonstration_optimization_000043 frame 8720](images/demonstration_optimization_000043_00.jpg)

**图 2，帧 8724**

![demonstration_optimization_000043 frame 8724](images/demonstration_optimization_000043_01.jpg)

**图 3，帧 8728**

![demonstration_optimization_000043 frame 8728](images/demonstration_optimization_000043_02.jpg)

**图 4，帧 8732**

![demonstration_optimization_000043 frame 8732](images/demonstration_optimization_000043_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 2 17 W MouseLeft ; Mouse 8 22 W MouseLeft ; Mouse 4 7 W MouseLeft ; Mouse 6 11 W MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 11 15 W MouseLeft ; Mouse 8 10 W MouseLeft ; Mouse 3 5 W MouseLeft ; Mouse 2 4 W MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 2 W MouseLeft ; Mouse 2 2 W MouseLeft ; W MouseLeft ; W MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W MouseLeft ; W MouseLeft ; Mouse 1 -9 W MouseLeft ; Mouse 1 -20 W MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 2 17 W MouseLeft ; Mouse 8 22 W MouseLeft ; Mouse 4 7 W MouseLeft ; Mouse 6 11 W MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 11 15 W MouseLeft ; Mouse 8 10 W MouseLeft ; Mouse 3 5 W MouseLeft ; Mouse 2 4 W MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 2 W MouseLeft ; Mouse 2 2 W MouseLeft ; W MouseLeft ; W MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W MouseLeft ; W MouseLeft ; Mouse 1 -9 W MouseLeft ; Mouse 1 -20 W MouseLeft <|action_end|>
```

## demonstration_optimization_000044

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20211226-215819` |
| 图片帧 | `[169, 173, 177, 181]` |
| 目标动作区间 | `[169, 185]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 169**

![demonstration_optimization_000044 frame 169](images/demonstration_optimization_000044_00.jpg)

**图 2，帧 173**

![demonstration_optimization_000044 frame 173](images/demonstration_optimization_000044_01.jpg)

**图 3，帧 177**

![demonstration_optimization_000044 frame 177](images/demonstration_optimization_000044_02.jpg)

**图 4，帧 181**

![demonstration_optimization_000044 frame 181](images/demonstration_optimization_000044_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift ; shift ; shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; shift ; Mouse 6 6 shift ; Mouse -37 2 ; Mouse 2 0 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 8 3 ;  ;  ;  <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -4 -2 ; Mouse -15 0 ; Mouse -10 -5 ; Mouse -4 0 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift ; shift ; shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; shift ; Mouse 6 6 shift ; Mouse -37 2 ; Mouse 2 0 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 8 3 ;  ;  ;  <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -4 -2 ; Mouse -15 0 ; Mouse -10 -5 ; Mouse -4 0 <|action_end|>
```

## demonstration_optimization_000045

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220222-224654` |
| 图片帧 | `[431, 435, 439, 443]` |
| 目标动作区间 | `[431, 447]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 431**

![demonstration_optimization_000045 frame 431](images/demonstration_optimization_000045_00.jpg)

**图 2，帧 435**

![demonstration_optimization_000045 frame 435](images/demonstration_optimization_000045_01.jpg)

**图 3，帧 439**

![demonstration_optimization_000045 frame 439](images/demonstration_optimization_000045_02.jpg)

**图 4，帧 443**

![demonstration_optimization_000045 frame 443](images/demonstration_optimization_000045_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 0 1 W A ; Mouse 7 0 W A ; Mouse 10 0 A ; D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; D ; Mouse 0 -1 D ; Mouse -26 -6 S D ; Mouse -29 -6 S D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -21 -10 S D ; Mouse -29 -23 S ; Mouse -8 -8 ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; Mouse 0 -4 MouseLeft ; Mouse -3 -13 MouseLeft ; Mouse -1 -4 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 1 W A ; Mouse 7 0 W A ; Mouse 10 0 A ; D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; D ; Mouse 0 -1 D ; Mouse -26 -6 S D ; Mouse -29 -6 S D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -21 -10 S D ; Mouse -29 -23 S ; Mouse -8 -8 ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; Mouse 0 -4 MouseLeft ; Mouse -3 -13 MouseLeft ; Mouse -1 -4 MouseLeft <|action_end|>
```

## demonstration_optimization_000046

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220113-001334` |
| 图片帧 | `[5428, 5432, 5436, 5440]` |
| 目标动作区间 | `[5428, 5444]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5428**

![demonstration_optimization_000046 frame 5428](images/demonstration_optimization_000046_00.jpg)

**图 2，帧 5432**

![demonstration_optimization_000046 frame 5432](images/demonstration_optimization_000046_01.jpg)

**图 3，帧 5436**

![demonstration_optimization_000046 frame 5436](images/demonstration_optimization_000046_02.jpg)

**图 4，帧 5440**

![demonstration_optimization_000046 frame 5440](images/demonstration_optimization_000046_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000047

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220215-182943` |
| 图片帧 | `[21, 25, 29, 33]` |
| 目标动作区间 | `[21, 37]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 21**

![demonstration_optimization_000047 frame 21](images/demonstration_optimization_000047_00.jpg)

**图 2，帧 25**

![demonstration_optimization_000047 frame 25](images/demonstration_optimization_000047_01.jpg)

**图 3，帧 29**

![demonstration_optimization_000047 frame 29](images/demonstration_optimization_000047_02.jpg)

**图 4，帧 33**

![demonstration_optimization_000047 frame 33](images/demonstration_optimization_000047_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 1 -1 ; Mouse -15 5 ; Mouse -58 9 ; Mouse -19 -8 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 46 -3 ; Mouse -3 -7 ; Mouse -127 -7 A ; Mouse -324 -6 A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -239 -3 A ; Mouse -85 1 W A ; W A ; W A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ; W A ; W A ; W A <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 -1 ; Mouse -15 5 ; Mouse -58 9 ; Mouse -19 -8 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 46 -3 ; Mouse -3 -7 ; Mouse -127 -7 A ; Mouse -324 -6 A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -239 -3 A ; Mouse -85 1 W A ; W A ; W A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ; W A ; W A ; W A <|action_end|>
```

## demonstration_optimization_000048

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-c9ec9ad8ca63-20220118-140050` |
| 图片帧 | `[324, 328, 332, 336]` |
| 目标动作区间 | `[324, 340]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 324**

![demonstration_optimization_000048 frame 324](images/demonstration_optimization_000048_00.jpg)

**图 2，帧 328**

![demonstration_optimization_000048 frame 328](images/demonstration_optimization_000048_01.jpg)

**图 3，帧 332**

![demonstration_optimization_000048 frame 332](images/demonstration_optimization_000048_02.jpg)

**图 4，帧 336**

![demonstration_optimization_000048 frame 336](images/demonstration_optimization_000048_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 2 0 ; Mouse 1 0 ; A ; A <|action_end|>
```

动作块 2：

```text
<|action_start|> ; A ; Mouse 2 0 A ; Mouse 3 0 A ; Mouse 4 1 A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 5 1 A ; Mouse 5 0 A ; Mouse 4 0 A ; Mouse 4 0 A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 5 0 A ; Mouse 3 0 A ; Mouse 4 0 A ; Mouse 2 0 A <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 2 0 ; Mouse 1 0 ; A ; A <|action_end|>
```

动作块 2：

```text
<|action_start|> ; A ; Mouse 2 0 A ; Mouse 3 0 A ; Mouse 4 1 A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 5 1 A ; Mouse 5 0 A ; Mouse 4 0 A ; Mouse 4 0 A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 5 0 A ; Mouse 3 0 A ; Mouse 4 0 A ; Mouse 2 0 A <|action_end|>
```

## demonstration_optimization_000049

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-90aeb6fb618f-20220201-180405` |
| 图片帧 | `[2702, 2706, 2710, 2714]` |
| 目标动作区间 | `[2702, 2718]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2702**

![demonstration_optimization_000049 frame 2702](images/demonstration_optimization_000049_00.jpg)

**图 2，帧 2706**

![demonstration_optimization_000049 frame 2706](images/demonstration_optimization_000049_01.jpg)

**图 3，帧 2710**

![demonstration_optimization_000049 frame 2710](images/demonstration_optimization_000049_02.jpg)

**图 4，帧 2714**

![demonstration_optimization_000049 frame 2714](images/demonstration_optimization_000049_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; shift ; shift ; shift ; shift 9 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; shift ; shift ; shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -2 1 shift ; Mouse -10 18 shift ; Mouse -11 44 shift ; Mouse -5 20 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -3 26 shift MouseLeft ; Mouse 0 2 shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift ; shift ; shift ; shift 9 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; shift ; shift ; shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -2 1 shift ; Mouse -10 18 shift ; Mouse -11 44 shift ; Mouse -5 20 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -3 26 shift MouseLeft ; Mouse 0 2 shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## demonstration_optimization_000050

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220209-214101` |
| 图片帧 | `[907, 911, 915, 919]` |
| 目标动作区间 | `[907, 923]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 907**

![demonstration_optimization_000050 frame 907](images/demonstration_optimization_000050_00.jpg)

**图 2，帧 911**

![demonstration_optimization_000050 frame 911](images/demonstration_optimization_000050_01.jpg)

**图 3，帧 915**

![demonstration_optimization_000050 frame 915](images/demonstration_optimization_000050_02.jpg)

**图 4，帧 919**

![demonstration_optimization_000050 frame 919](images/demonstration_optimization_000050_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; Mouse -26 0 W ; Mouse -110 2 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -137 7 W ; Mouse -65 -19 W ; Mouse -4 -11 W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; Mouse -26 0 W ; Mouse -110 2 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -137 7 W ; Mouse -65 -19 W ; Mouse -4 -11 W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## demonstration_optimization_000051

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220202-131354` |
| 图片帧 | `[3399, 3403, 3407, 3411]` |
| 目标动作区间 | `[3399, 3415]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3399**

![demonstration_optimization_000051 frame 3399](images/demonstration_optimization_000051_00.jpg)

**图 2，帧 3403**

![demonstration_optimization_000051 frame 3403](images/demonstration_optimization_000051_01.jpg)

**图 3，帧 3407**

![demonstration_optimization_000051 frame 3407](images/demonstration_optimization_000051_02.jpg)

**图 4，帧 3411**

![demonstration_optimization_000051 frame 3411](images/demonstration_optimization_000051_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -25 -5 W ; Mouse -17 -3 W ; Mouse -30 -1 ; Mouse -28 -2 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -19 -2 ; Mouse -19 0 ; Mouse -39 0 W ; Mouse -51 0 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -68 0 W ; Mouse -38 0 W ; Mouse -7 0 W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; Mouse 5 0 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -25 -5 W ; Mouse -17 -3 W ; Mouse -30 -1 ; Mouse -28 -2 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -19 -2 ; Mouse -19 0 ; Mouse -39 0 W ; Mouse -51 0 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -68 0 W ; Mouse -38 0 W ; Mouse -7 0 W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; Mouse 5 0 W <|action_end|>
```

## demonstration_optimization_000052

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `hazy-thistle-chipmunk-f153ac423f61-20220121-005952` |
| 图片帧 | `[76, 80, 84, 88]` |
| 目标动作区间 | `[76, 92]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 76**

![demonstration_optimization_000052 frame 76](images/demonstration_optimization_000052_00.jpg)

**图 2，帧 80**

![demonstration_optimization_000052 frame 80](images/demonstration_optimization_000052_01.jpg)

**图 3，帧 84**

![demonstration_optimization_000052 frame 84](images/demonstration_optimization_000052_02.jpg)

**图 4，帧 88**

![demonstration_optimization_000052 frame 88](images/demonstration_optimization_000052_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -5 -6 W ; Mouse 0 -14 W ; Mouse -2 -6 W ; Mouse -6 -3 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -30 13 W ; Mouse -14 7 ; Mouse -1 2 ; Mouse -12 29 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -5 17 ; Mouse -3 12 ; Mouse 5 1 ; Mouse 14 0 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; Mouse -1 0 MouseLeft ; Mouse 2 -1 ; Mouse 0 6 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 -6 W ; Mouse 0 -14 W ; Mouse -2 -6 W ; Mouse -6 -3 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -30 13 W ; Mouse -14 7 ; Mouse -1 2 ; Mouse -12 29 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -5 17 ; Mouse -3 12 ; Mouse 5 1 ; Mouse 14 0 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; Mouse -1 0 MouseLeft ; Mouse 2 -1 ; Mouse 0 6 <|action_end|>
```

## demonstration_optimization_000053

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220218-040456` |
| 图片帧 | `[138, 142, 146, 150]` |
| 目标动作区间 | `[138, 154]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 138**

![demonstration_optimization_000053 frame 138](images/demonstration_optimization_000053_00.jpg)

**图 2，帧 142**

![demonstration_optimization_000053 frame 142](images/demonstration_optimization_000053_01.jpg)

**图 3，帧 146**

![demonstration_optimization_000053 frame 146](images/demonstration_optimization_000053_02.jpg)

**图 4，帧 150**

![demonstration_optimization_000053 frame 150](images/demonstration_optimization_000053_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -20 0 W space ; Mouse -20 0 W space ; Mouse -64 -10 W space ; Mouse -54 -8 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -18 -4 W space ; Mouse -10 -8 W space ; Mouse -2 -2 W space ; W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -20 0 W space ; Mouse -20 0 W space ; Mouse -64 -10 W space ; Mouse -54 -8 W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -18 -4 W space ; Mouse -10 -8 W space ; Mouse -2 -2 W space ; W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W space ctrl <|action_end|>
```

## demonstration_optimization_000054

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `thirsty-lavender-koala-f153ac423f61-20220125-194043` |
| 图片帧 | `[18126, 18130, 18134, 18138]` |
| 目标动作区间 | `[18126, 18142]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 18126**

![demonstration_optimization_000054 frame 18126](images/demonstration_optimization_000054_00.jpg)

**图 2，帧 18130**

![demonstration_optimization_000054 frame 18130](images/demonstration_optimization_000054_01.jpg)

**图 3，帧 18134**

![demonstration_optimization_000054 frame 18134](images/demonstration_optimization_000054_02.jpg)

**图 4，帧 18138**

![demonstration_optimization_000054 frame 18138](images/demonstration_optimization_000054_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 24 -9 ; Mouse 170 -63 ; Mouse 244 -30 ; Mouse 175 -9 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 75 5 ; Mouse 15 7 ; Mouse -8 -1 W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; Mouse -6 7 W ; Mouse -20 7 W ; Mouse -24 10 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -28 10 W ; Mouse -32 16 W ; Mouse -25 8 W ; Mouse -13 4 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 24 -9 ; Mouse 170 -63 ; Mouse 244 -30 ; Mouse 175 -9 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 75 5 ; Mouse 15 7 ; Mouse -8 -1 W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; Mouse -6 7 W ; Mouse -20 7 W ; Mouse -24 10 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -28 10 W ; Mouse -32 16 W ; Mouse -25 8 W ; Mouse -13 4 W <|action_end|>
```

## demonstration_optimization_000055

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20211226-164019` |
| 图片帧 | `[17258, 17262, 17266, 17270]` |
| 目标动作区间 | `[17258, 17274]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 17258**

![demonstration_optimization_000055 frame 17258](images/demonstration_optimization_000055_00.jpg)

**图 2，帧 17262**

![demonstration_optimization_000055 frame 17262](images/demonstration_optimization_000055_01.jpg)

**图 3，帧 17266**

![demonstration_optimization_000055 frame 17266](images/demonstration_optimization_000055_02.jpg)

**图 4，帧 17270**

![demonstration_optimization_000055 frame 17270](images/demonstration_optimization_000055_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -23 -34 ; Mouse -13 -15 W ; Mouse -1 1 W ; Mouse -24 27 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -11 20 W D ; Mouse -3 9 W D ; Mouse -4 8 W D ; Mouse -9 15 W D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -5 7 W D ; Mouse -8 7 W ; Mouse -6 5 W ; Mouse -4 1 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -6 3 W ; Mouse -9 5 W ; Mouse -5 3 W ; Mouse -21 12 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -23 -34 ; Mouse -13 -15 W ; Mouse -1 1 W ; Mouse -24 27 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -11 20 W D ; Mouse -3 9 W D ; Mouse -4 8 W D ; Mouse -9 15 W D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -5 7 W D ; Mouse -8 7 W ; Mouse -6 5 W ; Mouse -4 1 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -6 3 W ; Mouse -9 5 W ; Mouse -5 3 W ; Mouse -21 12 W <|action_end|>
```

## demonstration_optimization_000056

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-50d9bd30442d-20220118-090121` |
| 图片帧 | `[335, 339, 343, 347]` |
| 目标动作区间 | `[335, 351]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 335**

![demonstration_optimization_000056 frame 335](images/demonstration_optimization_000056_00.jpg)

**图 2，帧 339**

![demonstration_optimization_000056 frame 339](images/demonstration_optimization_000056_01.jpg)

**图 3，帧 343**

![demonstration_optimization_000056 frame 343](images/demonstration_optimization_000056_02.jpg)

**图 4，帧 347**

![demonstration_optimization_000056 frame 347](images/demonstration_optimization_000056_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -6 -5 W space ; W space ; Mouse -3 -3 W space ; Mouse -10 -1 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -16 6 W space ctrl ; Mouse -7 6 W space ctrl ; Mouse -3 10 W space ctrl ; Mouse -10 22 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -17 29 W space ; Mouse -8 15 W space ; Mouse -7 15 W space ; Mouse 1 7 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 22 12 W space ctrl ; Mouse 9 0 W space ctrl ; Mouse 1 0 W space ctrl ; W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -6 -5 W space ; W space ; Mouse -3 -3 W space ; Mouse -10 -1 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -16 6 W space ctrl ; Mouse -7 6 W space ctrl ; Mouse -3 10 W space ctrl ; Mouse -10 22 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -17 29 W space ; Mouse -8 15 W space ; Mouse -7 15 W space ; Mouse 1 7 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 22 12 W space ctrl ; Mouse 9 0 W space ctrl ; Mouse 1 0 W space ctrl ; W space ctrl <|action_end|>
```

## demonstration_optimization_000057

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-f80804d55877-20211226-173217` |
| 图片帧 | `[11639, 11643, 11647, 11651]` |
| 目标动作区间 | `[11639, 11655]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 11639**

![demonstration_optimization_000057 frame 11639](images/demonstration_optimization_000057_00.jpg)

**图 2，帧 11643**

![demonstration_optimization_000057 frame 11643](images/demonstration_optimization_000057_01.jpg)

**图 3，帧 11647**

![demonstration_optimization_000057 frame 11647](images/demonstration_optimization_000057_02.jpg)

**图 4，帧 11651**

![demonstration_optimization_000057 frame 11651](images/demonstration_optimization_000057_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 64 7 MouseLeft ; Mouse 141 -15 MouseLeft ; Mouse 229 0 ; Mouse 74 0 D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 0 W D ; Mouse 1 -1 W D ; W D MouseLeft ; W D MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -15 27 W MouseLeft ; Mouse -11 28 MouseLeft ; Mouse 6 26 MouseLeft ; Mouse 25 27 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 22 9 MouseLeft ; Mouse 82 12 MouseLeft ; Mouse 67 -1 D MouseLeft ; Mouse 41 -12 D MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 64 7 MouseLeft ; Mouse 141 -15 MouseLeft ; Mouse 229 0 ; Mouse 74 0 D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 0 W D ; Mouse 1 -1 W D ; W D MouseLeft ; W D MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -15 27 W MouseLeft ; Mouse -11 28 MouseLeft ; Mouse 6 26 MouseLeft ; Mouse 25 27 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 22 9 MouseLeft ; Mouse 82 12 MouseLeft ; Mouse 67 -1 D MouseLeft ; Mouse 41 -12 D MouseLeft <|action_end|>
```

## demonstration_optimization_000058

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-02611186c812-20220107-153927` |
| 图片帧 | `[2109, 2113, 2117, 2121]` |
| 目标动作区间 | `[2109, 2125]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2109**

![demonstration_optimization_000058 frame 2109](images/demonstration_optimization_000058_00.jpg)

**图 2，帧 2113**

![demonstration_optimization_000058 frame 2113](images/demonstration_optimization_000058_01.jpg)

**图 3，帧 2117**

![demonstration_optimization_000058 frame 2117](images/demonstration_optimization_000058_02.jpg)

**图 4，帧 2121**

![demonstration_optimization_000058 frame 2121](images/demonstration_optimization_000058_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 6 6 W ; Mouse 0 2 W ; Mouse -1 13 ; Mouse -2 5 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 ; Mouse -22 -17 MouseRight ; Mouse -3 -3 MouseRight ; Mouse 2 7 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 15 12 ; Mouse 26 8 ; Mouse 16 8 ; Mouse 7 3 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 2 1 MouseRight ; Mouse 0 -7 MouseRight ; Mouse -2 -16 ; Mouse -4 -28 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 6 6 W ; Mouse 0 2 W ; Mouse -1 13 ; Mouse -2 5 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 ; Mouse -22 -17 MouseRight ; Mouse -3 -3 MouseRight ; Mouse 2 7 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 15 12 ; Mouse 26 8 ; Mouse 16 8 ; Mouse 7 3 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 2 1 MouseRight ; Mouse 0 -7 MouseRight ; Mouse -2 -16 ; Mouse -4 -28 <|action_end|>
```

## demonstration_optimization_000059

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `thirsty-lavender-koala-b343a535c597-20220224-203207` |
| 图片帧 | `[3958, 3962, 3966, 3970]` |
| 目标动作区间 | `[3958, 3974]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3958**

![demonstration_optimization_000059 frame 3958](images/demonstration_optimization_000059_00.jpg)

**图 2，帧 3962**

![demonstration_optimization_000059 frame 3962](images/demonstration_optimization_000059_01.jpg)

**图 3，帧 3966**

![demonstration_optimization_000059 frame 3966](images/demonstration_optimization_000059_02.jpg)

**图 4，帧 3970**

![demonstration_optimization_000059 frame 3970](images/demonstration_optimization_000059_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -34 -9 W ; Mouse -12 -8 W ; Mouse -4 -8 W ; Mouse -7 -9 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -6 -5 W ; Mouse -5 -4 W ; Mouse -5 -5 W ; Mouse -5 -6 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -7 -6 W ; Mouse -21 -6 W ; Mouse -18 0 W ; Mouse -15 -1 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -6 -2 W ; Mouse -15 -2 W ; Mouse -19 0 W ; Mouse -4 3 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -34 -9 W ; Mouse -12 -8 W ; Mouse -4 -8 W ; Mouse -7 -9 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -6 -5 W ; Mouse -5 -4 W ; Mouse -5 -5 W ; Mouse -5 -6 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -7 -6 W ; Mouse -21 -6 W ; Mouse -18 0 W ; Mouse -15 -1 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -6 -2 W ; Mouse -15 -2 W ; Mouse -19 0 W ; Mouse -4 3 W <|action_end|>
```

## demonstration_optimization_000060

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-1bf8d17bfa41-20220110-065008` |
| 图片帧 | `[90, 94, 98, 102]` |
| 目标动作区间 | `[90, 106]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 90**

![demonstration_optimization_000060 frame 90](images/demonstration_optimization_000060_00.jpg)

**图 2，帧 94**

![demonstration_optimization_000060 frame 94](images/demonstration_optimization_000060_01.jpg)

**图 3，帧 98**

![demonstration_optimization_000060 frame 98](images/demonstration_optimization_000060_02.jpg)

**图 4，帧 102**

![demonstration_optimization_000060 frame 102](images/demonstration_optimization_000060_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; Mouse -1 -1 MouseLeft ; Mouse 0 -1 MouseLeft ; Mouse 0 -3 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 -5 ; Mouse 1 -3 ; 3 ; Mouse -1 14 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 0 1 MouseRight ; MouseRight ; Mouse 7 -16 MouseRight ; Mouse 22 -31 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 30 -68 ; Mouse 34 -85 ; Mouse 15 -49 ; Mouse 10 -60 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; Mouse -1 -1 MouseLeft ; Mouse 0 -1 MouseLeft ; Mouse 0 -3 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 -5 ; Mouse 1 -3 ; 3 ; Mouse -1 14 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 0 1 MouseRight ; MouseRight ; Mouse 7 -16 MouseRight ; Mouse 22 -31 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 30 -68 ; Mouse 34 -85 ; Mouse 15 -49 ; Mouse 10 -60 <|action_end|>
```

## demonstration_optimization_000061

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-92fbe372b030-20220119-103310` |
| 图片帧 | `[729, 733, 737, 741]` |
| 目标动作区间 | `[729, 745]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 729**

![demonstration_optimization_000061 frame 729](images/demonstration_optimization_000061_00.jpg)

**图 2，帧 733**

![demonstration_optimization_000061 frame 733](images/demonstration_optimization_000061_01.jpg)

**图 3，帧 737**

![demonstration_optimization_000061 frame 737](images/demonstration_optimization_000061_02.jpg)

**图 4，帧 741**

![demonstration_optimization_000061 frame 741](images/demonstration_optimization_000061_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 -10 shift ; Mouse -26 -20 shift ; Mouse -28 -18 shift ; Mouse -10 -14 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -2 -9 shift ; Mouse 0 -8 shift ; Mouse -1 -2 shift ; Mouse -2 -2 shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -3 -6 shift ; Mouse -2 -4 shift ; Mouse -2 -1 shift ; Mouse 0 -1 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift ; shift ; shift ; shift MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 -10 shift ; Mouse -26 -20 shift ; Mouse -28 -18 shift ; Mouse -10 -14 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -2 -9 shift ; Mouse 0 -8 shift ; Mouse -1 -2 shift ; Mouse -2 -2 shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -3 -6 shift ; Mouse -2 -4 shift ; Mouse -2 -1 shift ; Mouse 0 -1 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift ; shift ; shift ; shift MouseLeft <|action_end|>
```

## demonstration_optimization_000062

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220214-144020` |
| 图片帧 | `[11542, 11546, 11550, 11554]` |
| 目标动作区间 | `[11542, 11558]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 11542**

![demonstration_optimization_000062 frame 11542](images/demonstration_optimization_000062_00.jpg)

**图 2，帧 11546**

![demonstration_optimization_000062 frame 11546](images/demonstration_optimization_000062_01.jpg)

**图 3，帧 11550**

![demonstration_optimization_000062 frame 11550](images/demonstration_optimization_000062_02.jpg)

**图 4，帧 11554**

![demonstration_optimization_000062 frame 11554](images/demonstration_optimization_000062_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -62 20 D space ; Mouse -66 20 D space ; Mouse -40 10 D space ; Mouse -4 2 D space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -20 4 D 4 ; Mouse -14 2 D ; Mouse -34 0 D 5 ; Mouse -48 4 D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -54 4 ; Mouse -18 0 ; Mouse -26 6 ; 4 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; 3 ; Mouse -8 -5 ; Mouse -34 -26 ; Mouse -38 -42 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -62 20 D space ; Mouse -66 20 D space ; Mouse -40 10 D space ; Mouse -4 2 D space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -20 4 D 4 ; Mouse -14 2 D ; Mouse -34 0 D 5 ; Mouse -48 4 D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -54 4 ; Mouse -18 0 ; Mouse -26 6 ; 4 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; 3 ; Mouse -8 -5 ; Mouse -34 -26 ; Mouse -38 -42 <|action_end|>
```

## demonstration_optimization_000063

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `tasty-brass-devil-a47f39f57c24-20220305-195728` |
| 图片帧 | `[4419, 4423, 4427, 4431]` |
| 目标动作区间 | `[4419, 4435]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4419**

![demonstration_optimization_000063 frame 4419](images/demonstration_optimization_000063_00.jpg)

**图 2，帧 4423**

![demonstration_optimization_000063 frame 4423](images/demonstration_optimization_000063_01.jpg)

**图 3，帧 4427**

![demonstration_optimization_000063 frame 4427](images/demonstration_optimization_000063_02.jpg)

**图 4，帧 4431**

![demonstration_optimization_000063 frame 4431](images/demonstration_optimization_000063_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 15 -3 W A MouseLeft ; Mouse 25 -5 A MouseLeft ; Mouse 17 -6 MouseLeft ; Mouse 26 -9 MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 23 -8 MouseLeft ; Mouse 13 -6 MouseLeft ; Mouse 1 -1 MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -4 0 MouseLeft ; Mouse -61 0 MouseLeft ; Mouse -92 0 MouseLeft ; Mouse -12 0 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 15 -3 W A MouseLeft ; Mouse 25 -5 A MouseLeft ; Mouse 17 -6 MouseLeft ; Mouse 26 -9 MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 23 -8 MouseLeft ; Mouse 13 -6 MouseLeft ; Mouse 1 -1 MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -4 0 MouseLeft ; Mouse -61 0 MouseLeft ; Mouse -92 0 MouseLeft ; Mouse -12 0 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000064

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220226-162818` |
| 图片帧 | `[6363, 6367, 6371, 6375]` |
| 目标动作区间 | `[6363, 6379]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6363**

![demonstration_optimization_000064 frame 6363](images/demonstration_optimization_000064_00.jpg)

**图 2，帧 6367**

![demonstration_optimization_000064 frame 6367](images/demonstration_optimization_000064_01.jpg)

**图 3，帧 6371**

![demonstration_optimization_000064 frame 6371](images/demonstration_optimization_000064_02.jpg)

**图 4，帧 6375**

![demonstration_optimization_000064 frame 6375](images/demonstration_optimization_000064_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse -11 23 D MouseLeft ; Mouse -20 45 D MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -23 82 W D MouseLeft ; Mouse -17 32 W D MouseLeft ; Mouse -8 17 W MouseLeft ; Mouse -3 6 W MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse -11 23 D MouseLeft ; Mouse -20 45 D MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -23 82 W D MouseLeft ; Mouse -17 32 W D MouseLeft ; Mouse -8 17 W MouseLeft ; Mouse -3 6 W MouseLeft <|action_end|>
```

## demonstration_optimization_000065

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220228-081512` |
| 图片帧 | `[9846, 9850, 9854, 9858]` |
| 目标动作区间 | `[9846, 9862]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9846**

![demonstration_optimization_000065 frame 9846](images/demonstration_optimization_000065_00.jpg)

**图 2，帧 9850**

![demonstration_optimization_000065 frame 9850](images/demonstration_optimization_000065_01.jpg)

**图 3，帧 9854**

![demonstration_optimization_000065 frame 9854](images/demonstration_optimization_000065_02.jpg)

**图 4，帧 9858**

![demonstration_optimization_000065 frame 9858](images/demonstration_optimization_000065_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse 0 1 W ctrl ; Mouse 4 9 W ctrl ; Mouse 10 13 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 10 18 W ctrl ; Mouse 9 17 W ctrl ; Mouse 9 26 W ctrl ; Mouse 7 49 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 3 32 W space ctrl ; Mouse 5 21 W space ctrl ; Mouse 10 34 W space ctrl ; Mouse 21 43 ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 4 12 ctrl ; Mouse 6 16 ctrl ; Mouse 5 9 ctrl ; Mouse 5 7 S <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse 0 1 W ctrl ; Mouse 4 9 W ctrl ; Mouse 10 13 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 10 18 W ctrl ; Mouse 9 17 W ctrl ; Mouse 9 26 W ctrl ; Mouse 7 49 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 3 32 W space ctrl ; Mouse 5 21 W space ctrl ; Mouse 10 34 W space ctrl ; Mouse 21 43 ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 4 12 ctrl ; Mouse 6 16 ctrl ; Mouse 5 9 ctrl ; Mouse 5 7 S <|action_end|>
```

## demonstration_optimization_000066

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-c7816efe5449-20220123-040410` |
| 图片帧 | `[5744, 5748, 5752, 5756]` |
| 目标动作区间 | `[5744, 5760]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5744**

![demonstration_optimization_000066 frame 5744](images/demonstration_optimization_000066_00.jpg)

**图 2，帧 5748**

![demonstration_optimization_000066 frame 5748](images/demonstration_optimization_000066_01.jpg)

**图 3，帧 5752**

![demonstration_optimization_000066 frame 5752](images/demonstration_optimization_000066_02.jpg)

**图 4，帧 5756**

![demonstration_optimization_000066 frame 5756](images/demonstration_optimization_000066_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -30 -4 E ; Mouse -84 0 MouseLeft ; Mouse -60 0 MouseLeft ; Mouse -20 0 MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -2 -2 MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -30 -4 E ; Mouse -84 0 MouseLeft ; Mouse -60 0 MouseLeft ; Mouse -20 0 MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -2 -2 MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000067

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `tasty-brass-devil-11af2aaacde4-20220304-012055` |
| 图片帧 | `[9913, 9917, 9921, 9925]` |
| 目标动作区间 | `[9913, 9929]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9913**

![demonstration_optimization_000067 frame 9913](images/demonstration_optimization_000067_00.jpg)

**图 2，帧 9917**

![demonstration_optimization_000067 frame 9917](images/demonstration_optimization_000067_01.jpg)

**图 3，帧 9921**

![demonstration_optimization_000067 frame 9921](images/demonstration_optimization_000067_02.jpg)

**图 4，帧 9925**

![demonstration_optimization_000067 frame 9925](images/demonstration_optimization_000067_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift MouseLeft ; W shift MouseLeft ; Mouse -1 -1 W shift MouseLeft ; Mouse -2 -10 W shift MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 -5 W shift MouseLeft ; Mouse -4 -32 W shift MouseLeft ; Mouse -9 -46 shift MouseLeft ; Mouse -3 -35 shift MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift MouseLeft ; W shift MouseLeft ; Mouse -1 -1 W shift MouseLeft ; Mouse -2 -10 W shift MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 -5 W shift MouseLeft ; Mouse -4 -32 W shift MouseLeft ; Mouse -9 -46 shift MouseLeft ; Mouse -3 -35 shift MouseLeft <|action_end|>
```

## demonstration_optimization_000068

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `tasty-brass-devil-f153ac423f61-20220208-110347` |
| 图片帧 | `[16176, 16180, 16184, 16188]` |
| 目标动作区间 | `[16176, 16192]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 16176**

![demonstration_optimization_000068 frame 16176](images/demonstration_optimization_000068_00.jpg)

**图 2，帧 16180**

![demonstration_optimization_000068 frame 16180](images/demonstration_optimization_000068_01.jpg)

**图 3，帧 16184**

![demonstration_optimization_000068 frame 16184](images/demonstration_optimization_000068_02.jpg)

**图 4，帧 16188**

![demonstration_optimization_000068 frame 16188](images/demonstration_optimization_000068_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 9 W ; Mouse -4 13 W ; Mouse -4 23 W ; Mouse 5 18 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 17 15 W ; Mouse 6 1 W ; Mouse 6 2 W ; Mouse 5 0 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 2 0 W ; Mouse 2 0 W ; W ; Mouse 4 0 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 3 -2 ; Mouse 8 -3 ; Mouse 6 -1 ; Mouse 5 0 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 9 W ; Mouse -4 13 W ; Mouse -4 23 W ; Mouse 5 18 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 17 15 W ; Mouse 6 1 W ; Mouse 6 2 W ; Mouse 5 0 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 2 0 W ; Mouse 2 0 W ; W ; Mouse 4 0 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 3 -2 ; Mouse 8 -3 ; Mouse 6 -1 ; Mouse 5 0 <|action_end|>
```

## demonstration_optimization_000069

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `wiggy-aquamarine-tapir-f153ac423f61-20220203-182709` |
| 图片帧 | `[6656, 6660, 6664, 6668]` |
| 目标动作区间 | `[6656, 6672]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6656**

![demonstration_optimization_000069 frame 6656](images/demonstration_optimization_000069_00.jpg)

**图 2，帧 6660**

![demonstration_optimization_000069 frame 6660](images/demonstration_optimization_000069_01.jpg)

**图 3，帧 6664**

![demonstration_optimization_000069 frame 6664](images/demonstration_optimization_000069_02.jpg)

**图 4，帧 6668**

![demonstration_optimization_000069 frame 6668](images/demonstration_optimization_000069_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -4 2 W space ; Mouse -23 3 W ; Mouse -24 0 W ; Mouse -10 0 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 W ; Mouse -1 1 W space ; Mouse 0 3 W space ; Mouse -2 11 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -2 9 W space ; Mouse -1 2 W MouseRight ; MouseRight ; Mouse 1 -13 2 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 -16 2 ; Mouse -6 -35 2 ; Mouse 5 -10 ; Mouse 2 5 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -4 2 W space ; Mouse -23 3 W ; Mouse -24 0 W ; Mouse -10 0 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 W ; Mouse -1 1 W space ; Mouse 0 3 W space ; Mouse -2 11 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -2 9 W space ; Mouse -1 2 W MouseRight ; MouseRight ; Mouse 1 -13 2 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 -16 2 ; Mouse -6 -35 2 ; Mouse 5 -10 ; Mouse 2 5 <|action_end|>
```

## demonstration_optimization_000070

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `wiggy-aquamarine-tapir-8bafa56b14f5-20220214-001357` |
| 图片帧 | `[1862, 1866, 1870, 1874]` |
| 目标动作区间 | `[1862, 1878]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1862**

![demonstration_optimization_000070 frame 1862](images/demonstration_optimization_000070_00.jpg)

**图 2，帧 1866**

![demonstration_optimization_000070 frame 1866](images/demonstration_optimization_000070_01.jpg)

**图 3，帧 1870**

![demonstration_optimization_000070 frame 1870](images/demonstration_optimization_000070_02.jpg)

**图 4，帧 1874**

![demonstration_optimization_000070 frame 1874](images/demonstration_optimization_000070_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -11 122 ; Mouse -6 97 W ; Mouse -7 46 W ; Mouse -5 23 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -6 33 W ; Mouse -4 35 W ; Mouse 0 14 W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; Mouse 2 0 ; Mouse 25 -4 ; Mouse 11 -3 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 3 -8 ; Mouse -41 -39 ; Mouse -30 -17 ; Mouse -3 -4 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -11 122 ; Mouse -6 97 W ; Mouse -7 46 W ; Mouse -5 23 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -6 33 W ; Mouse -4 35 W ; Mouse 0 14 W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; Mouse 2 0 ; Mouse 25 -4 ; Mouse 11 -3 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 3 -8 ; Mouse -41 -39 ; Mouse -30 -17 ; Mouse -3 -4 <|action_end|>
```

## demonstration_optimization_000071

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-3074e7f751e9-20220123-194843` |
| 图片帧 | `[6940, 6944, 6948, 6952]` |
| 目标动作区间 | `[6940, 6956]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6940**

![demonstration_optimization_000071 frame 6940](images/demonstration_optimization_000071_00.jpg)

**图 2，帧 6944**

![demonstration_optimization_000071 frame 6944](images/demonstration_optimization_000071_01.jpg)

**图 3，帧 6948**

![demonstration_optimization_000071 frame 6948](images/demonstration_optimization_000071_02.jpg)

**图 4，帧 6952**

![demonstration_optimization_000071 frame 6952](images/demonstration_optimization_000071_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -51 -8 ; Mouse -8 -4 ; Mouse -1 -1 MouseRight ; MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseRight ; D ; D ; Mouse -60 -33 D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -86 -33 ; Mouse -45 -16 MouseRight ; MouseRight ; MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseRight ; Mouse 3 -1 ; Mouse 11 -1 ; Mouse 2 0 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -51 -8 ; Mouse -8 -4 ; Mouse -1 -1 MouseRight ; MouseRight <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseRight ; D ; D ; Mouse -60 -33 D <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -86 -33 ; Mouse -45 -16 MouseRight ; MouseRight ; MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseRight ; Mouse 3 -1 ; Mouse 11 -1 ; Mouse 2 0 <|action_end|>
```

## demonstration_optimization_000072

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-d00cdb4b06b3-20220203-022749` |
| 图片帧 | `[3, 7, 11, 15]` |
| 目标动作区间 | `[3, 19]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3**

![demonstration_optimization_000072 frame 3](images/demonstration_optimization_000072_00.jpg)

**图 2，帧 7**

![demonstration_optimization_000072 frame 7](images/demonstration_optimization_000072_01.jpg)

**图 3，帧 11**

![demonstration_optimization_000072 frame 11](images/demonstration_optimization_000072_02.jpg)

**图 4，帧 15**

![demonstration_optimization_000072 frame 15](images/demonstration_optimization_000072_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -10 -18 W D ; Mouse -15 -15 W D ; Mouse -16 -9 W D ; Mouse -17 -3 W D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -31 -1 W D ; Mouse -56 -1 W ; Mouse -71 0 W ; Mouse -58 0 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -30 2 W ; Mouse -9 1 W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; Mouse 0 1 W D <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -10 -18 W D ; Mouse -15 -15 W D ; Mouse -16 -9 W D ; Mouse -17 -3 W D <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -31 -1 W D ; Mouse -56 -1 W ; Mouse -71 0 W ; Mouse -58 0 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -30 2 W ; Mouse -9 1 W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; Mouse 0 1 W D <|action_end|>
```

## demonstration_optimization_000073

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-3610baa2fb13-20220105-114427` |
| 图片帧 | `[7718, 7722, 7726, 7730]` |
| 目标动作区间 | `[7718, 7734]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7718**

![demonstration_optimization_000073 frame 7718](images/demonstration_optimization_000073_00.jpg)

**图 2，帧 7722**

![demonstration_optimization_000073 frame 7722](images/demonstration_optimization_000073_01.jpg)

**图 3，帧 7726**

![demonstration_optimization_000073 frame 7726](images/demonstration_optimization_000073_02.jpg)

**图 4，帧 7730**

![demonstration_optimization_000073 frame 7730](images/demonstration_optimization_000073_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -1 12 MouseLeft ; Mouse -1 8 MouseLeft ; Mouse 0 2 MouseLeft ; Mouse -2 7 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -1 12 MouseLeft ; Mouse -1 8 MouseLeft ; Mouse 0 2 MouseLeft ; Mouse -2 7 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000074

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `hazy-thistle-chipmunk-bd629529d5e9-20220201-012537` |
| 图片帧 | `[2499, 2503, 2507, 2511]` |
| 目标动作区间 | `[2499, 2515]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2499**

![demonstration_optimization_000074 frame 2499](images/demonstration_optimization_000074_00.jpg)

**图 2，帧 2503**

![demonstration_optimization_000074 frame 2503](images/demonstration_optimization_000074_01.jpg)

**图 3，帧 2507**

![demonstration_optimization_000074 frame 2507](images/demonstration_optimization_000074_02.jpg)

**图 4，帧 2511**

![demonstration_optimization_000074 frame 2511](images/demonstration_optimization_000074_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -9 -2 W space ctrl ; Mouse -10 -5 W space ctrl ; Mouse -7 -6 W space ctrl ; Mouse -1 0 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 -1 W space ctrl ; W space ctrl ; Mouse 4 0 W space ctrl ; Mouse 10 0 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 14 0 W space ctrl ; Mouse 18 3 W space ctrl ; Mouse 5 2 W space ctrl ; W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 0 W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -9 -2 W space ctrl ; Mouse -10 -5 W space ctrl ; Mouse -7 -6 W space ctrl ; Mouse -1 0 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 -1 W space ctrl ; W space ctrl ; Mouse 4 0 W space ctrl ; Mouse 10 0 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 14 0 W space ctrl ; Mouse 18 3 W space ctrl ; Mouse 5 2 W space ctrl ; W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 1 0 W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

## demonstration_optimization_000075

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220211-164216` |
| 图片帧 | `[3987, 3991, 3995, 3999]` |
| 目标动作区间 | `[3987, 4003]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3987**

![demonstration_optimization_000075 frame 3987](images/demonstration_optimization_000075_00.jpg)

**图 2，帧 3991**

![demonstration_optimization_000075 frame 3991](images/demonstration_optimization_000075_01.jpg)

**图 3，帧 3995**

![demonstration_optimization_000075 frame 3995](images/demonstration_optimization_000075_02.jpg)

**图 4，帧 3999**

![demonstration_optimization_000075 frame 3999](images/demonstration_optimization_000075_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ; W A ; W A ; Mouse 38 -2 W A <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ; W A ; W A ; Mouse 38 -2 W A <|action_end|>
```

## demonstration_optimization_000076

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220210-043455` |
| 图片帧 | `[10898, 10902, 10906, 10910]` |
| 目标动作区间 | `[10898, 10914]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10898**

![demonstration_optimization_000076 frame 10898](images/demonstration_optimization_000076_00.jpg)

**图 2，帧 10902**

![demonstration_optimization_000076 frame 10902](images/demonstration_optimization_000076_01.jpg)

**图 3，帧 10906**

![demonstration_optimization_000076 frame 10906](images/demonstration_optimization_000076_02.jpg)

**图 4，帧 10910**

![demonstration_optimization_000076 frame 10910](images/demonstration_optimization_000076_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W space ; Mouse -8 23 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -2 10 W space ; Mouse 0 2 W space ; W space ; Mouse 3 8 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 10 W ; Mouse -7 22 W ; Mouse -44 22 W ; Mouse -30 8 W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W space ; Mouse -8 23 W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -2 10 W space ; Mouse 0 2 W space ; W space ; Mouse 3 8 W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 10 W ; Mouse -7 22 W ; Mouse -44 22 W ; Mouse -30 8 W space <|action_end|>
```

## demonstration_optimization_000077

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-a487c0c9e81e-20220228-014500` |
| 图片帧 | `[3, 7, 11, 15]` |
| 目标动作区间 | `[3, 19]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3**

![demonstration_optimization_000077 frame 3](images/demonstration_optimization_000077_00.jpg)

**图 2，帧 7**

![demonstration_optimization_000077 frame 7](images/demonstration_optimization_000077_01.jpg)

**图 3，帧 11**

![demonstration_optimization_000077 frame 11](images/demonstration_optimization_000077_02.jpg)

**图 4，帧 15**

![demonstration_optimization_000077 frame 15](images/demonstration_optimization_000077_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -35 6 ; Mouse -231 -17 ; Mouse -166 -14 W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 16 -2 W ; Mouse 2 -2 W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 -1 W ; W ; W ; W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -35 6 ; Mouse -231 -17 ; Mouse -166 -14 W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 16 -2 W ; Mouse 2 -2 W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 -1 W ; W ; W ; W <|action_end|>
```

## demonstration_optimization_000078

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `whiny-ecru-cougar-42fde0b700d9-20220120-004754` |
| 图片帧 | `[2320, 2324, 2328, 2332]` |
| 目标动作区间 | `[2320, 2336]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2320**

![demonstration_optimization_000078 frame 2320](images/demonstration_optimization_000078_00.jpg)

**图 2，帧 2324**

![demonstration_optimization_000078 frame 2324](images/demonstration_optimization_000078_01.jpg)

**图 3，帧 2328**

![demonstration_optimization_000078 frame 2328](images/demonstration_optimization_000078_02.jpg)

**图 4，帧 2332**

![demonstration_optimization_000078 frame 2332](images/demonstration_optimization_000078_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -3 2 W space ; Mouse -3 3 W space ; Mouse -2 1 W space ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -1 0 W ; Mouse -2 1 W ; Mouse -1 1 W ; W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -3 2 W space ; Mouse -3 3 W space ; Mouse -2 1 W space ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -1 0 W ; Mouse -2 1 W ; Mouse -1 1 W ; W <|action_end|>
```

## demonstration_optimization_000079

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-cb0d36f9697e-20220206-055848` |
| 图片帧 | `[3483, 3487, 3491, 3495]` |
| 目标动作区间 | `[3483, 3499]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3483**

![demonstration_optimization_000079 frame 3483](images/demonstration_optimization_000079_00.jpg)

**图 2，帧 3487**

![demonstration_optimization_000079 frame 3487](images/demonstration_optimization_000079_01.jpg)

**图 3，帧 3491**

![demonstration_optimization_000079 frame 3491](images/demonstration_optimization_000079_02.jpg)

**图 4，帧 3495**

![demonstration_optimization_000079 frame 3495](images/demonstration_optimization_000079_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; shift ; Mouse 0 4 shift ; Mouse -1 10 shift ; Mouse -2 8 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 2 shift ; Mouse 0 3 shift ; Mouse 0 2 shift ; Mouse 0 1 shift MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift ; shift ; shift ; Mouse 1 0 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 6 1 shift ; Mouse 12 6 shift ; Mouse 3 0 shift MouseLeft ; shift <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift ; Mouse 0 4 shift ; Mouse -1 10 shift ; Mouse -2 8 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 2 shift ; Mouse 0 3 shift ; Mouse 0 2 shift ; Mouse 0 1 shift MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift ; shift ; shift ; Mouse 1 0 shift <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 6 1 shift ; Mouse 12 6 shift ; Mouse 3 0 shift MouseLeft ; shift <|action_end|>
```

## demonstration_optimization_000080

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player407-f153ac423f61-20211118-212048` |
| 图片帧 | `[9075, 9079, 9083, 9087]` |
| 目标动作区间 | `[9075, 9091]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9075**

![demonstration_optimization_000080 frame 9075](images/demonstration_optimization_000080_00.jpg)

**图 2，帧 9079**

![demonstration_optimization_000080 frame 9079](images/demonstration_optimization_000080_01.jpg)

**图 3，帧 9083**

![demonstration_optimization_000080 frame 9083](images/demonstration_optimization_000080_02.jpg)

**图 4，帧 9087**

![demonstration_optimization_000080 frame 9087](images/demonstration_optimization_000080_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -3 0 MouseLeft ; Mouse -10 0 MouseLeft ; Mouse -10 2 MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; Mouse -2 0 MouseLeft ; Mouse -5 2 MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse 0 1 MouseLeft ; Mouse -3 5 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -1 0 MouseLeft ; MouseLeft ; Mouse -5 2 MouseLeft ; Mouse -15 12 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 0 MouseLeft ; Mouse -10 0 MouseLeft ; Mouse -10 2 MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; Mouse -2 0 MouseLeft ; Mouse -5 2 MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse 0 1 MouseLeft ; Mouse -3 5 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -1 0 MouseLeft ; MouseLeft ; Mouse -5 2 MouseLeft ; Mouse -15 12 MouseLeft <|action_end|>
```

## demonstration_optimization_000081

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player635-f153ac423f61-20220120-133452` |
| 图片帧 | `[13650, 13654, 13658, 13662]` |
| 目标动作区间 | `[13650, 13666]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 13650**

![demonstration_optimization_000081 frame 13650](images/demonstration_optimization_000081_00.jpg)

**图 2，帧 13654**

![demonstration_optimization_000081 frame 13654](images/demonstration_optimization_000081_01.jpg)

**图 3，帧 13658**

![demonstration_optimization_000081 frame 13658](images/demonstration_optimization_000081_02.jpg)

**图 4，帧 13662**

![demonstration_optimization_000081 frame 13662](images/demonstration_optimization_000081_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 22 32 shift MouseLeft ; Mouse 108 55 shift MouseLeft ; Mouse 49 11 shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 9 1 shift MouseLeft ; Mouse 3 0 shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 22 32 shift MouseLeft ; Mouse 108 55 shift MouseLeft ; Mouse 49 11 shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 9 1 shift MouseLeft ; Mouse 3 0 shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## demonstration_optimization_000082

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-20f4188b8a75-20220225-061816` |
| 图片帧 | `[811, 815, 819, 823]` |
| 目标动作区间 | `[811, 827]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 811**

![demonstration_optimization_000082 frame 811](images/demonstration_optimization_000082_00.jpg)

**图 2，帧 815**

![demonstration_optimization_000082 frame 815](images/demonstration_optimization_000082_01.jpg)

**图 3，帧 819**

![demonstration_optimization_000082 frame 819](images/demonstration_optimization_000082_02.jpg)

**图 4，帧 823**

![demonstration_optimization_000082 frame 823](images/demonstration_optimization_000082_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ;  ; Mouse -3 4 ; Mouse 0 10 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 6 ; Mouse 2 6 ; Mouse 1 2 ;  <|action_end|>
```

动作块 3：

```text
<|action_start|> ;  ; Mouse 0 -2 ; Mouse 0 -2 ; Mouse 6 0 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 15 2 ; Mouse 14 -1 ; Mouse 4 -6 ; Mouse 0 -2 shift <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ;  ; Mouse -3 4 ; Mouse 0 10 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 6 ; Mouse 2 6 ; Mouse 1 2 ;  <|action_end|>
```

动作块 3：

```text
<|action_start|> ;  ; Mouse 0 -2 ; Mouse 0 -2 ; Mouse 6 0 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 15 2 ; Mouse 14 -1 ; Mouse 4 -6 ; Mouse 0 -2 shift <|action_end|>
```

## demonstration_optimization_000083

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-ae4b549ad589-20220206-165700` |
| 图片帧 | `[267, 271, 275, 279]` |
| 目标动作区间 | `[267, 283]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 267**

![demonstration_optimization_000083 frame 267](images/demonstration_optimization_000083_00.jpg)

**图 2，帧 271**

![demonstration_optimization_000083 frame 271](images/demonstration_optimization_000083_01.jpg)

**图 3，帧 275**

![demonstration_optimization_000083 frame 275](images/demonstration_optimization_000083_02.jpg)

**图 4，帧 279**

![demonstration_optimization_000083 frame 279](images/demonstration_optimization_000083_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 16 W ctrl ; Mouse -12 24 W ctrl ; Mouse -8 14 W ctrl ; Mouse -5 10 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 7 W ctrl ; Mouse 0 1 W ctrl ; W ctrl MouseLeft ; W ctrl MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ctrl MouseLeft ; W ctrl MouseLeft ; W ctrl ; W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl MouseLeft ; W ctrl MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 16 W ctrl ; Mouse -12 24 W ctrl ; Mouse -8 14 W ctrl ; Mouse -5 10 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 7 W ctrl ; Mouse 0 1 W ctrl ; W ctrl MouseLeft ; W ctrl MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ctrl MouseLeft ; W ctrl MouseLeft ; W ctrl ; W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl MouseLeft ; W ctrl MouseLeft <|action_end|>
```

## demonstration_optimization_000084

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20220307-110608` |
| 图片帧 | `[2000, 2004, 2008, 2012]` |
| 目标动作区间 | `[2000, 2016]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2000**

![demonstration_optimization_000084 frame 2000](images/demonstration_optimization_000084_00.jpg)

**图 2，帧 2004**

![demonstration_optimization_000084 frame 2004](images/demonstration_optimization_000084_01.jpg)

**图 3，帧 2008**

![demonstration_optimization_000084 frame 2008](images/demonstration_optimization_000084_02.jpg)

**图 4，帧 2012**

![demonstration_optimization_000084 frame 2012](images/demonstration_optimization_000084_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -39 86 ; Mouse -16 64 ; Mouse -5 25 ; Mouse -6 13 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 ; S ; S ; S <|action_end|>
```

动作块 3：

```text
<|action_start|> ; S ; S ; S A ; A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; A ; Mouse 1 -1 ; A ; Mouse 1 -2 A <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -39 86 ; Mouse -16 64 ; Mouse -5 25 ; Mouse -6 13 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 1 ; S ; S ; S <|action_end|>
```

动作块 3：

```text
<|action_start|> ; S ; S ; S A ; A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; A ; Mouse 1 -1 ; A ; Mouse 1 -2 A <|action_end|>
```

## demonstration_optimization_000085

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-23a9a14e250b-20220215-175646` |
| 图片帧 | `[3864, 3868, 3872, 3876]` |
| 目标动作区间 | `[3864, 3880]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3864**

![demonstration_optimization_000085 frame 3864](images/demonstration_optimization_000085_00.jpg)

**图 2，帧 3868**

![demonstration_optimization_000085 frame 3868](images/demonstration_optimization_000085_01.jpg)

**图 3，帧 3872**

![demonstration_optimization_000085 frame 3872](images/demonstration_optimization_000085_02.jpg)

**图 4，帧 3876**

![demonstration_optimization_000085 frame 3876](images/demonstration_optimization_000085_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -3 0 W A space ctrl ; W A space ctrl ; W space ctrl ; Mouse 5 -1 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 -1 W space ctrl ; Mouse 0 -1 W space ctrl ; Mouse 1 0 W space ctrl ; W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ctrl ; W space ; W space ; W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 0 W A space ctrl ; W A space ctrl ; W space ctrl ; Mouse 5 -1 W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 -1 W space ctrl ; Mouse 0 -1 W space ctrl ; Mouse 1 0 W space ctrl ; W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ctrl ; W space ; W space ; W space <|action_end|>
```

## demonstration_optimization_000086

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220126-001006` |
| 图片帧 | `[129, 133, 137, 141]` |
| 目标动作区间 | `[129, 145]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 129**

![demonstration_optimization_000086 frame 129](images/demonstration_optimization_000086_00.jpg)

**图 2，帧 133**

![demonstration_optimization_000086 frame 133](images/demonstration_optimization_000086_01.jpg)

**图 3，帧 137**

![demonstration_optimization_000086 frame 137](images/demonstration_optimization_000086_02.jpg)

**图 4，帧 141**

![demonstration_optimization_000086 frame 141](images/demonstration_optimization_000086_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; W ; W ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

## demonstration_optimization_000087

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `whiny-ecru-cougar-f247d314669f-20211227-235538` |
| 图片帧 | `[1572, 1576, 1580, 1584]` |
| 目标动作区间 | `[1572, 1588]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1572**

![demonstration_optimization_000087 frame 1572](images/demonstration_optimization_000087_00.jpg)

**图 2，帧 1576**

![demonstration_optimization_000087 frame 1576](images/demonstration_optimization_000087_01.jpg)

**图 3，帧 1580**

![demonstration_optimization_000087 frame 1580](images/demonstration_optimization_000087_02.jpg)

**图 4，帧 1584**

![demonstration_optimization_000087 frame 1584](images/demonstration_optimization_000087_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; Mouse 0 1 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 15 3 W ctrl ; Mouse 100 11 W ctrl ; Mouse 7 1 W D ctrl ; W D ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W D ctrl ; W D ctrl ; W ctrl ; W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -4 -2 W ctrl ; Mouse -13 -6 W ctrl ; Mouse -36 -7 W ctrl ; Mouse -31 -5 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; Mouse 0 1 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 15 3 W ctrl ; Mouse 100 11 W ctrl ; Mouse 7 1 W D ctrl ; W D ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W D ctrl ; W D ctrl ; W ctrl ; W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -4 -2 W ctrl ; Mouse -13 -6 W ctrl ; Mouse -36 -7 W ctrl ; Mouse -31 -5 W <|action_end|>
```

## demonstration_optimization_000088

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-2378cec9be47-20220114-092014` |
| 图片帧 | `[1276, 1280, 1284, 1288]` |
| 目标动作区间 | `[1276, 1292]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1276**

![demonstration_optimization_000088 frame 1276](images/demonstration_optimization_000088_00.jpg)

**图 2，帧 1280**

![demonstration_optimization_000088 frame 1280](images/demonstration_optimization_000088_01.jpg)

**图 3，帧 1284**

![demonstration_optimization_000088 frame 1284](images/demonstration_optimization_000088_02.jpg)

**图 4，帧 1288**

![demonstration_optimization_000088 frame 1288](images/demonstration_optimization_000088_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 30 0 W A ; Mouse 87 0 W A ; Mouse 132 -10 W A ; Mouse 98 -20 A <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 109 -21 ; Mouse 37 -6 ; Mouse 1 0 ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 3 1 MouseLeft ; Mouse 6 2 MouseLeft ; Mouse 11 5 MouseLeft ; Mouse 5 0 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 2 2 MouseLeft ; MouseLeft ; MouseLeft ; Mouse 3 1 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 30 0 W A ; Mouse 87 0 W A ; Mouse 132 -10 W A ; Mouse 98 -20 A <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 109 -21 ; Mouse 37 -6 ; Mouse 1 0 ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 3 1 MouseLeft ; Mouse 6 2 MouseLeft ; Mouse 11 5 MouseLeft ; Mouse 5 0 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 2 2 MouseLeft ; MouseLeft ; MouseLeft ; Mouse 3 1 MouseLeft <|action_end|>
```

## demonstration_optimization_000089

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220324-200023` |
| 图片帧 | `[2699, 2703, 2707, 2711]` |
| 目标动作区间 | `[2699, 2715]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2699**

![demonstration_optimization_000089 frame 2699](images/demonstration_optimization_000089_00.jpg)

**图 2，帧 2703**

![demonstration_optimization_000089 frame 2703](images/demonstration_optimization_000089_01.jpg)

**图 3，帧 2707**

![demonstration_optimization_000089 frame 2707](images/demonstration_optimization_000089_02.jpg)

**图 4，帧 2711**

![demonstration_optimization_000089 frame 2711](images/demonstration_optimization_000089_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

## demonstration_optimization_000090

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220218-110233` |
| 图片帧 | `[8687, 8691, 8695, 8699]` |
| 目标动作区间 | `[8687, 8703]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8687**

![demonstration_optimization_000090 frame 8687](images/demonstration_optimization_000090_00.jpg)

**图 2，帧 8691**

![demonstration_optimization_000090 frame 8691](images/demonstration_optimization_000090_01.jpg)

**图 3，帧 8695**

![demonstration_optimization_000090 frame 8695](images/demonstration_optimization_000090_02.jpg)

**图 4，帧 8699**

![demonstration_optimization_000090 frame 8699](images/demonstration_optimization_000090_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 0 -4 W ctrl ; Mouse 2 -2 W ; Mouse 3 -3 W ; Mouse 1 -3 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 -11 W ; Mouse 7 -24 W ; Mouse 6 -46 W ; Mouse -13 -77 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -19 -37 ; Mouse -4 -9 ; Mouse 0 -3 ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W space ; W space ; W space <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -4 W ctrl ; Mouse 2 -2 W ; Mouse 3 -3 W ; Mouse 1 -3 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 -11 W ; Mouse 7 -24 W ; Mouse 6 -46 W ; Mouse -13 -77 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -19 -37 ; Mouse -4 -9 ; Mouse 0 -3 ; W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W space ; W space ; W space <|action_end|>
```

## demonstration_optimization_000091

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220127-132116` |
| 图片帧 | `[869, 873, 877, 881]` |
| 目标动作区间 | `[869, 885]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 869**

![demonstration_optimization_000091 frame 869](images/demonstration_optimization_000091_00.jpg)

**图 2，帧 873**

![demonstration_optimization_000091 frame 873](images/demonstration_optimization_000091_01.jpg)

**图 3，帧 877**

![demonstration_optimization_000091 frame 877](images/demonstration_optimization_000091_02.jpg)

**图 4，帧 881**

![demonstration_optimization_000091 frame 881](images/demonstration_optimization_000091_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -3 3 W ; Mouse 2 13 W ; Mouse 11 20 ; Mouse 5 17 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 15 MouseLeft ; Mouse 0 12 MouseLeft ; Mouse -1 6 MouseLeft ; Mouse -13 4 W MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -49 -1 W ; Mouse -51 -2 W ; Mouse -16 -9 W ; Mouse 1 -17 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 -13 ; MouseLeft ; Mouse -1 -1 MouseLeft ; MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 3 W ; Mouse 2 13 W ; Mouse 11 20 ; Mouse 5 17 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 0 15 MouseLeft ; Mouse 0 12 MouseLeft ; Mouse -1 6 MouseLeft ; Mouse -13 4 W MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -49 -1 W ; Mouse -51 -2 W ; Mouse -16 -9 W ; Mouse 1 -17 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 -13 ; MouseLeft ; Mouse -1 -1 MouseLeft ; MouseLeft <|action_end|>
```

## demonstration_optimization_000092

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-28b3f886dc8d-20220228-144641` |
| 图片帧 | `[8618, 8622, 8626, 8630]` |
| 目标动作区间 | `[8618, 8634]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8618**

![demonstration_optimization_000092 frame 8618](images/demonstration_optimization_000092_00.jpg)

**图 2，帧 8622**

![demonstration_optimization_000092 frame 8622](images/demonstration_optimization_000092_01.jpg)

**图 3，帧 8626**

![demonstration_optimization_000092 frame 8626](images/demonstration_optimization_000092_02.jpg)

**图 4，帧 8630**

![demonstration_optimization_000092 frame 8630](images/demonstration_optimization_000092_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -1 -1 ; W ; Mouse -4 1 W ; Mouse -2 19 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 21 W ; Mouse 3 19 W ; Mouse 2 14 W A ; Mouse 5 11 W A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 8 10 W A ; Mouse 2 16 W ; Mouse 4 11 W ; Mouse 3 7 W A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 12 13 W A ; Mouse 21 20 W ; Mouse 69 49 W ; Mouse 34 8 W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 -1 ; W ; Mouse -4 1 W ; Mouse -2 19 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 1 21 W ; Mouse 3 19 W ; Mouse 2 14 W A ; Mouse 5 11 W A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 8 10 W A ; Mouse 2 16 W ; Mouse 4 11 W ; Mouse 3 7 W A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 12 13 W A ; Mouse 21 20 W ; Mouse 69 49 W ; Mouse 34 8 W <|action_end|>
```

## demonstration_optimization_000093

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-796f012807e6-20220130-115914` |
| 图片帧 | `[1871, 1875, 1879, 1883]` |
| 目标动作区间 | `[1871, 1887]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1871**

![demonstration_optimization_000093 frame 1871](images/demonstration_optimization_000093_00.jpg)

**图 2，帧 1875**

![demonstration_optimization_000093 frame 1875](images/demonstration_optimization_000093_01.jpg)

**图 3，帧 1879**

![demonstration_optimization_000093 frame 1879](images/demonstration_optimization_000093_02.jpg)

**图 4，帧 1883**

![demonstration_optimization_000093 frame 1883](images/demonstration_optimization_000093_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 -5 W space ctrl ; Mouse 15 -12 W space ctrl ; Mouse 4 -3 W space ctrl ; Mouse 2 0 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 0 W space ctrl ; Mouse 2 0 W space ctrl ; Mouse 1 0 W space ctrl ; Mouse 5 1 W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 9 4 W space ctrl ; Mouse 4 1 W space ctrl ; Mouse 1 1 W space ctrl ; Mouse 0 5 W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 6 -5 W space ctrl ; Mouse 15 -12 W space ctrl ; Mouse 4 -3 W space ctrl ; Mouse 2 0 W space ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 1 0 W space ctrl ; Mouse 2 0 W space ctrl ; Mouse 1 0 W space ctrl ; Mouse 5 1 W space ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 9 4 W space ctrl ; Mouse 4 1 W space ctrl ; Mouse 1 1 W space ctrl ; Mouse 0 5 W space ctrl <|action_end|>
```

## demonstration_optimization_000094

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `Player757-f153ac423f61-20211202-230024` |
| 图片帧 | `[10299, 10303, 10307, 10311]` |
| 目标动作区间 | `[10299, 10315]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10299**

![demonstration_optimization_000094 frame 10299](images/demonstration_optimization_000094_00.jpg)

**图 2，帧 10303**

![demonstration_optimization_000094 frame 10303](images/demonstration_optimization_000094_01.jpg)

**图 3，帧 10307**

![demonstration_optimization_000094 frame 10307](images/demonstration_optimization_000094_02.jpg)

**图 4，帧 10311**

![demonstration_optimization_000094 frame 10311](images/demonstration_optimization_000094_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; Mouse 13 2 W ; Mouse 7 20 W ; Mouse 106 -2 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 268 -6 W ; Mouse 243 3 W ; Mouse 85 7 W ; W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W ; Mouse 13 2 W ; Mouse 7 20 W ; Mouse 106 -2 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 268 -6 W ; Mouse 243 3 W ; Mouse 85 7 W ; W <|action_end|>
```

## demonstration_optimization_000095

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-181e27264bdb-20220307-094552` |
| 图片帧 | `[1079, 1083, 1087, 1091]` |
| 目标动作区间 | `[1079, 1095]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1079**

![demonstration_optimization_000095 frame 1079](images/demonstration_optimization_000095_00.jpg)

**图 2，帧 1083**

![demonstration_optimization_000095 frame 1083](images/demonstration_optimization_000095_01.jpg)

**图 3，帧 1087**

![demonstration_optimization_000095 frame 1087](images/demonstration_optimization_000095_02.jpg)

**图 4，帧 1091**

![demonstration_optimization_000095 frame 1091](images/demonstration_optimization_000095_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W D space ctrl ; Mouse -9 5 W D space ctrl ; Mouse -65 15 W D space ctrl ; Mouse -127 4 W D ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -190 -16 W A ctrl ; Mouse -181 -18 W A ctrl ; Mouse -181 -23 W A ctrl ; Mouse -124 -11 W A ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -23 -1 W A ctrl ; Mouse -10 2 W A ctrl ; Mouse 1 0 W A ctrl ; Mouse 2 -1 W A ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ctrl ; W A ctrl ; Mouse 0 1 W A ctrl ; Mouse 3 0 W A ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D space ctrl ; Mouse -9 5 W D space ctrl ; Mouse -65 15 W D space ctrl ; Mouse -127 4 W D ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -190 -16 W A ctrl ; Mouse -181 -18 W A ctrl ; Mouse -181 -23 W A ctrl ; Mouse -124 -11 W A ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -23 -1 W A ctrl ; Mouse -10 2 W A ctrl ; Mouse 1 0 W A ctrl ; Mouse 2 -1 W A ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W A ctrl ; W A ctrl ; Mouse 0 1 W A ctrl ; Mouse 3 0 W A ctrl <|action_end|>
```

## demonstration_optimization_000096

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `woozy-ruby-ostrich-f153ac423f61-20220111-154456` |
| 图片帧 | `[13995, 13999, 14003, 14007]` |
| 目标动作区间 | `[13995, 14011]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 13995**

![demonstration_optimization_000096 frame 13995](images/demonstration_optimization_000096_00.jpg)

**图 2，帧 13999**

![demonstration_optimization_000096 frame 13999](images/demonstration_optimization_000096_01.jpg)

**图 3，帧 14003**

![demonstration_optimization_000096 frame 14003](images/demonstration_optimization_000096_02.jpg)

**图 4，帧 14007**

![demonstration_optimization_000096 frame 14007](images/demonstration_optimization_000096_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -25 -11 W ; Mouse -20 -15 W ; Mouse -9 -8 W ; Mouse -6 -5 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 -5 W A ; Mouse 0 -1 W A ; W A ; Mouse -1 -4 W A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -1 -5 W ; Mouse 0 -5 W ; Mouse 0 -7 W ; Mouse -1 -2 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -25 -11 W ; Mouse -20 -15 W ; Mouse -9 -8 W ; Mouse -6 -5 W <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -4 -5 W A ; Mouse 0 -1 W A ; W A ; Mouse -1 -4 W A <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -1 -5 W ; Mouse 0 -5 W ; Mouse 0 -7 W ; Mouse -1 -2 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## demonstration_optimization_000097

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220202-125840` |
| 图片帧 | `[1035, 1039, 1043, 1047]` |
| 目标动作区间 | `[1035, 1051]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1035**

![demonstration_optimization_000097 frame 1035](images/demonstration_optimization_000097_00.jpg)

**图 2，帧 1039**

![demonstration_optimization_000097 frame 1039](images/demonstration_optimization_000097_01.jpg)

**图 3，帧 1043**

![demonstration_optimization_000097 frame 1043](images/demonstration_optimization_000097_02.jpg)

**图 4，帧 1047**

![demonstration_optimization_000097 frame 1047](images/demonstration_optimization_000097_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 6 -2 ; Mouse 7 -7 ; Mouse 4 -7 ; Mouse 0 -6 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -5 -22 ; Mouse -12 -29 W ; Mouse -4 -19 W ; Mouse 2 -17 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 4 -8 W A ; Mouse 14 -10 W A ; Mouse 7 -6 W A ; Mouse 9 -10 W A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 17 -13 W A ; Mouse 25 -15 W A ; Mouse 19 -10 W A ; Mouse 10 -6 W A <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 6 -2 ; Mouse 7 -7 ; Mouse 4 -7 ; Mouse 0 -6 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -5 -22 ; Mouse -12 -29 W ; Mouse -4 -19 W ; Mouse 2 -17 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 4 -8 W A ; Mouse 14 -10 W A ; Mouse 7 -6 W A ; Mouse 9 -10 W A <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 17 -13 W A ; Mouse 25 -15 W A ; Mouse 19 -10 W A ; Mouse 10 -6 W A <|action_end|>
```

## demonstration_optimization_000098

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `shabby-pink-molly-9295e2fa6c8b-20220125-151525` |
| 图片帧 | `[3999, 4003, 4007, 4011]` |
| 目标动作区间 | `[3999, 4015]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3999**

![demonstration_optimization_000098 frame 3999](images/demonstration_optimization_000098_00.jpg)

**图 2，帧 4003**

![demonstration_optimization_000098 frame 4003](images/demonstration_optimization_000098_01.jpg)

**图 3，帧 4007**

![demonstration_optimization_000098 frame 4007](images/demonstration_optimization_000098_02.jpg)

**图 4，帧 4011**

![demonstration_optimization_000098 frame 4011](images/demonstration_optimization_000098_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -34 8 shift ; Mouse -44 -2 shift ; Mouse -34 -5 shift ; Mouse -16 0 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 shift ; Mouse -24 -9 shift ; Mouse -27 -12 shift ; Mouse -4 -2 shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift ; shift MouseRight ; shift MouseRight ; shift MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift ; W shift ; W shift ; W shift <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -34 8 shift ; Mouse -44 -2 shift ; Mouse -34 -5 shift ; Mouse -16 0 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 shift ; Mouse -24 -9 shift ; Mouse -27 -12 shift ; Mouse -4 -2 shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; shift ; shift MouseRight ; shift MouseRight ; shift MouseRight <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift ; W shift ; W shift ; W shift <|action_end|>
```

## demonstration_optimization_000099

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220121-155633` |
| 图片帧 | `[615, 619, 623, 627]` |
| 目标动作区间 | `[615, 631]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 615**

![demonstration_optimization_000099 frame 615](images/demonstration_optimization_000099_00.jpg)

**图 2，帧 619**

![demonstration_optimization_000099 frame 619](images/demonstration_optimization_000099_01.jpg)

**图 3，帧 623**

![demonstration_optimization_000099 frame 623](images/demonstration_optimization_000099_02.jpg)

**图 4，帧 627**

![demonstration_optimization_000099 frame 627](images/demonstration_optimization_000099_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse 0 3 ; Mouse 1 2 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse 0 3 ; Mouse 1 2 <|action_end|>
```

## image_sequence_to_action_000000

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-59a4bace2b0c-20220125-094601` |
| 图片帧 | `[275, 276, 277, 278, 279]` |
| 目标动作区间 | `[275, 279]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 275**

![image_sequence_to_action_000000 frame 275](images/image_sequence_to_action_000000_00.jpg)

**图 2，帧 276**

![image_sequence_to_action_000000 frame 276](images/image_sequence_to_action_000000_01.jpg)

**图 3，帧 277**

![image_sequence_to_action_000000 frame 277](images/image_sequence_to_action_000000_02.jpg)

**图 4，帧 278**

![image_sequence_to_action_000000 frame 278](images/image_sequence_to_action_000000_03.jpg)

**图 5，帧 279**

![image_sequence_to_action_000000 frame 279](images/image_sequence_to_action_000000_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -8 -11 W space ; Mouse -2 -6 W space ; W space ; Mouse 0 -1 W space <|action_end|>
```

## image_sequence_to_action_000001

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-db279e17529b-20220114-113556` |
| 图片帧 | `[142, 143, 144, 145, 146]` |
| 目标动作区间 | `[142, 146]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 142**

![image_sequence_to_action_000001 frame 142](images/image_sequence_to_action_000001_00.jpg)

**图 2，帧 143**

![image_sequence_to_action_000001 frame 143](images/image_sequence_to_action_000001_01.jpg)

**图 3，帧 144**

![image_sequence_to_action_000001 frame 144](images/image_sequence_to_action_000001_02.jpg)

**图 4，帧 145**

![image_sequence_to_action_000001 frame 145](images/image_sequence_to_action_000001_03.jpg)

**图 5，帧 146**

![image_sequence_to_action_000001 frame 146](images/image_sequence_to_action_000001_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; D shift ; D shift ; Mouse -1 3 D shift ; D shift <|action_end|>
```

## image_sequence_to_action_000002

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20220122-121926` |
| 图片帧 | `[6937, 6938, 6939, 6940, 6941]` |
| 目标动作区间 | `[6937, 6941]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6937**

![image_sequence_to_action_000002 frame 6937](images/image_sequence_to_action_000002_00.jpg)

**图 2，帧 6938**

![image_sequence_to_action_000002 frame 6938](images/image_sequence_to_action_000002_01.jpg)

**图 3，帧 6939**

![image_sequence_to_action_000002 frame 6939](images/image_sequence_to_action_000002_02.jpg)

**图 4，帧 6940**

![image_sequence_to_action_000002 frame 6940](images/image_sequence_to_action_000002_03.jpg)

**图 5，帧 6941**

![image_sequence_to_action_000002 frame 6941](images/image_sequence_to_action_000002_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -78 15 ; Mouse -40 5 ; Mouse -16 0 ; MouseRight <|action_end|>
```

## image_sequence_to_action_000003

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player635-f153ac423f61-20220120-133452` |
| 图片帧 | `[8866, 8867, 8868, 8869, 8870]` |
| 目标动作区间 | `[8866, 8870]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8866**

![image_sequence_to_action_000003 frame 8866](images/image_sequence_to_action_000003_00.jpg)

**图 2，帧 8867**

![image_sequence_to_action_000003 frame 8867](images/image_sequence_to_action_000003_01.jpg)

**图 3，帧 8868**

![image_sequence_to_action_000003 frame 8868](images/image_sequence_to_action_000003_02.jpg)

**图 4，帧 8869**

![image_sequence_to_action_000003 frame 8869](images/image_sequence_to_action_000003_03.jpg)

**图 5，帧 8870**

![image_sequence_to_action_000003 frame 8870](images/image_sequence_to_action_000003_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 5 -1 W ; Mouse 9 -3 W ; Mouse 7 -2 W ; Mouse 4 -1 W <|action_end|>
```

## image_sequence_to_action_000004

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `thirsty-lavender-koala-a7e20be37793-20220216-141306` |
| 图片帧 | `[3058, 3059, 3060, 3061, 3062]` |
| 目标动作区间 | `[3058, 3062]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3058**

![image_sequence_to_action_000004 frame 3058](images/image_sequence_to_action_000004_00.jpg)

**图 2，帧 3059**

![image_sequence_to_action_000004 frame 3059](images/image_sequence_to_action_000004_01.jpg)

**图 3，帧 3060**

![image_sequence_to_action_000004 frame 3060](images/image_sequence_to_action_000004_02.jpg)

**图 4，帧 3061**

![image_sequence_to_action_000004 frame 3061](images/image_sequence_to_action_000004_03.jpg)

**图 5，帧 3062**

![image_sequence_to_action_000004 frame 3062](images/image_sequence_to_action_000004_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; Mouse -13 14 MouseLeft ; Mouse -18 17 MouseLeft <|action_end|>
```

## image_sequence_to_action_000005

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-6d758aad3947-20220306-121356` |
| 图片帧 | `[5701, 5702, 5703, 5704, 5705]` |
| 目标动作区间 | `[5701, 5705]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5701**

![image_sequence_to_action_000005 frame 5701](images/image_sequence_to_action_000005_00.jpg)

**图 2，帧 5702**

![image_sequence_to_action_000005 frame 5702](images/image_sequence_to_action_000005_01.jpg)

**图 3，帧 5703**

![image_sequence_to_action_000005 frame 5703](images/image_sequence_to_action_000005_02.jpg)

**图 4，帧 5704**

![image_sequence_to_action_000005 frame 5704](images/image_sequence_to_action_000005_03.jpg)

**图 5，帧 5705**

![image_sequence_to_action_000005 frame 5705](images/image_sequence_to_action_000005_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000006

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220225-144906` |
| 图片帧 | `[4442, 4443, 4444, 4445, 4446]` |
| 目标动作区间 | `[4442, 4446]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4442**

![image_sequence_to_action_000006 frame 4442](images/image_sequence_to_action_000006_00.jpg)

**图 2，帧 4443**

![image_sequence_to_action_000006 frame 4443](images/image_sequence_to_action_000006_01.jpg)

**图 3，帧 4444**

![image_sequence_to_action_000006 frame 4444](images/image_sequence_to_action_000006_02.jpg)

**图 4，帧 4445**

![image_sequence_to_action_000006 frame 4445](images/image_sequence_to_action_000006_03.jpg)

**图 5，帧 4446**

![image_sequence_to_action_000006 frame 4446](images/image_sequence_to_action_000006_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W MouseRight ; W MouseRight ; W MouseRight ; W MouseRight <|action_end|>
```

## image_sequence_to_action_000007

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220308-080453` |
| 图片帧 | `[1709, 1710, 1711, 1712, 1713]` |
| 目标动作区间 | `[1709, 1713]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1709**

![image_sequence_to_action_000007 frame 1709](images/image_sequence_to_action_000007_00.jpg)

**图 2，帧 1710**

![image_sequence_to_action_000007 frame 1710](images/image_sequence_to_action_000007_01.jpg)

**图 3，帧 1711**

![image_sequence_to_action_000007 frame 1711](images/image_sequence_to_action_000007_02.jpg)

**图 4，帧 1712**

![image_sequence_to_action_000007 frame 1712](images/image_sequence_to_action_000007_03.jpg)

**图 5，帧 1713**

![image_sequence_to_action_000007 frame 1713](images/image_sequence_to_action_000007_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 3 shift ; Mouse -4 0 shift ; Mouse -7 2 shift ; Mouse -5 0 shift <|action_end|>
```

## image_sequence_to_action_000008

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-9ad6d5efb98c-20220306-172457` |
| 图片帧 | `[3855, 3856, 3857, 3858, 3859]` |
| 目标动作区间 | `[3855, 3859]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3855**

![image_sequence_to_action_000008 frame 3855](images/image_sequence_to_action_000008_00.jpg)

**图 2，帧 3856**

![image_sequence_to_action_000008 frame 3856](images/image_sequence_to_action_000008_01.jpg)

**图 3，帧 3857**

![image_sequence_to_action_000008 frame 3857](images/image_sequence_to_action_000008_02.jpg)

**图 4，帧 3858**

![image_sequence_to_action_000008 frame 3858](images/image_sequence_to_action_000008_03.jpg)

**图 5，帧 3859**

![image_sequence_to_action_000008 frame 3859](images/image_sequence_to_action_000008_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -27 6 MouseLeft ; Mouse -52 84 MouseLeft ; Mouse -86 114 MouseLeft ; Mouse -160 -16 MouseLeft <|action_end|>
```

## image_sequence_to_action_000009

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `snippy-chartreuse-mastiff-c636490d741c-20220227-144110` |
| 图片帧 | `[12668, 12669, 12670, 12671, 12672]` |
| 目标动作区间 | `[12668, 12672]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12668**

![image_sequence_to_action_000009 frame 12668](images/image_sequence_to_action_000009_00.jpg)

**图 2，帧 12669**

![image_sequence_to_action_000009 frame 12669](images/image_sequence_to_action_000009_01.jpg)

**图 3，帧 12670**

![image_sequence_to_action_000009 frame 12670](images/image_sequence_to_action_000009_02.jpg)

**图 4，帧 12671**

![image_sequence_to_action_000009 frame 12671](images/image_sequence_to_action_000009_03.jpg)

**图 5，帧 12672**

![image_sequence_to_action_000009 frame 12672](images/image_sequence_to_action_000009_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; Mouse 0 -1 W ; W <|action_end|>
```

## image_sequence_to_action_000010

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220108-221706` |
| 图片帧 | `[4178, 4179, 4180, 4181, 4182]` |
| 目标动作区间 | `[4178, 4182]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4178**

![image_sequence_to_action_000010 frame 4178](images/image_sequence_to_action_000010_00.jpg)

**图 2，帧 4179**

![image_sequence_to_action_000010 frame 4179](images/image_sequence_to_action_000010_01.jpg)

**图 3，帧 4180**

![image_sequence_to_action_000010 frame 4180](images/image_sequence_to_action_000010_02.jpg)

**图 4，帧 4181**

![image_sequence_to_action_000010 frame 4181](images/image_sequence_to_action_000010_03.jpg)

**图 5，帧 4182**

![image_sequence_to_action_000010 frame 4182](images/image_sequence_to_action_000010_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; Mouse -4 0 W ; W ; Mouse 6 32 W <|action_end|>
```

## image_sequence_to_action_000011

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-0fc32e74de74-20220220-185133` |
| 图片帧 | `[3653, 3654, 3655, 3656, 3657]` |
| 目标动作区间 | `[3653, 3657]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3653**

![image_sequence_to_action_000011 frame 3653](images/image_sequence_to_action_000011_00.jpg)

**图 2，帧 3654**

![image_sequence_to_action_000011 frame 3654](images/image_sequence_to_action_000011_01.jpg)

**图 3，帧 3655**

![image_sequence_to_action_000011 frame 3655](images/image_sequence_to_action_000011_02.jpg)

**图 4，帧 3656**

![image_sequence_to_action_000011 frame 3656](images/image_sequence_to_action_000011_03.jpg)

**图 5，帧 3657**

![image_sequence_to_action_000011 frame 3657](images/image_sequence_to_action_000011_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 29 -7 ; Mouse 16 -2 ; MouseRight ; Mouse 1 0 MouseRight <|action_end|>
```

## image_sequence_to_action_000012

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-50d9bd30442d-20220118-094130` |
| 图片帧 | `[316, 317, 318, 319, 320]` |
| 目标动作区间 | `[316, 320]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 316**

![image_sequence_to_action_000012 frame 316](images/image_sequence_to_action_000012_00.jpg)

**图 2，帧 317**

![image_sequence_to_action_000012 frame 317](images/image_sequence_to_action_000012_01.jpg)

**图 3，帧 318**

![image_sequence_to_action_000012 frame 318](images/image_sequence_to_action_000012_02.jpg)

**图 4，帧 319**

![image_sequence_to_action_000012 frame 319](images/image_sequence_to_action_000012_03.jpg)

**图 5，帧 320**

![image_sequence_to_action_000012 frame 320](images/image_sequence_to_action_000012_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -29 87 W ; Mouse -25 65 ; Mouse 0 30 ; Mouse 19 9 <|action_end|>
```

## image_sequence_to_action_000013

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-f153ac423f61-20220121-172710` |
| 图片帧 | `[222, 223, 224, 225, 226]` |
| 目标动作区间 | `[222, 226]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 222**

![image_sequence_to_action_000013 frame 222](images/image_sequence_to_action_000013_00.jpg)

**图 2，帧 223**

![image_sequence_to_action_000013 frame 223](images/image_sequence_to_action_000013_01.jpg)

**图 3，帧 224**

![image_sequence_to_action_000013 frame 224](images/image_sequence_to_action_000013_02.jpg)

**图 4，帧 225**

![image_sequence_to_action_000013 frame 225](images/image_sequence_to_action_000013_03.jpg)

**图 5，帧 226**

![image_sequence_to_action_000013 frame 226](images/image_sequence_to_action_000013_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -15 2 ; Mouse -17 9 MouseRight ; Mouse -7 6 MouseRight ; MouseRight <|action_end|>
```

## image_sequence_to_action_000014

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220125-190246` |
| 图片帧 | `[13069, 13070, 13071, 13072, 13073]` |
| 目标动作区间 | `[13069, 13073]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 13069**

![image_sequence_to_action_000014 frame 13069](images/image_sequence_to_action_000014_00.jpg)

**图 2，帧 13070**

![image_sequence_to_action_000014 frame 13070](images/image_sequence_to_action_000014_01.jpg)

**图 3，帧 13071**

![image_sequence_to_action_000014 frame 13071](images/image_sequence_to_action_000014_02.jpg)

**图 4，帧 13072**

![image_sequence_to_action_000014 frame 13072](images/image_sequence_to_action_000014_03.jpg)

**图 5，帧 13073**

![image_sequence_to_action_000014 frame 13073](images/image_sequence_to_action_000014_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 0 W ; W space ; W space ; Mouse -119 -18 W space <|action_end|>
```

## image_sequence_to_action_000015

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `woozy-ruby-ostrich-23a9a14e250b-20220215-175646` |
| 图片帧 | `[2463, 2464, 2465, 2466, 2467]` |
| 目标动作区间 | `[2463, 2467]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2463**

![image_sequence_to_action_000015 frame 2463](images/image_sequence_to_action_000015_00.jpg)

**图 2，帧 2464**

![image_sequence_to_action_000015 frame 2464](images/image_sequence_to_action_000015_01.jpg)

**图 3，帧 2465**

![image_sequence_to_action_000015 frame 2465](images/image_sequence_to_action_000015_02.jpg)

**图 4，帧 2466**

![image_sequence_to_action_000015 frame 2466](images/image_sequence_to_action_000015_03.jpg)

**图 5，帧 2467**

![image_sequence_to_action_000015 frame 2467](images/image_sequence_to_action_000015_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W A space ctrl ; W A space ctrl ; W A space ctrl ; W A space ctrl <|action_end|>
```

## image_sequence_to_action_000016

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220118-221055` |
| 图片帧 | `[22318, 22319, 22320, 22321, 22322]` |
| 目标动作区间 | `[22318, 22322]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 22318**

![image_sequence_to_action_000016 frame 22318](images/image_sequence_to_action_000016_00.jpg)

**图 2，帧 22319**

![image_sequence_to_action_000016 frame 22319](images/image_sequence_to_action_000016_01.jpg)

**图 3，帧 22320**

![image_sequence_to_action_000016 frame 22320](images/image_sequence_to_action_000016_02.jpg)

**图 4，帧 22321**

![image_sequence_to_action_000016 frame 22321](images/image_sequence_to_action_000016_03.jpg)

**图 5，帧 22322**

![image_sequence_to_action_000016 frame 22322](images/image_sequence_to_action_000016_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 3 0 D MouseRight ; Mouse 9 -1 D MouseRight ; Mouse 6 -3 D MouseRight ; Mouse 10 -2 D MouseRight <|action_end|>
```

## image_sequence_to_action_000017

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-97650c85c6b5-20211231-194436` |
| 图片帧 | `[572, 573, 574, 575, 576]` |
| 目标动作区间 | `[572, 576]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 572**

![image_sequence_to_action_000017 frame 572](images/image_sequence_to_action_000017_00.jpg)

**图 2，帧 573**

![image_sequence_to_action_000017 frame 573](images/image_sequence_to_action_000017_01.jpg)

**图 3，帧 574**

![image_sequence_to_action_000017 frame 574](images/image_sequence_to_action_000017_02.jpg)

**图 4，帧 575**

![image_sequence_to_action_000017 frame 575](images/image_sequence_to_action_000017_03.jpg)

**图 5，帧 576**

![image_sequence_to_action_000017 frame 576](images/image_sequence_to_action_000017_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000018

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-a041ac54f509-20220104-150107` |
| 图片帧 | `[8640, 8641, 8642, 8643, 8644]` |
| 目标动作区间 | `[8640, 8644]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8640**

![image_sequence_to_action_000018 frame 8640](images/image_sequence_to_action_000018_00.jpg)

**图 2，帧 8641**

![image_sequence_to_action_000018 frame 8641](images/image_sequence_to_action_000018_01.jpg)

**图 3，帧 8642**

![image_sequence_to_action_000018 frame 8642](images/image_sequence_to_action_000018_02.jpg)

**图 4，帧 8643**

![image_sequence_to_action_000018 frame 8643](images/image_sequence_to_action_000018_03.jpg)

**图 5，帧 8644**

![image_sequence_to_action_000018 frame 8644](images/image_sequence_to_action_000018_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; Mouse -1 6 W space ctrl ; Mouse 6 9 W space ctrl <|action_end|>
```

## image_sequence_to_action_000019

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `woozy-ruby-ostrich-90a647e39947-20220308-092828` |
| 图片帧 | `[2748, 2749, 2750, 2751, 2752]` |
| 目标动作区间 | `[2748, 2752]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2748**

![image_sequence_to_action_000019 frame 2748](images/image_sequence_to_action_000019_00.jpg)

**图 2，帧 2749**

![image_sequence_to_action_000019 frame 2749](images/image_sequence_to_action_000019_01.jpg)

**图 3，帧 2750**

![image_sequence_to_action_000019 frame 2750](images/image_sequence_to_action_000019_02.jpg)

**图 4，帧 2751**

![image_sequence_to_action_000019 frame 2751](images/image_sequence_to_action_000019_03.jpg)

**图 5，帧 2752**

![image_sequence_to_action_000019 frame 2752](images/image_sequence_to_action_000019_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 6 0 W A ; Mouse 52 -19 W A ; Mouse 72 -28 W A MouseLeft ; Mouse 30 -18 A MouseLeft <|action_end|>
```

## image_sequence_to_action_000020

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player29-f153ac423f61-20211125-215930` |
| 图片帧 | `[454, 455, 456, 457, 458]` |
| 目标动作区间 | `[454, 458]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 454**

![image_sequence_to_action_000020 frame 454](images/image_sequence_to_action_000020_00.jpg)

**图 2，帧 455**

![image_sequence_to_action_000020 frame 455](images/image_sequence_to_action_000020_01.jpg)

**图 3，帧 456**

![image_sequence_to_action_000020 frame 456](images/image_sequence_to_action_000020_02.jpg)

**图 4，帧 457**

![image_sequence_to_action_000020 frame 457](images/image_sequence_to_action_000020_03.jpg)

**图 5，帧 458**

![image_sequence_to_action_000020 frame 458](images/image_sequence_to_action_000020_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 4 4 S D MouseLeft ; Mouse 2 17 S D space MouseLeft ; Mouse 4 10 S D space MouseLeft ; Mouse 11 24 S D space MouseLeft <|action_end|>
```

## image_sequence_to_action_000021

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-3aac5baa2627-20220130-145506` |
| 图片帧 | `[849, 850, 851, 852, 853]` |
| 目标动作区间 | `[849, 853]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 849**

![image_sequence_to_action_000021 frame 849](images/image_sequence_to_action_000021_00.jpg)

**图 2，帧 850**

![image_sequence_to_action_000021 frame 850](images/image_sequence_to_action_000021_01.jpg)

**图 3，帧 851**

![image_sequence_to_action_000021 frame 851](images/image_sequence_to_action_000021_02.jpg)

**图 4，帧 852**

![image_sequence_to_action_000021 frame 852](images/image_sequence_to_action_000021_03.jpg)

**图 5，帧 853**

![image_sequence_to_action_000021 frame 853](images/image_sequence_to_action_000021_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 54 1 W space ctrl ; Mouse 81 0 W space ctrl ; Mouse 62 0 W space ctrl ; Mouse 30 -1 W space ctrl <|action_end|>
```

## image_sequence_to_action_000022

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220124-202201` |
| 图片帧 | `[53, 54, 55, 56, 57]` |
| 目标动作区间 | `[53, 57]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 53**

![image_sequence_to_action_000022 frame 53](images/image_sequence_to_action_000022_00.jpg)

**图 2，帧 54**

![image_sequence_to_action_000022 frame 54](images/image_sequence_to_action_000022_01.jpg)

**图 3，帧 55**

![image_sequence_to_action_000022 frame 55](images/image_sequence_to_action_000022_02.jpg)

**图 4，帧 56**

![image_sequence_to_action_000022 frame 56](images/image_sequence_to_action_000022_03.jpg)

**图 5，帧 57**

![image_sequence_to_action_000022 frame 57](images/image_sequence_to_action_000022_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ;  ;  ;  <|action_end|>
```

## image_sequence_to_action_000023

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220209-050728` |
| 图片帧 | `[5649, 5650, 5651, 5652, 5653]` |
| 目标动作区间 | `[5649, 5653]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5649**

![image_sequence_to_action_000023 frame 5649](images/image_sequence_to_action_000023_00.jpg)

**图 2，帧 5650**

![image_sequence_to_action_000023 frame 5650](images/image_sequence_to_action_000023_01.jpg)

**图 3，帧 5651**

![image_sequence_to_action_000023 frame 5651](images/image_sequence_to_action_000023_02.jpg)

**图 4，帧 5652**

![image_sequence_to_action_000023 frame 5652](images/image_sequence_to_action_000023_03.jpg)

**图 5，帧 5653**

![image_sequence_to_action_000023 frame 5653](images/image_sequence_to_action_000023_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 4 W ctrl ; W ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

## image_sequence_to_action_000024

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player565-f153ac423f61-20220204-212225` |
| 图片帧 | `[2416, 2417, 2418, 2419, 2420]` |
| 目标动作区间 | `[2416, 2420]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2416**

![image_sequence_to_action_000024 frame 2416](images/image_sequence_to_action_000024_00.jpg)

**图 2，帧 2417**

![image_sequence_to_action_000024 frame 2417](images/image_sequence_to_action_000024_01.jpg)

**图 3，帧 2418**

![image_sequence_to_action_000024 frame 2418](images/image_sequence_to_action_000024_02.jpg)

**图 4，帧 2419**

![image_sequence_to_action_000024 frame 2419](images/image_sequence_to_action_000024_03.jpg)

**图 5，帧 2420**

![image_sequence_to_action_000024 frame 2420](images/image_sequence_to_action_000024_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -84 10 W A shift MouseLeft ; Mouse -155 2 W A shift MouseLeft ; Mouse -113 -1 W A shift MouseLeft ; Mouse -33 0 W A shift MouseLeft <|action_end|>
```

## image_sequence_to_action_000025

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220324-195004` |
| 图片帧 | `[11146, 11147, 11148, 11149, 11150]` |
| 目标动作区间 | `[11146, 11150]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 11146**

![image_sequence_to_action_000025 frame 11146](images/image_sequence_to_action_000025_00.jpg)

**图 2，帧 11147**

![image_sequence_to_action_000025 frame 11147](images/image_sequence_to_action_000025_01.jpg)

**图 3，帧 11148**

![image_sequence_to_action_000025 frame 11148](images/image_sequence_to_action_000025_02.jpg)

**图 4，帧 11149**

![image_sequence_to_action_000025 frame 11149](images/image_sequence_to_action_000025_03.jpg)

**图 5，帧 11150**

![image_sequence_to_action_000025 frame 11150](images/image_sequence_to_action_000025_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 0 W ; W ; W ; W A <|action_end|>
```

## image_sequence_to_action_000026

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `woozy-ruby-ostrich-01090972f489-20220221-000457` |
| 图片帧 | `[1616, 1617, 1618, 1619, 1620]` |
| 目标动作区间 | `[1616, 1620]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1616**

![image_sequence_to_action_000026 frame 1616](images/image_sequence_to_action_000026_00.jpg)

**图 2，帧 1617**

![image_sequence_to_action_000026 frame 1617](images/image_sequence_to_action_000026_01.jpg)

**图 3，帧 1618**

![image_sequence_to_action_000026 frame 1618](images/image_sequence_to_action_000026_02.jpg)

**图 4，帧 1619**

![image_sequence_to_action_000026 frame 1619](images/image_sequence_to_action_000026_03.jpg)

**图 5，帧 1620**

![image_sequence_to_action_000026 frame 1620](images/image_sequence_to_action_000026_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -14 26 ; Mouse -8 41 ; Mouse 0 40 ; Mouse 4 29 <|action_end|>
```

## image_sequence_to_action_000027

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-11e399e3a64a-20211226-121208` |
| 图片帧 | `[613, 614, 615, 616, 617]` |
| 目标动作区间 | `[613, 617]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 613**

![image_sequence_to_action_000027 frame 613](images/image_sequence_to_action_000027_00.jpg)

**图 2，帧 614**

![image_sequence_to_action_000027 frame 614](images/image_sequence_to_action_000027_01.jpg)

**图 3，帧 615**

![image_sequence_to_action_000027 frame 615](images/image_sequence_to_action_000027_02.jpg)

**图 4，帧 616**

![image_sequence_to_action_000027 frame 616](images/image_sequence_to_action_000027_03.jpg)

**图 5，帧 617**

![image_sequence_to_action_000027 frame 617](images/image_sequence_to_action_000027_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000028

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220305-105808` |
| 图片帧 | `[199, 200, 201, 202, 203]` |
| 目标动作区间 | `[199, 203]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 199**

![image_sequence_to_action_000028 frame 199](images/image_sequence_to_action_000028_00.jpg)

**图 2，帧 200**

![image_sequence_to_action_000028 frame 200](images/image_sequence_to_action_000028_01.jpg)

**图 3，帧 201**

![image_sequence_to_action_000028 frame 201](images/image_sequence_to_action_000028_02.jpg)

**图 4，帧 202**

![image_sequence_to_action_000028 frame 202](images/image_sequence_to_action_000028_03.jpg)

**图 5，帧 203**

![image_sequence_to_action_000028 frame 203](images/image_sequence_to_action_000028_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 ; Mouse 16 -8 ; Mouse 21 -17 ; Mouse -38 -38 <|action_end|>
```

## image_sequence_to_action_000029

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player788-f153ac423f61-20220111-203145` |
| 图片帧 | `[92, 93, 94, 95, 96]` |
| 目标动作区间 | `[92, 96]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 92**

![image_sequence_to_action_000029 frame 92](images/image_sequence_to_action_000029_00.jpg)

**图 2，帧 93**

![image_sequence_to_action_000029 frame 93](images/image_sequence_to_action_000029_01.jpg)

**图 3，帧 94**

![image_sequence_to_action_000029 frame 94](images/image_sequence_to_action_000029_02.jpg)

**图 4，帧 95**

![image_sequence_to_action_000029 frame 95](images/image_sequence_to_action_000029_03.jpg)

**图 5，帧 96**

![image_sequence_to_action_000029 frame 96](images/image_sequence_to_action_000029_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## image_sequence_to_action_000030

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `whiny-ecru-cougar-8d6747022b65-20220203-162922` |
| 图片帧 | `[11828, 11829, 11830, 11831, 11832]` |
| 目标动作区间 | `[11828, 11832]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 11828**

![image_sequence_to_action_000030 frame 11828](images/image_sequence_to_action_000030_00.jpg)

**图 2，帧 11829**

![image_sequence_to_action_000030 frame 11829](images/image_sequence_to_action_000030_01.jpg)

**图 3，帧 11830**

![image_sequence_to_action_000030 frame 11830](images/image_sequence_to_action_000030_02.jpg)

**图 4，帧 11831**

![image_sequence_to_action_000030 frame 11831](images/image_sequence_to_action_000030_03.jpg)

**图 5，帧 11832**

![image_sequence_to_action_000030 frame 11832](images/image_sequence_to_action_000030_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 33 3 W ; W ; W ; W <|action_end|>
```

## image_sequence_to_action_000031

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `tasty-brass-devil-a789e1a7b476-20220226-152930` |
| 图片帧 | `[3239, 3240, 3241, 3242, 3243]` |
| 目标动作区间 | `[3239, 3243]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3239**

![image_sequence_to_action_000031 frame 3239](images/image_sequence_to_action_000031_00.jpg)

**图 2，帧 3240**

![image_sequence_to_action_000031 frame 3240](images/image_sequence_to_action_000031_01.jpg)

**图 3，帧 3241**

![image_sequence_to_action_000031 frame 3241](images/image_sequence_to_action_000031_02.jpg)

**图 4，帧 3242**

![image_sequence_to_action_000031 frame 3242](images/image_sequence_to_action_000031_03.jpg)

**图 5，帧 3243**

![image_sequence_to_action_000031 frame 3243](images/image_sequence_to_action_000031_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -11 7 W ; Mouse -2 2 W ; W ; W <|action_end|>
```

## image_sequence_to_action_000032

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220117-211944` |
| 图片帧 | `[1471, 1472, 1473, 1474, 1475]` |
| 目标动作区间 | `[1471, 1475]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1471**

![image_sequence_to_action_000032 frame 1471](images/image_sequence_to_action_000032_00.jpg)

**图 2，帧 1472**

![image_sequence_to_action_000032 frame 1472](images/image_sequence_to_action_000032_01.jpg)

**图 3，帧 1473**

![image_sequence_to_action_000032 frame 1473](images/image_sequence_to_action_000032_02.jpg)

**图 4，帧 1474**

![image_sequence_to_action_000032 frame 1474](images/image_sequence_to_action_000032_03.jpg)

**图 5，帧 1475**

![image_sequence_to_action_000032 frame 1475](images/image_sequence_to_action_000032_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 4 3 W ; W ctrl ; W ctrl ; Mouse 5 11 W ctrl <|action_end|>
```

## image_sequence_to_action_000033

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `squeaky-magnolia-ocelot-f153ac423f61-20220302-154945` |
| 图片帧 | `[2153, 2154, 2155, 2156, 2157]` |
| 目标动作区间 | `[2153, 2157]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2153**

![image_sequence_to_action_000033 frame 2153](images/image_sequence_to_action_000033_00.jpg)

**图 2，帧 2154**

![image_sequence_to_action_000033 frame 2154](images/image_sequence_to_action_000033_01.jpg)

**图 3，帧 2155**

![image_sequence_to_action_000033 frame 2155](images/image_sequence_to_action_000033_02.jpg)

**图 4，帧 2156**

![image_sequence_to_action_000033 frame 2156](images/image_sequence_to_action_000033_03.jpg)

**图 5，帧 2157**

![image_sequence_to_action_000033 frame 2157](images/image_sequence_to_action_000033_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; S shift ; S shift ; S shift ; S shift <|action_end|>
```

## image_sequence_to_action_000034

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220220-103519` |
| 图片帧 | `[7035, 7036, 7037, 7038, 7039]` |
| 目标动作区间 | `[7035, 7039]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7035**

![image_sequence_to_action_000034 frame 7035](images/image_sequence_to_action_000034_00.jpg)

**图 2，帧 7036**

![image_sequence_to_action_000034 frame 7036](images/image_sequence_to_action_000034_01.jpg)

**图 3，帧 7037**

![image_sequence_to_action_000034 frame 7037](images/image_sequence_to_action_000034_02.jpg)

**图 4，帧 7038**

![image_sequence_to_action_000034 frame 7038](images/image_sequence_to_action_000034_03.jpg)

**图 5，帧 7039**

![image_sequence_to_action_000034 frame 7039](images/image_sequence_to_action_000034_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -9 -2 W ; Mouse -28 0 W ; Mouse -64 5 W ; Mouse -45 2 W <|action_end|>
```

## image_sequence_to_action_000035

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-1e6fe4a9b042-20220202-010948` |
| 图片帧 | `[5672, 5673, 5674, 5675, 5676]` |
| 目标动作区间 | `[5672, 5676]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5672**

![image_sequence_to_action_000035 frame 5672](images/image_sequence_to_action_000035_00.jpg)

**图 2，帧 5673**

![image_sequence_to_action_000035 frame 5673](images/image_sequence_to_action_000035_01.jpg)

**图 3，帧 5674**

![image_sequence_to_action_000035 frame 5674](images/image_sequence_to_action_000035_02.jpg)

**图 4，帧 5675**

![image_sequence_to_action_000035 frame 5675](images/image_sequence_to_action_000035_03.jpg)

**图 5，帧 5676**

![image_sequence_to_action_000035 frame 5676](images/image_sequence_to_action_000035_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## image_sequence_to_action_000036

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player135-f153ac423f61-20220118-022831` |
| 图片帧 | `[1050, 1051, 1052, 1053, 1054]` |
| 目标动作区间 | `[1050, 1054]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1050**

![image_sequence_to_action_000036 frame 1050](images/image_sequence_to_action_000036_00.jpg)

**图 2，帧 1051**

![image_sequence_to_action_000036 frame 1051](images/image_sequence_to_action_000036_01.jpg)

**图 3，帧 1052**

![image_sequence_to_action_000036 frame 1052](images/image_sequence_to_action_000036_02.jpg)

**图 4，帧 1053**

![image_sequence_to_action_000036 frame 1053](images/image_sequence_to_action_000036_03.jpg)

**图 5，帧 1054**

![image_sequence_to_action_000036 frame 1054](images/image_sequence_to_action_000036_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse 0 1 MouseLeft <|action_end|>
```

## image_sequence_to_action_000037

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220125-215036` |
| 图片帧 | `[8106, 8107, 8108, 8109, 8110]` |
| 目标动作区间 | `[8106, 8110]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8106**

![image_sequence_to_action_000037 frame 8106](images/image_sequence_to_action_000037_00.jpg)

**图 2，帧 8107**

![image_sequence_to_action_000037 frame 8107](images/image_sequence_to_action_000037_01.jpg)

**图 3，帧 8108**

![image_sequence_to_action_000037 frame 8108](images/image_sequence_to_action_000037_02.jpg)

**图 4，帧 8109**

![image_sequence_to_action_000037 frame 8109](images/image_sequence_to_action_000037_03.jpg)

**图 5，帧 8110**

![image_sequence_to_action_000037 frame 8110](images/image_sequence_to_action_000037_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 0 W ; W ; W ; W <|action_end|>
```

## image_sequence_to_action_000038

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player92-f153ac423f61-20220116-011114` |
| 图片帧 | `[1433, 1434, 1435, 1436, 1437]` |
| 目标动作区间 | `[1433, 1437]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1433**

![image_sequence_to_action_000038 frame 1433](images/image_sequence_to_action_000038_00.jpg)

**图 2，帧 1434**

![image_sequence_to_action_000038 frame 1434](images/image_sequence_to_action_000038_01.jpg)

**图 3，帧 1435**

![image_sequence_to_action_000038 frame 1435](images/image_sequence_to_action_000038_02.jpg)

**图 4，帧 1436**

![image_sequence_to_action_000038 frame 1436](images/image_sequence_to_action_000038_03.jpg)

**图 5，帧 1437**

![image_sequence_to_action_000038 frame 1437](images/image_sequence_to_action_000038_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

## image_sequence_to_action_000039

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `wiggy-aquamarine-tapir-7421a7f7ee12-20220303-190821` |
| 图片帧 | `[23, 24, 25, 26, 27]` |
| 目标动作区间 | `[23, 27]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 23**

![image_sequence_to_action_000039 frame 23](images/image_sequence_to_action_000039_00.jpg)

**图 2，帧 24**

![image_sequence_to_action_000039 frame 24](images/image_sequence_to_action_000039_01.jpg)

**图 3，帧 25**

![image_sequence_to_action_000039 frame 25](images/image_sequence_to_action_000039_02.jpg)

**图 4，帧 26**

![image_sequence_to_action_000039 frame 26](images/image_sequence_to_action_000039_03.jpg)

**图 5，帧 27**

![image_sequence_to_action_000039 frame 27](images/image_sequence_to_action_000039_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -9 0 W A ; Mouse 49 1 W A ; Mouse 36 5 W A ; W A <|action_end|>
```

## image_sequence_to_action_000040

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `lovely-persimmon-angora-f153ac423f61-20220307-222611` |
| 图片帧 | `[2085, 2086, 2087, 2088, 2089]` |
| 目标动作区间 | `[2085, 2089]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2085**

![image_sequence_to_action_000040 frame 2085](images/image_sequence_to_action_000040_00.jpg)

**图 2，帧 2086**

![image_sequence_to_action_000040 frame 2086](images/image_sequence_to_action_000040_01.jpg)

**图 3，帧 2087**

![image_sequence_to_action_000040 frame 2087](images/image_sequence_to_action_000040_02.jpg)

**图 4，帧 2088**

![image_sequence_to_action_000040 frame 2088](images/image_sequence_to_action_000040_03.jpg)

**图 5，帧 2089**

![image_sequence_to_action_000040 frame 2089](images/image_sequence_to_action_000040_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -2 ; Mouse 10 20 ; Mouse 33 44 ; Mouse 62 79 <|action_end|>
```

## image_sequence_to_action_000041

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220212-141013` |
| 图片帧 | `[3404, 3405, 3406, 3407, 3408]` |
| 目标动作区间 | `[3404, 3408]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3404**

![image_sequence_to_action_000041 frame 3404](images/image_sequence_to_action_000041_00.jpg)

**图 2，帧 3405**

![image_sequence_to_action_000041 frame 3405](images/image_sequence_to_action_000041_01.jpg)

**图 3，帧 3406**

![image_sequence_to_action_000041 frame 3406](images/image_sequence_to_action_000041_02.jpg)

**图 4，帧 3407**

![image_sequence_to_action_000041 frame 3407](images/image_sequence_to_action_000041_03.jpg)

**图 5，帧 3408**

![image_sequence_to_action_000041 frame 3408](images/image_sequence_to_action_000041_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -13 6 W MouseLeft ; W MouseLeft ; W MouseLeft ; Mouse 4 -3 W MouseLeft <|action_end|>
```

## image_sequence_to_action_000042

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220225-211824` |
| 图片帧 | `[8591, 8592, 8593, 8594, 8595]` |
| 目标动作区间 | `[8591, 8595]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8591**

![image_sequence_to_action_000042 frame 8591](images/image_sequence_to_action_000042_00.jpg)

**图 2，帧 8592**

![image_sequence_to_action_000042 frame 8592](images/image_sequence_to_action_000042_01.jpg)

**图 3，帧 8593**

![image_sequence_to_action_000042 frame 8593](images/image_sequence_to_action_000042_02.jpg)

**图 4，帧 8594**

![image_sequence_to_action_000042 frame 8594](images/image_sequence_to_action_000042_03.jpg)

**图 5，帧 8595**

![image_sequence_to_action_000042 frame 8595](images/image_sequence_to_action_000042_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 3 -2 W ; W ; W ; W MouseRight <|action_end|>
```

## image_sequence_to_action_000043

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220203-123905` |
| 图片帧 | `[9840, 9841, 9842, 9843, 9844]` |
| 目标动作区间 | `[9840, 9844]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9840**

![image_sequence_to_action_000043 frame 9840](images/image_sequence_to_action_000043_00.jpg)

**图 2，帧 9841**

![image_sequence_to_action_000043 frame 9841](images/image_sequence_to_action_000043_01.jpg)

**图 3，帧 9842**

![image_sequence_to_action_000043 frame 9842](images/image_sequence_to_action_000043_02.jpg)

**图 4，帧 9843**

![image_sequence_to_action_000043 frame 9843](images/image_sequence_to_action_000043_03.jpg)

**图 5，帧 9844**

![image_sequence_to_action_000043 frame 9844](images/image_sequence_to_action_000043_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D space ; W D space ; W D space ; W D <|action_end|>
```

## image_sequence_to_action_000044

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player122-cf730e8ac786-20220205-143045` |
| 图片帧 | `[7375, 7376, 7377, 7378, 7379]` |
| 目标动作区间 | `[7375, 7379]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7375**

![image_sequence_to_action_000044 frame 7375](images/image_sequence_to_action_000044_00.jpg)

**图 2，帧 7376**

![image_sequence_to_action_000044 frame 7376](images/image_sequence_to_action_000044_01.jpg)

**图 3，帧 7377**

![image_sequence_to_action_000044 frame 7377](images/image_sequence_to_action_000044_02.jpg)

**图 4，帧 7378**

![image_sequence_to_action_000044 frame 7378](images/image_sequence_to_action_000044_03.jpg)

**图 5，帧 7379**

![image_sequence_to_action_000044 frame 7379](images/image_sequence_to_action_000044_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 5 3 shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## image_sequence_to_action_000045

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220116-165244` |
| 图片帧 | `[1523, 1524, 1525, 1526, 1527]` |
| 目标动作区间 | `[1523, 1527]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1523**

![image_sequence_to_action_000045 frame 1523](images/image_sequence_to_action_000045_00.jpg)

**图 2，帧 1524**

![image_sequence_to_action_000045 frame 1524](images/image_sequence_to_action_000045_01.jpg)

**图 3，帧 1525**

![image_sequence_to_action_000045 frame 1525](images/image_sequence_to_action_000045_02.jpg)

**图 4，帧 1526**

![image_sequence_to_action_000045 frame 1526](images/image_sequence_to_action_000045_03.jpg)

**图 5，帧 1527**

![image_sequence_to_action_000045 frame 1527](images/image_sequence_to_action_000045_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 82 12 W space ctrl ; Mouse 115 16 W A space ctrl ; Mouse 75 11 W A space ctrl ; Mouse 50 14 W A space ctrl <|action_end|>
```

## image_sequence_to_action_000046

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-f153ac423f61-20220121-172710` |
| 图片帧 | `[655, 656, 657, 658, 659]` |
| 目标动作区间 | `[655, 659]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 655**

![image_sequence_to_action_000046 frame 655](images/image_sequence_to_action_000046_00.jpg)

**图 2，帧 656**

![image_sequence_to_action_000046 frame 656](images/image_sequence_to_action_000046_01.jpg)

**图 3，帧 657**

![image_sequence_to_action_000046 frame 657](images/image_sequence_to_action_000046_02.jpg)

**图 4，帧 658**

![image_sequence_to_action_000046 frame 658](images/image_sequence_to_action_000046_03.jpg)

**图 5，帧 659**

![image_sequence_to_action_000046 frame 659](images/image_sequence_to_action_000046_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 9 ; Mouse 0 2 7 ; Mouse 1 13 ; Mouse -2 9 <|action_end|>
```

## image_sequence_to_action_000047

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-367498e0bec4-20220104-161819` |
| 图片帧 | `[2633, 2634, 2635, 2636, 2637]` |
| 目标动作区间 | `[2633, 2637]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2633**

![image_sequence_to_action_000047 frame 2633](images/image_sequence_to_action_000047_00.jpg)

**图 2，帧 2634**

![image_sequence_to_action_000047 frame 2634](images/image_sequence_to_action_000047_01.jpg)

**图 3，帧 2635**

![image_sequence_to_action_000047 frame 2635](images/image_sequence_to_action_000047_02.jpg)

**图 4，帧 2636**

![image_sequence_to_action_000047 frame 2636](images/image_sequence_to_action_000047_03.jpg)

**图 5，帧 2637**

![image_sequence_to_action_000047 frame 2637](images/image_sequence_to_action_000047_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 27 13 A MouseLeft ; Mouse 43 8 A MouseLeft ; Mouse 43 8 MouseLeft ; Mouse 7 0 MouseLeft <|action_end|>
```

## image_sequence_to_action_000048

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-2d35dc8efa94-20220110-201247` |
| 图片帧 | `[12705, 12706, 12707, 12708, 12709]` |
| 目标动作区间 | `[12705, 12709]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12705**

![image_sequence_to_action_000048 frame 12705](images/image_sequence_to_action_000048_00.jpg)

**图 2，帧 12706**

![image_sequence_to_action_000048 frame 12706](images/image_sequence_to_action_000048_01.jpg)

**图 3，帧 12707**

![image_sequence_to_action_000048 frame 12707](images/image_sequence_to_action_000048_02.jpg)

**图 4，帧 12708**

![image_sequence_to_action_000048 frame 12708](images/image_sequence_to_action_000048_03.jpg)

**图 5，帧 12709**

![image_sequence_to_action_000048 frame 12709](images/image_sequence_to_action_000048_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -11 7 W D space ctrl ; Mouse -22 12 W D space ctrl ; Mouse 37 56 W D space ctrl ; Mouse 118 28 W space ctrl <|action_end|>
```

## image_sequence_to_action_000049

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-6520153cd78d-20220123-032231` |
| 图片帧 | `[7930, 7931, 7932, 7933, 7934]` |
| 目标动作区间 | `[7930, 7934]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7930**

![image_sequence_to_action_000049 frame 7930](images/image_sequence_to_action_000049_00.jpg)

**图 2，帧 7931**

![image_sequence_to_action_000049 frame 7931](images/image_sequence_to_action_000049_01.jpg)

**图 3，帧 7932**

![image_sequence_to_action_000049 frame 7932](images/image_sequence_to_action_000049_02.jpg)

**图 4，帧 7933**

![image_sequence_to_action_000049 frame 7933](images/image_sequence_to_action_000049_03.jpg)

**图 5，帧 7934**

![image_sequence_to_action_000049 frame 7934](images/image_sequence_to_action_000049_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000050

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `thirsty-lavender-koala-f153ac423f61-20220114-221926` |
| 图片帧 | `[3049, 3050, 3051, 3052, 3053]` |
| 目标动作区间 | `[3049, 3053]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3049**

![image_sequence_to_action_000050 frame 3049](images/image_sequence_to_action_000050_00.jpg)

**图 2，帧 3050**

![image_sequence_to_action_000050 frame 3050](images/image_sequence_to_action_000050_01.jpg)

**图 3，帧 3051**

![image_sequence_to_action_000050 frame 3051](images/image_sequence_to_action_000050_02.jpg)

**图 4，帧 3052**

![image_sequence_to_action_000050 frame 3052](images/image_sequence_to_action_000050_03.jpg)

**图 5，帧 3053**

![image_sequence_to_action_000050 frame 3053](images/image_sequence_to_action_000050_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -229 -38 ; Mouse -324 -67 ; Mouse -18 -21 ; Mouse 25 -2 <|action_end|>
```

## image_sequence_to_action_000051

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220220-103519` |
| 图片帧 | `[9368, 9369, 9370, 9371, 9372]` |
| 目标动作区间 | `[9368, 9372]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9368**

![image_sequence_to_action_000051 frame 9368](images/image_sequence_to_action_000051_00.jpg)

**图 2，帧 9369**

![image_sequence_to_action_000051 frame 9369](images/image_sequence_to_action_000051_01.jpg)

**图 3，帧 9370**

![image_sequence_to_action_000051 frame 9370](images/image_sequence_to_action_000051_02.jpg)

**图 4，帧 9371**

![image_sequence_to_action_000051 frame 9371](images/image_sequence_to_action_000051_03.jpg)

**图 5，帧 9372**

![image_sequence_to_action_000051 frame 9372](images/image_sequence_to_action_000051_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 9 W MouseRight ; Mouse 2 30 W MouseRight ; Mouse -11 52 ; Mouse -15 31 <|action_end|>
```

## image_sequence_to_action_000052

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220214-200049` |
| 图片帧 | `[8294, 8295, 8296, 8297, 8298]` |
| 目标动作区间 | `[8294, 8298]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8294**

![image_sequence_to_action_000052 frame 8294](images/image_sequence_to_action_000052_00.jpg)

**图 2，帧 8295**

![image_sequence_to_action_000052 frame 8295](images/image_sequence_to_action_000052_01.jpg)

**图 3，帧 8296**

![image_sequence_to_action_000052 frame 8296](images/image_sequence_to_action_000052_02.jpg)

**图 4，帧 8297**

![image_sequence_to_action_000052 frame 8297](images/image_sequence_to_action_000052_03.jpg)

**图 5，帧 8298**

![image_sequence_to_action_000052 frame 8298](images/image_sequence_to_action_000052_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 23 -2 MouseLeft ; Mouse 17 0 MouseLeft ; Mouse 3 -1 W D MouseLeft ; W D MouseLeft <|action_end|>
```

## image_sequence_to_action_000053

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220112-103458` |
| 图片帧 | `[15835, 15836, 15837, 15838, 15839]` |
| 目标动作区间 | `[15835, 15839]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 15835**

![image_sequence_to_action_000053 frame 15835](images/image_sequence_to_action_000053_00.jpg)

**图 2，帧 15836**

![image_sequence_to_action_000053 frame 15836](images/image_sequence_to_action_000053_01.jpg)

**图 3，帧 15837**

![image_sequence_to_action_000053 frame 15837](images/image_sequence_to_action_000053_02.jpg)

**图 4，帧 15838**

![image_sequence_to_action_000053 frame 15838](images/image_sequence_to_action_000053_03.jpg)

**图 5，帧 15839**

![image_sequence_to_action_000053 frame 15839](images/image_sequence_to_action_000053_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## image_sequence_to_action_000054

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-f153ac423f61-20220222-160721` |
| 图片帧 | `[5302, 5303, 5304, 5305, 5306]` |
| 目标动作区间 | `[5302, 5306]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5302**

![image_sequence_to_action_000054 frame 5302](images/image_sequence_to_action_000054_00.jpg)

**图 2，帧 5303**

![image_sequence_to_action_000054 frame 5303](images/image_sequence_to_action_000054_01.jpg)

**图 3，帧 5304**

![image_sequence_to_action_000054 frame 5304](images/image_sequence_to_action_000054_02.jpg)

**图 4，帧 5305**

![image_sequence_to_action_000054 frame 5305](images/image_sequence_to_action_000054_03.jpg)

**图 5，帧 5306**

![image_sequence_to_action_000054 frame 5306](images/image_sequence_to_action_000054_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; S D ; D ; W D ; W D <|action_end|>
```

## image_sequence_to_action_000055

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-2378cec9be47-20220114-084501` |
| 图片帧 | `[3876, 3877, 3878, 3879, 3880]` |
| 目标动作区间 | `[3876, 3880]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3876**

![image_sequence_to_action_000055 frame 3876](images/image_sequence_to_action_000055_00.jpg)

**图 2，帧 3877**

![image_sequence_to_action_000055 frame 3877](images/image_sequence_to_action_000055_01.jpg)

**图 3，帧 3878**

![image_sequence_to_action_000055 frame 3878](images/image_sequence_to_action_000055_02.jpg)

**图 4，帧 3879**

![image_sequence_to_action_000055 frame 3879](images/image_sequence_to_action_000055_03.jpg)

**图 5，帧 3880**

![image_sequence_to_action_000055 frame 3880](images/image_sequence_to_action_000055_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000056

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-2362d417a38e-20220123-032005` |
| 图片帧 | `[444, 445, 446, 447, 448]` |
| 目标动作区间 | `[444, 448]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 444**

![image_sequence_to_action_000056 frame 444](images/image_sequence_to_action_000056_00.jpg)

**图 2，帧 445**

![image_sequence_to_action_000056 frame 445](images/image_sequence_to_action_000056_01.jpg)

**图 3，帧 446**

![image_sequence_to_action_000056 frame 446](images/image_sequence_to_action_000056_02.jpg)

**图 4，帧 447**

![image_sequence_to_action_000056 frame 447](images/image_sequence_to_action_000056_03.jpg)

**图 5，帧 448**

![image_sequence_to_action_000056 frame 448](images/image_sequence_to_action_000056_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 16 -3 W ctrl ; Mouse 8 -2 W ctrl ; W A ctrl ; W A ctrl <|action_end|>
```

## image_sequence_to_action_000057

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-22727c32bcd1-20220225-132653` |
| 图片帧 | `[1011, 1012, 1013, 1014, 1015]` |
| 目标动作区间 | `[1011, 1015]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1011**

![image_sequence_to_action_000057 frame 1011](images/image_sequence_to_action_000057_00.jpg)

**图 2，帧 1012**

![image_sequence_to_action_000057 frame 1012](images/image_sequence_to_action_000057_01.jpg)

**图 3，帧 1013**

![image_sequence_to_action_000057 frame 1013](images/image_sequence_to_action_000057_02.jpg)

**图 4，帧 1014**

![image_sequence_to_action_000057 frame 1014](images/image_sequence_to_action_000057_03.jpg)

**图 5，帧 1015**

![image_sequence_to_action_000057 frame 1015](images/image_sequence_to_action_000057_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 -10 MouseLeft ; Mouse -1 0 MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000058

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `whiny-ecru-cougar-1b7b22794974-20211228-002701` |
| 图片帧 | `[7679, 7680, 7681, 7682, 7683]` |
| 目标动作区间 | `[7679, 7683]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7679**

![image_sequence_to_action_000058 frame 7679](images/image_sequence_to_action_000058_00.jpg)

**图 2，帧 7680**

![image_sequence_to_action_000058 frame 7680](images/image_sequence_to_action_000058_01.jpg)

**图 3，帧 7681**

![image_sequence_to_action_000058 frame 7681](images/image_sequence_to_action_000058_02.jpg)

**图 4，帧 7682**

![image_sequence_to_action_000058 frame 7682](images/image_sequence_to_action_000058_03.jpg)

**图 5，帧 7683**

![image_sequence_to_action_000058 frame 7683](images/image_sequence_to_action_000058_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -6 0 ; Mouse -11 0 ; Mouse -14 5 ; Mouse -29 9 <|action_end|>
```

## image_sequence_to_action_000059

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `tasty-brass-devil-948b2bbb986b-20220204-004518` |
| 图片帧 | `[1338, 1339, 1340, 1341, 1342]` |
| 目标动作区间 | `[1338, 1342]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1338**

![image_sequence_to_action_000059 frame 1338](images/image_sequence_to_action_000059_00.jpg)

**图 2，帧 1339**

![image_sequence_to_action_000059 frame 1339](images/image_sequence_to_action_000059_01.jpg)

**图 3，帧 1340**

![image_sequence_to_action_000059 frame 1340](images/image_sequence_to_action_000059_02.jpg)

**图 4，帧 1341**

![image_sequence_to_action_000059 frame 1341](images/image_sequence_to_action_000059_03.jpg)

**图 5，帧 1342**

![image_sequence_to_action_000059 frame 1342](images/image_sequence_to_action_000059_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000060

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220117-214959` |
| 图片帧 | `[3518, 3519, 3520, 3521, 3522]` |
| 目标动作区间 | `[3518, 3522]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3518**

![image_sequence_to_action_000060 frame 3518](images/image_sequence_to_action_000060_00.jpg)

**图 2，帧 3519**

![image_sequence_to_action_000060 frame 3519](images/image_sequence_to_action_000060_01.jpg)

**图 3，帧 3520**

![image_sequence_to_action_000060 frame 3520](images/image_sequence_to_action_000060_02.jpg)

**图 4，帧 3521**

![image_sequence_to_action_000060 frame 3521](images/image_sequence_to_action_000060_03.jpg)

**图 5，帧 3522**

![image_sequence_to_action_000060 frame 3522](images/image_sequence_to_action_000060_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 41 -46 W ; Mouse 38 -35 W ; W ; Mouse 1 -6 W <|action_end|>
```

## image_sequence_to_action_000061

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `pokey-cyan-spitz-226d83886623-20211226-212517` |
| 图片帧 | `[10246, 10247, 10248, 10249, 10250]` |
| 目标动作区间 | `[10246, 10250]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10246**

![image_sequence_to_action_000061 frame 10246](images/image_sequence_to_action_000061_00.jpg)

**图 2，帧 10247**

![image_sequence_to_action_000061 frame 10247](images/image_sequence_to_action_000061_01.jpg)

**图 3，帧 10248**

![image_sequence_to_action_000061 frame 10248](images/image_sequence_to_action_000061_02.jpg)

**图 4，帧 10249**

![image_sequence_to_action_000061 frame 10249](images/image_sequence_to_action_000061_03.jpg)

**图 5，帧 10250**

![image_sequence_to_action_000061 frame 10250](images/image_sequence_to_action_000061_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -65 3 ; Mouse -86 4 ; Mouse -60 0 ; Mouse -29 0 <|action_end|>
```

## image_sequence_to_action_000062

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220125-012040` |
| 图片帧 | `[9188, 9189, 9190, 9191, 9192]` |
| 目标动作区间 | `[9188, 9192]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9188**

![image_sequence_to_action_000062 frame 9188](images/image_sequence_to_action_000062_00.jpg)

**图 2，帧 9189**

![image_sequence_to_action_000062 frame 9189](images/image_sequence_to_action_000062_01.jpg)

**图 3，帧 9190**

![image_sequence_to_action_000062 frame 9190](images/image_sequence_to_action_000062_02.jpg)

**图 4，帧 9191**

![image_sequence_to_action_000062 frame 9191](images/image_sequence_to_action_000062_03.jpg)

**图 5，帧 9192**

![image_sequence_to_action_000062 frame 9192](images/image_sequence_to_action_000062_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 -15 W space ; Mouse -1 -1 W ; W ; W <|action_end|>
```

## image_sequence_to_action_000063

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `wiggy-aquamarine-tapir-f153ac423f61-20220228-140347` |
| 图片帧 | `[8871, 8872, 8873, 8874, 8875]` |
| 目标动作区间 | `[8871, 8875]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8871**

![image_sequence_to_action_000063 frame 8871](images/image_sequence_to_action_000063_00.jpg)

**图 2，帧 8872**

![image_sequence_to_action_000063 frame 8872](images/image_sequence_to_action_000063_01.jpg)

**图 3，帧 8873**

![image_sequence_to_action_000063 frame 8873](images/image_sequence_to_action_000063_02.jpg)

**图 4，帧 8874**

![image_sequence_to_action_000063 frame 8874](images/image_sequence_to_action_000063_03.jpg)

**图 5，帧 8875**

![image_sequence_to_action_000063 frame 8875](images/image_sequence_to_action_000063_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ; W ; Mouse 9 0 W <|action_end|>
```

## image_sequence_to_action_000064

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `woozy-ruby-ostrich-f153ac423f61-20220123-194441` |
| 图片帧 | `[2229, 2230, 2231, 2232, 2233]` |
| 目标动作区间 | `[2229, 2233]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2229**

![image_sequence_to_action_000064 frame 2229](images/image_sequence_to_action_000064_00.jpg)

**图 2，帧 2230**

![image_sequence_to_action_000064 frame 2230](images/image_sequence_to_action_000064_01.jpg)

**图 3，帧 2231**

![image_sequence_to_action_000064 frame 2231](images/image_sequence_to_action_000064_02.jpg)

**图 4，帧 2232**

![image_sequence_to_action_000064 frame 2232](images/image_sequence_to_action_000064_03.jpg)

**图 5，帧 2233**

![image_sequence_to_action_000064 frame 2233](images/image_sequence_to_action_000064_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 0 ; Mouse 0 6 ; A ; Mouse 0 1 A <|action_end|>
```

## image_sequence_to_action_000065

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220305-105808` |
| 图片帧 | `[1613, 1614, 1615, 1616, 1617]` |
| 目标动作区间 | `[1613, 1617]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1613**

![image_sequence_to_action_000065 frame 1613](images/image_sequence_to_action_000065_00.jpg)

**图 2，帧 1614**

![image_sequence_to_action_000065 frame 1614](images/image_sequence_to_action_000065_01.jpg)

**图 3，帧 1615**

![image_sequence_to_action_000065 frame 1615](images/image_sequence_to_action_000065_02.jpg)

**图 4，帧 1616**

![image_sequence_to_action_000065 frame 1616](images/image_sequence_to_action_000065_03.jpg)

**图 5，帧 1617**

![image_sequence_to_action_000065 frame 1617](images/image_sequence_to_action_000065_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000066

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220202-101422` |
| 图片帧 | `[10585, 10586, 10587, 10588, 10589]` |
| 目标动作区间 | `[10585, 10589]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10585**

![image_sequence_to_action_000066 frame 10585](images/image_sequence_to_action_000066_00.jpg)

**图 2，帧 10586**

![image_sequence_to_action_000066 frame 10586](images/image_sequence_to_action_000066_01.jpg)

**图 3，帧 10587**

![image_sequence_to_action_000066 frame 10587](images/image_sequence_to_action_000066_02.jpg)

**图 4，帧 10588**

![image_sequence_to_action_000066 frame 10588](images/image_sequence_to_action_000066_03.jpg)

**图 5，帧 10589**

![image_sequence_to_action_000066 frame 10589](images/image_sequence_to_action_000066_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000067

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220202-001412` |
| 图片帧 | `[148, 149, 150, 151, 152]` |
| 目标动作区间 | `[148, 152]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 148**

![image_sequence_to_action_000067 frame 148](images/image_sequence_to_action_000067_00.jpg)

**图 2，帧 149**

![image_sequence_to_action_000067 frame 149](images/image_sequence_to_action_000067_01.jpg)

**图 3，帧 150**

![image_sequence_to_action_000067 frame 150](images/image_sequence_to_action_000067_02.jpg)

**图 4，帧 151**

![image_sequence_to_action_000067 frame 151](images/image_sequence_to_action_000067_03.jpg)

**图 5，帧 152**

![image_sequence_to_action_000067 frame 152](images/image_sequence_to_action_000067_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse -2 -2 MouseLeft <|action_end|>
```

## image_sequence_to_action_000068

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-1466c62bae85-20220222-163515` |
| 图片帧 | `[15435, 15436, 15437, 15438, 15439]` |
| 目标动作区间 | `[15435, 15439]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 15435**

![image_sequence_to_action_000068 frame 15435](images/image_sequence_to_action_000068_00.jpg)

**图 2，帧 15436**

![image_sequence_to_action_000068 frame 15436](images/image_sequence_to_action_000068_01.jpg)

**图 3，帧 15437**

![image_sequence_to_action_000068 frame 15437](images/image_sequence_to_action_000068_02.jpg)

**图 4，帧 15438**

![image_sequence_to_action_000068 frame 15438](images/image_sequence_to_action_000068_03.jpg)

**图 5，帧 15439**

![image_sequence_to_action_000068 frame 15439](images/image_sequence_to_action_000068_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000069

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `Player110-e21477c373ad-20220125-165451` |
| 图片帧 | `[2420, 2421, 2422, 2423, 2424]` |
| 目标动作区间 | `[2420, 2424]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2420**

![image_sequence_to_action_000069 frame 2420](images/image_sequence_to_action_000069_00.jpg)

**图 2，帧 2421**

![image_sequence_to_action_000069 frame 2421](images/image_sequence_to_action_000069_01.jpg)

**图 3，帧 2422**

![image_sequence_to_action_000069 frame 2422](images/image_sequence_to_action_000069_02.jpg)

**图 4，帧 2423**

![image_sequence_to_action_000069 frame 2423](images/image_sequence_to_action_000069_03.jpg)

**图 5，帧 2424**

![image_sequence_to_action_000069 frame 2424](images/image_sequence_to_action_000069_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; S A ; Mouse -1 10 S A ; Mouse 0 6 S A ; S A <|action_end|>
```

## image_sequence_to_action_000070

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-3ad2a5c7b232-20220213-142937` |
| 图片帧 | `[1790, 1791, 1792, 1793, 1794]` |
| 目标动作区间 | `[1790, 1794]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1790**

![image_sequence_to_action_000070 frame 1790](images/image_sequence_to_action_000070_00.jpg)

**图 2，帧 1791**

![image_sequence_to_action_000070 frame 1791](images/image_sequence_to_action_000070_01.jpg)

**图 3，帧 1792**

![image_sequence_to_action_000070 frame 1792](images/image_sequence_to_action_000070_02.jpg)

**图 4，帧 1793**

![image_sequence_to_action_000070 frame 1793](images/image_sequence_to_action_000070_03.jpg)

**图 5，帧 1794**

![image_sequence_to_action_000070 frame 1794](images/image_sequence_to_action_000070_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ; MouseRight ; Mouse -1 -6 MouseRight ; Mouse -4 -4 MouseRight <|action_end|>
```

## image_sequence_to_action_000071

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `tasty-brass-devil-f153ac423f61-20220224-115401` |
| 图片帧 | `[1910, 1911, 1912, 1913, 1914]` |
| 目标动作区间 | `[1910, 1914]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1910**

![image_sequence_to_action_000071 frame 1910](images/image_sequence_to_action_000071_00.jpg)

**图 2，帧 1911**

![image_sequence_to_action_000071 frame 1911](images/image_sequence_to_action_000071_01.jpg)

**图 3，帧 1912**

![image_sequence_to_action_000071 frame 1912](images/image_sequence_to_action_000071_02.jpg)

**图 4，帧 1913**

![image_sequence_to_action_000071 frame 1913](images/image_sequence_to_action_000071_03.jpg)

**图 5，帧 1914**

![image_sequence_to_action_000071 frame 1914](images/image_sequence_to_action_000071_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W A ; W A ; W A ; W A <|action_end|>
```

## image_sequence_to_action_000072

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `wiggy-aquamarine-tapir-1f8f05e06d74-20220213-212835` |
| 图片帧 | `[7367, 7368, 7369, 7370, 7371]` |
| 目标动作区间 | `[7367, 7371]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7367**

![image_sequence_to_action_000072 frame 7367](images/image_sequence_to_action_000072_00.jpg)

**图 2，帧 7368**

![image_sequence_to_action_000072 frame 7368](images/image_sequence_to_action_000072_01.jpg)

**图 3，帧 7369**

![image_sequence_to_action_000072 frame 7369](images/image_sequence_to_action_000072_02.jpg)

**图 4，帧 7370**

![image_sequence_to_action_000072 frame 7370](images/image_sequence_to_action_000072_03.jpg)

**图 5，帧 7371**

![image_sequence_to_action_000072 frame 7371](images/image_sequence_to_action_000072_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 4 2 MouseLeft ; Mouse 2 3 MouseLeft ; Mouse 4 1 MouseLeft ; Mouse 1 1 MouseLeft <|action_end|>
```

## image_sequence_to_action_000073

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-87789993cf11-20220125-112928` |
| 图片帧 | `[128, 129, 130, 131, 132]` |
| 目标动作区间 | `[128, 132]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 128**

![image_sequence_to_action_000073 frame 128](images/image_sequence_to_action_000073_00.jpg)

**图 2，帧 129**

![image_sequence_to_action_000073 frame 129](images/image_sequence_to_action_000073_01.jpg)

**图 3，帧 130**

![image_sequence_to_action_000073 frame 130](images/image_sequence_to_action_000073_02.jpg)

**图 4，帧 131**

![image_sequence_to_action_000073 frame 131](images/image_sequence_to_action_000073_03.jpg)

**图 5，帧 132**

![image_sequence_to_action_000073 frame 132](images/image_sequence_to_action_000073_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W MouseRight ; W MouseRight ; Mouse 3 5 W MouseRight <|action_end|>
```

## image_sequence_to_action_000074

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-b2817ff38352-20220110-192213` |
| 图片帧 | `[10944, 10945, 10946, 10947, 10948]` |
| 目标动作区间 | `[10944, 10948]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10944**

![image_sequence_to_action_000074 frame 10944](images/image_sequence_to_action_000074_00.jpg)

**图 2，帧 10945**

![image_sequence_to_action_000074 frame 10945](images/image_sequence_to_action_000074_01.jpg)

**图 3，帧 10946**

![image_sequence_to_action_000074 frame 10946](images/image_sequence_to_action_000074_02.jpg)

**图 4，帧 10947**

![image_sequence_to_action_000074 frame 10947](images/image_sequence_to_action_000074_03.jpg)

**图 5，帧 10948**

![image_sequence_to_action_000074 frame 10948](images/image_sequence_to_action_000074_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -43 4 W D MouseRight ; Mouse -57 0 W D MouseRight ; Mouse -80 0 W D MouseRight ; Mouse -41 0 D MouseRight <|action_end|>
```

## image_sequence_to_action_000075

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220220-200721` |
| 图片帧 | `[5592, 5593, 5594, 5595, 5596]` |
| 目标动作区间 | `[5592, 5596]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5592**

![image_sequence_to_action_000075 frame 5592](images/image_sequence_to_action_000075_00.jpg)

**图 2，帧 5593**

![image_sequence_to_action_000075 frame 5593](images/image_sequence_to_action_000075_01.jpg)

**图 3，帧 5594**

![image_sequence_to_action_000075 frame 5594](images/image_sequence_to_action_000075_02.jpg)

**图 4，帧 5595**

![image_sequence_to_action_000075 frame 5595](images/image_sequence_to_action_000075_03.jpg)

**图 5，帧 5596**

![image_sequence_to_action_000075 frame 5596](images/image_sequence_to_action_000075_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 9 W MouseRight ; W MouseRight ; Mouse 4 -7 W ; Mouse 4 -9 W <|action_end|>
```

## image_sequence_to_action_000076

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `trippy-red-llama-3aa637371999-20220123-162318` |
| 图片帧 | `[26, 27, 28, 29, 30]` |
| 目标动作区间 | `[26, 30]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 26**

![image_sequence_to_action_000076 frame 26](images/image_sequence_to_action_000076_00.jpg)

**图 2，帧 27**

![image_sequence_to_action_000076 frame 27](images/image_sequence_to_action_000076_01.jpg)

**图 3，帧 28**

![image_sequence_to_action_000076 frame 28](images/image_sequence_to_action_000076_02.jpg)

**图 4，帧 29**

![image_sequence_to_action_000076 frame 29](images/image_sequence_to_action_000076_03.jpg)

**图 5，帧 30**

![image_sequence_to_action_000076 frame 30](images/image_sequence_to_action_000076_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -22 18 W ; Mouse -47 13 W ; Mouse -2 2 W space ; W space <|action_end|>
```

## image_sequence_to_action_000077

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20211231-113655` |
| 图片帧 | `[1106, 1107, 1108, 1109, 1110]` |
| 目标动作区间 | `[1106, 1110]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1106**

![image_sequence_to_action_000077 frame 1106](images/image_sequence_to_action_000077_00.jpg)

**图 2，帧 1107**

![image_sequence_to_action_000077 frame 1107](images/image_sequence_to_action_000077_01.jpg)

**图 3，帧 1108**

![image_sequence_to_action_000077 frame 1108](images/image_sequence_to_action_000077_02.jpg)

**图 4，帧 1109**

![image_sequence_to_action_000077 frame 1109](images/image_sequence_to_action_000077_03.jpg)

**图 5，帧 1110**

![image_sequence_to_action_000077 frame 1110](images/image_sequence_to_action_000077_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## image_sequence_to_action_000078

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-50d9bd30442d-20220118-084109` |
| 图片帧 | `[2528, 2529, 2530, 2531, 2532]` |
| 目标动作区间 | `[2528, 2532]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2528**

![image_sequence_to_action_000078 frame 2528](images/image_sequence_to_action_000078_00.jpg)

**图 2，帧 2529**

![image_sequence_to_action_000078 frame 2529](images/image_sequence_to_action_000078_01.jpg)

**图 3，帧 2530**

![image_sequence_to_action_000078 frame 2530](images/image_sequence_to_action_000078_02.jpg)

**图 4，帧 2531**

![image_sequence_to_action_000078 frame 2531](images/image_sequence_to_action_000078_03.jpg)

**图 5，帧 2532**

![image_sequence_to_action_000078 frame 2532](images/image_sequence_to_action_000078_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -6 0 W ; Mouse -10 0 W ctrl ; Mouse -1 -1 W ctrl ; W ctrl <|action_end|>
```

## image_sequence_to_action_000079

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `trippy-red-llama-ce2e814344aa-20220212-210551` |
| 图片帧 | `[195, 196, 197, 198, 199]` |
| 目标动作区间 | `[195, 199]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 195**

![image_sequence_to_action_000079 frame 195](images/image_sequence_to_action_000079_00.jpg)

**图 2，帧 196**

![image_sequence_to_action_000079 frame 196](images/image_sequence_to_action_000079_01.jpg)

**图 3，帧 197**

![image_sequence_to_action_000079 frame 197](images/image_sequence_to_action_000079_02.jpg)

**图 4，帧 198**

![image_sequence_to_action_000079 frame 198](images/image_sequence_to_action_000079_03.jpg)

**图 5，帧 199**

![image_sequence_to_action_000079 frame 199](images/image_sequence_to_action_000079_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## image_sequence_to_action_000080

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220202-131354` |
| 图片帧 | `[2254, 2255, 2256, 2257, 2258]` |
| 目标动作区间 | `[2254, 2258]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2254**

![image_sequence_to_action_000080 frame 2254](images/image_sequence_to_action_000080_00.jpg)

**图 2，帧 2255**

![image_sequence_to_action_000080 frame 2255](images/image_sequence_to_action_000080_01.jpg)

**图 3，帧 2256**

![image_sequence_to_action_000080 frame 2256](images/image_sequence_to_action_000080_02.jpg)

**图 4，帧 2257**

![image_sequence_to_action_000080 frame 2257](images/image_sequence_to_action_000080_03.jpg)

**图 5，帧 2258**

![image_sequence_to_action_000080 frame 2258](images/image_sequence_to_action_000080_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -10 5 W space ; Mouse -4 3 W space ; Mouse -8 6 W space ; Mouse -10 6 W space ctrl <|action_end|>
```

## image_sequence_to_action_000081

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-f153ac423f61-20220123-185704` |
| 图片帧 | `[3961, 3962, 3963, 3964, 3965]` |
| 目标动作区间 | `[3961, 3965]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3961**

![image_sequence_to_action_000081 frame 3961](images/image_sequence_to_action_000081_00.jpg)

**图 2，帧 3962**

![image_sequence_to_action_000081 frame 3962](images/image_sequence_to_action_000081_01.jpg)

**图 3，帧 3963**

![image_sequence_to_action_000081 frame 3963](images/image_sequence_to_action_000081_02.jpg)

**图 4，帧 3964**

![image_sequence_to_action_000081 frame 3964](images/image_sequence_to_action_000081_03.jpg)

**图 5，帧 3965**

![image_sequence_to_action_000081 frame 3965](images/image_sequence_to_action_000081_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 W A space ctrl ; W A space ctrl ; Mouse 2 0 W A space ctrl ; Mouse 41 3 W space ctrl <|action_end|>
```

## image_sequence_to_action_000082

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-ac1f92c757ab-20220107-152129` |
| 图片帧 | `[887, 888, 889, 890, 891]` |
| 目标动作区间 | `[887, 891]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 887**

![image_sequence_to_action_000082 frame 887](images/image_sequence_to_action_000082_00.jpg)

**图 2，帧 888**

![image_sequence_to_action_000082 frame 888](images/image_sequence_to_action_000082_01.jpg)

**图 3，帧 889**

![image_sequence_to_action_000082 frame 889](images/image_sequence_to_action_000082_02.jpg)

**图 4，帧 890**

![image_sequence_to_action_000082 frame 890](images/image_sequence_to_action_000082_03.jpg)

**图 5，帧 891**

![image_sequence_to_action_000082 frame 891](images/image_sequence_to_action_000082_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ;  ; Mouse -4 -15 ; Mouse -11 -34 <|action_end|>
```

## image_sequence_to_action_000083

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220225-023144` |
| 图片帧 | `[22835, 22836, 22837, 22838, 22839]` |
| 目标动作区间 | `[22835, 22839]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 22835**

![image_sequence_to_action_000083 frame 22835](images/image_sequence_to_action_000083_00.jpg)

**图 2，帧 22836**

![image_sequence_to_action_000083 frame 22836](images/image_sequence_to_action_000083_01.jpg)

**图 3，帧 22837**

![image_sequence_to_action_000083 frame 22837](images/image_sequence_to_action_000083_02.jpg)

**图 4，帧 22838**

![image_sequence_to_action_000083 frame 22838](images/image_sequence_to_action_000083_03.jpg)

**图 5，帧 22839**

![image_sequence_to_action_000083 frame 22839](images/image_sequence_to_action_000083_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; D shift MouseLeft ; D shift MouseLeft ; D shift MouseLeft <|action_end|>
```

## image_sequence_to_action_000084

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `woozy-ruby-ostrich-01090972f489-20220221-000457` |
| 图片帧 | `[934, 935, 936, 937, 938]` |
| 目标动作区间 | `[934, 938]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 934**

![image_sequence_to_action_000084 frame 934](images/image_sequence_to_action_000084_00.jpg)

**图 2，帧 935**

![image_sequence_to_action_000084 frame 935](images/image_sequence_to_action_000084_01.jpg)

**图 3，帧 936**

![image_sequence_to_action_000084 frame 936](images/image_sequence_to_action_000084_02.jpg)

**图 4，帧 937**

![image_sequence_to_action_000084 frame 937](images/image_sequence_to_action_000084_03.jpg)

**图 5，帧 938**

![image_sequence_to_action_000084 frame 938](images/image_sequence_to_action_000084_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ; MouseRight ; Mouse -3 1 S MouseRight ; Mouse -15 6 S MouseRight <|action_end|>
```

## image_sequence_to_action_000085

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-3877b94f878c-20220130-080015` |
| 图片帧 | `[9451, 9452, 9453, 9454, 9455]` |
| 目标动作区间 | `[9451, 9455]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9451**

![image_sequence_to_action_000085 frame 9451](images/image_sequence_to_action_000085_00.jpg)

**图 2，帧 9452**

![image_sequence_to_action_000085 frame 9452](images/image_sequence_to_action_000085_01.jpg)

**图 3，帧 9453**

![image_sequence_to_action_000085 frame 9453](images/image_sequence_to_action_000085_02.jpg)

**图 4，帧 9454**

![image_sequence_to_action_000085 frame 9454](images/image_sequence_to_action_000085_03.jpg)

**图 5，帧 9455**

![image_sequence_to_action_000085 frame 9455](images/image_sequence_to_action_000085_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 27 -11 W ; Mouse 12 -4 W ; Mouse 4 0 ; Mouse 1 0 <|action_end|>
```

## image_sequence_to_action_000086

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `lovely-persimmon-angora-f153ac423f61-20220109-185722` |
| 图片帧 | `[63, 64, 65, 66, 67]` |
| 目标动作区间 | `[63, 67]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 63**

![image_sequence_to_action_000086 frame 63](images/image_sequence_to_action_000086_00.jpg)

**图 2，帧 64**

![image_sequence_to_action_000086 frame 64](images/image_sequence_to_action_000086_01.jpg)

**图 3，帧 65**

![image_sequence_to_action_000086 frame 65](images/image_sequence_to_action_000086_02.jpg)

**图 4，帧 66**

![image_sequence_to_action_000086 frame 66](images/image_sequence_to_action_000086_03.jpg)

**图 5，帧 67**

![image_sequence_to_action_000086 frame 67](images/image_sequence_to_action_000086_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 36 -20 ; Mouse 79 -9 ; Mouse 107 -2 ; Mouse -30 48 <|action_end|>
```

## image_sequence_to_action_000087

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `whiny-ecru-cougar-3a2f8d0993df-20211231-011653` |
| 图片帧 | `[1761, 1762, 1763, 1764, 1765]` |
| 目标动作区间 | `[1761, 1765]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1761**

![image_sequence_to_action_000087 frame 1761](images/image_sequence_to_action_000087_00.jpg)

**图 2，帧 1762**

![image_sequence_to_action_000087 frame 1762](images/image_sequence_to_action_000087_01.jpg)

**图 3，帧 1763**

![image_sequence_to_action_000087 frame 1763](images/image_sequence_to_action_000087_02.jpg)

**图 4，帧 1764**

![image_sequence_to_action_000087 frame 1764](images/image_sequence_to_action_000087_03.jpg)

**图 5，帧 1765**

![image_sequence_to_action_000087 frame 1765](images/image_sequence_to_action_000087_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## image_sequence_to_action_000088

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220113-140957` |
| 图片帧 | `[19240, 19241, 19242, 19243, 19244]` |
| 目标动作区间 | `[19240, 19244]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 19240**

![image_sequence_to_action_000088 frame 19240](images/image_sequence_to_action_000088_00.jpg)

**图 2，帧 19241**

![image_sequence_to_action_000088 frame 19241](images/image_sequence_to_action_000088_01.jpg)

**图 3，帧 19242**

![image_sequence_to_action_000088 frame 19242](images/image_sequence_to_action_000088_02.jpg)

**图 4，帧 19243**

![image_sequence_to_action_000088 frame 19243](images/image_sequence_to_action_000088_03.jpg)

**图 5，帧 19244**

![image_sequence_to_action_000088 frame 19244](images/image_sequence_to_action_000088_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift ; shift ; shift ; shift <|action_end|>
```

## image_sequence_to_action_000089

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `tasty-brass-devil-bf432676e0f1-20220128-003713` |
| 图片帧 | `[3964, 3965, 3966, 3967, 3968]` |
| 目标动作区间 | `[3964, 3968]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3964**

![image_sequence_to_action_000089 frame 3964](images/image_sequence_to_action_000089_00.jpg)

**图 2，帧 3965**

![image_sequence_to_action_000089 frame 3965](images/image_sequence_to_action_000089_01.jpg)

**图 3，帧 3966**

![image_sequence_to_action_000089 frame 3966](images/image_sequence_to_action_000089_02.jpg)

**图 4，帧 3967**

![image_sequence_to_action_000089 frame 3967](images/image_sequence_to_action_000089_03.jpg)

**图 5，帧 3968**

![image_sequence_to_action_000089 frame 3968](images/image_sequence_to_action_000089_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; Mouse -1 -3 W MouseLeft ; Mouse 5 -18 W MouseLeft ; Mouse 12 -19 W <|action_end|>
```

## image_sequence_to_action_000090

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `wiggy-aquamarine-tapir-9a7ae70f4c8d-20220213-193820` |
| 图片帧 | `[82, 83, 84, 85, 86]` |
| 目标动作区间 | `[82, 86]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 82**

![image_sequence_to_action_000090 frame 82](images/image_sequence_to_action_000090_00.jpg)

**图 2，帧 83**

![image_sequence_to_action_000090 frame 83](images/image_sequence_to_action_000090_01.jpg)

**图 3，帧 84**

![image_sequence_to_action_000090 frame 84](images/image_sequence_to_action_000090_02.jpg)

**图 4，帧 85**

![image_sequence_to_action_000090 frame 85](images/image_sequence_to_action_000090_03.jpg)

**图 5，帧 86**

![image_sequence_to_action_000090 frame 86](images/image_sequence_to_action_000090_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -8 MouseLeft ; Mouse 0 -11 MouseLeft ; Mouse 7 -27 MouseLeft ; Mouse 33 -34 shift MouseLeft <|action_end|>
```

## image_sequence_to_action_000091

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `gimpy-jade-panda-55fad1d6ca5a-20220109-162933` |
| 图片帧 | `[8635, 8636, 8637, 8638, 8639]` |
| 目标动作区间 | `[8635, 8639]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8635**

![image_sequence_to_action_000091 frame 8635](images/image_sequence_to_action_000091_00.jpg)

**图 2，帧 8636**

![image_sequence_to_action_000091 frame 8636](images/image_sequence_to_action_000091_01.jpg)

**图 3，帧 8637**

![image_sequence_to_action_000091 frame 8637](images/image_sequence_to_action_000091_02.jpg)

**图 4，帧 8638**

![image_sequence_to_action_000091 frame 8638](images/image_sequence_to_action_000091_03.jpg)

**图 5，帧 8639**

![image_sequence_to_action_000091 frame 8639](images/image_sequence_to_action_000091_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 5 16 W ; Mouse 1 5 W ; Mouse 0 1 W ; W <|action_end|>
```

## image_sequence_to_action_000092

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-4ddffd8d9270-20220113-114501` |
| 图片帧 | `[9360, 9361, 9362, 9363, 9364]` |
| 目标动作区间 | `[9360, 9364]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9360**

![image_sequence_to_action_000092 frame 9360](images/image_sequence_to_action_000092_00.jpg)

**图 2，帧 9361**

![image_sequence_to_action_000092 frame 9361](images/image_sequence_to_action_000092_01.jpg)

**图 3，帧 9362**

![image_sequence_to_action_000092 frame 9362](images/image_sequence_to_action_000092_02.jpg)

**图 4，帧 9363**

![image_sequence_to_action_000092 frame 9363](images/image_sequence_to_action_000092_03.jpg)

**图 5，帧 9364**

![image_sequence_to_action_000092 frame 9364](images/image_sequence_to_action_000092_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -28 -25 W ; Mouse -36 -11 W ; Mouse -72 -9 W ; Mouse -92 -2 <|action_end|>
```

## image_sequence_to_action_000093

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220216-120945` |
| 图片帧 | `[2171, 2172, 2173, 2174, 2175]` |
| 目标动作区间 | `[2171, 2175]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2171**

![image_sequence_to_action_000093 frame 2171](images/image_sequence_to_action_000093_00.jpg)

**图 2，帧 2172**

![image_sequence_to_action_000093 frame 2172](images/image_sequence_to_action_000093_01.jpg)

**图 3，帧 2173**

![image_sequence_to_action_000093 frame 2173](images/image_sequence_to_action_000093_02.jpg)

**图 4，帧 2174**

![image_sequence_to_action_000093 frame 2174](images/image_sequence_to_action_000093_03.jpg)

**图 5，帧 2175**

![image_sequence_to_action_000093 frame 2175](images/image_sequence_to_action_000093_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 11 W ; Mouse -3 9 W ; Mouse 0 2 W ; Mouse 2 9 W <|action_end|>
```

## image_sequence_to_action_000094

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220106-182352` |
| 图片帧 | `[3560, 3561, 3562, 3563, 3564]` |
| 目标动作区间 | `[3560, 3564]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3560**

![image_sequence_to_action_000094 frame 3560](images/image_sequence_to_action_000094_00.jpg)

**图 2，帧 3561**

![image_sequence_to_action_000094 frame 3561](images/image_sequence_to_action_000094_01.jpg)

**图 3，帧 3562**

![image_sequence_to_action_000094 frame 3562](images/image_sequence_to_action_000094_02.jpg)

**图 4，帧 3563**

![image_sequence_to_action_000094 frame 3563](images/image_sequence_to_action_000094_03.jpg)

**图 5，帧 3564**

![image_sequence_to_action_000094 frame 3564](images/image_sequence_to_action_000094_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 MouseLeft ; Mouse 1 0 MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## image_sequence_to_action_000095

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `jumpy-denim-lion-a2d4504838ef-20220109-153255` |
| 图片帧 | `[2163, 2164, 2165, 2166, 2167]` |
| 目标动作区间 | `[2163, 2167]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2163**

![image_sequence_to_action_000095 frame 2163](images/image_sequence_to_action_000095_00.jpg)

**图 2，帧 2164**

![image_sequence_to_action_000095 frame 2164](images/image_sequence_to_action_000095_01.jpg)

**图 3，帧 2165**

![image_sequence_to_action_000095 frame 2165](images/image_sequence_to_action_000095_02.jpg)

**图 4，帧 2166**

![image_sequence_to_action_000095 frame 2166](images/image_sequence_to_action_000095_03.jpg)

**图 5，帧 2167**

![image_sequence_to_action_000095 frame 2167](images/image_sequence_to_action_000095_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; Mouse -2 0 W space ; W space <|action_end|>
```

## image_sequence_to_action_000096

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20220128-105451` |
| 图片帧 | `[1118, 1119, 1120, 1121, 1122]` |
| 目标动作区间 | `[1118, 1122]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1118**

![image_sequence_to_action_000096 frame 1118](images/image_sequence_to_action_000096_00.jpg)

**图 2，帧 1119**

![image_sequence_to_action_000096 frame 1119](images/image_sequence_to_action_000096_01.jpg)

**图 3，帧 1120**

![image_sequence_to_action_000096 frame 1120](images/image_sequence_to_action_000096_02.jpg)

**图 4，帧 1121**

![image_sequence_to_action_000096 frame 1121](images/image_sequence_to_action_000096_03.jpg)

**图 5，帧 1122**

![image_sequence_to_action_000096 frame 1122](images/image_sequence_to_action_000096_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 5 0 ; Mouse 3 0 W ; W ; W <|action_end|>
```

## image_sequence_to_action_000097

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `tasty-brass-devil-8e711959f7f8-20220226-141438` |
| 图片帧 | `[24047, 24048, 24049, 24050, 24051]` |
| 目标动作区间 | `[24047, 24051]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 24047**

![image_sequence_to_action_000097 frame 24047](images/image_sequence_to_action_000097_00.jpg)

**图 2，帧 24048**

![image_sequence_to_action_000097 frame 24048](images/image_sequence_to_action_000097_01.jpg)

**图 3，帧 24049**

![image_sequence_to_action_000097 frame 24049](images/image_sequence_to_action_000097_02.jpg)

**图 4，帧 24050**

![image_sequence_to_action_000097 frame 24050](images/image_sequence_to_action_000097_03.jpg)

**图 5，帧 24051**

![image_sequence_to_action_000097 frame 24051](images/image_sequence_to_action_000097_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 -4 ; Mouse 0 -16 ; Mouse 0 -12 ; MouseRight <|action_end|>
```

## image_sequence_to_action_000098

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-a4305c5a3df9-20220115-193329` |
| 图片帧 | `[100, 101, 102, 103, 104]` |
| 目标动作区间 | `[100, 104]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 100**

![image_sequence_to_action_000098 frame 100](images/image_sequence_to_action_000098_00.jpg)

**图 2，帧 101**

![image_sequence_to_action_000098 frame 101](images/image_sequence_to_action_000098_01.jpg)

**图 3，帧 102**

![image_sequence_to_action_000098 frame 102](images/image_sequence_to_action_000098_02.jpg)

**图 4，帧 103**

![image_sequence_to_action_000098 frame 103](images/image_sequence_to_action_000098_03.jpg)

**图 5，帧 104**

![image_sequence_to_action_000098 frame 104](images/image_sequence_to_action_000098_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 171 44 W D ; Mouse 3 6 W D ctrl ; Mouse -8 -12 W ctrl ; Mouse -5 -93 W A ctrl <|action_end|>
```

## image_sequence_to_action_000099

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `tasty-brass-devil-89d8eb8d6ef9-20220128-001239` |
| 图片帧 | `[5644, 5645, 5646, 5647, 5648]` |
| 目标动作区间 | `[5644, 5648]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5644**

![image_sequence_to_action_000099 frame 5644](images/image_sequence_to_action_000099_00.jpg)

**图 2，帧 5645**

![image_sequence_to_action_000099 frame 5645](images/image_sequence_to_action_000099_01.jpg)

**图 3，帧 5646**

![image_sequence_to_action_000099 frame 5646](images/image_sequence_to_action_000099_02.jpg)

**图 4，帧 5647**

![image_sequence_to_action_000099 frame 5647](images/image_sequence_to_action_000099_03.jpg)

**图 5，帧 5648**

![image_sequence_to_action_000099 frame 5648](images/image_sequence_to_action_000099_04.jpg)

### 问题

The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced the transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -50 13 W space ; Mouse -61 14 W space ; Mouse -3 1 W ; W <|action_end|>
```

## history_to_future_action_000000

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `lovely-persimmon-angora-f153ac423f61-20220109-185722` |
| 图片帧 | `[80, 84, 88, 92]` |
| 目标动作区间 | `[92, 96]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 80**

![history_to_future_action_000000 frame 80](images/history_to_future_action_000000_00.jpg)

**图 2，帧 84**

![history_to_future_action_000000 frame 84](images/history_to_future_action_000000_01.jpg)

**图 3，帧 88**

![history_to_future_action_000000 frame 88](images/history_to_future_action_000000_02.jpg)

**图 4，帧 92**

![history_to_future_action_000000 frame 92](images/history_to_future_action_000000_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 15 ; space ; space ; space <|action_end|>
```

## history_to_future_action_000001

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-972909e183ca-20220114-095446` |
| 图片帧 | `[725, 729, 733, 737]` |
| 目标动作区间 | `[737, 741]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 725**

![history_to_future_action_000001 frame 725](images/history_to_future_action_000001_00.jpg)

**图 2，帧 729**

![history_to_future_action_000001 frame 729](images/history_to_future_action_000001_01.jpg)

**图 3，帧 733**

![history_to_future_action_000001 frame 733](images/history_to_future_action_000001_02.jpg)

**图 4，帧 737**

![history_to_future_action_000001 frame 737](images/history_to_future_action_000001_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 2 ; MouseLeft ;  ;  <|action_end|>
```

## history_to_future_action_000002

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220327-102238` |
| 图片帧 | `[14398, 14402, 14406, 14410]` |
| 目标动作区间 | `[14410, 14414]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 14398**

![history_to_future_action_000002 frame 14398](images/history_to_future_action_000002_00.jpg)

**图 2，帧 14402**

![history_to_future_action_000002 frame 14402](images/history_to_future_action_000002_01.jpg)

**图 3，帧 14406**

![history_to_future_action_000002 frame 14406](images/history_to_future_action_000002_02.jpg)

**图 4，帧 14410**

![history_to_future_action_000002 frame 14410](images/history_to_future_action_000002_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; space shift ; space shift MouseRight ; space shift MouseRight ; space shift MouseRight <|action_end|>
```

## history_to_future_action_000003

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `hazy-thistle-chipmunk-f153ac423f61-20220128-140808` |
| 图片帧 | `[7, 11, 15, 19]` |
| 目标动作区间 | `[19, 23]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7**

![history_to_future_action_000003 frame 7](images/history_to_future_action_000003_00.jpg)

**图 2，帧 11**

![history_to_future_action_000003 frame 11](images/history_to_future_action_000003_01.jpg)

**图 3，帧 15**

![history_to_future_action_000003 frame 15](images/history_to_future_action_000003_02.jpg)

**图 4，帧 19**

![history_to_future_action_000003 frame 19](images/history_to_future_action_000003_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; Mouse -65 14 W ; Mouse -113 7 W <|action_end|>
```

## history_to_future_action_000004

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `tasty-brass-devil-04dc56016fe8-20220208-120426` |
| 图片帧 | `[207, 211, 215, 219]` |
| 目标动作区间 | `[219, 223]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 207**

![history_to_future_action_000004 frame 207](images/history_to_future_action_000004_00.jpg)

**图 2，帧 211**

![history_to_future_action_000004 frame 211](images/history_to_future_action_000004_01.jpg)

**图 3，帧 215**

![history_to_future_action_000004 frame 215](images/history_to_future_action_000004_02.jpg)

**图 4，帧 219**

![history_to_future_action_000004 frame 219](images/history_to_future_action_000004_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 3 W ; Mouse -5 2 W ; Mouse -5 0 W ; Mouse -6 2 <|action_end|>
```

## history_to_future_action_000005

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220304-225526` |
| 图片帧 | `[1976, 1980, 1984, 1988]` |
| 目标动作区间 | `[1988, 1992]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1976**

![history_to_future_action_000005 frame 1976](images/history_to_future_action_000005_00.jpg)

**图 2，帧 1980**

![history_to_future_action_000005 frame 1980](images/history_to_future_action_000005_01.jpg)

**图 3，帧 1984**

![history_to_future_action_000005 frame 1984](images/history_to_future_action_000005_02.jpg)

**图 4，帧 1988**

![history_to_future_action_000005 frame 1988](images/history_to_future_action_000005_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -14 4 W ; Mouse -14 3 W ; Mouse -14 4 W ; Mouse -13 3 W <|action_end|>
```

## history_to_future_action_000006

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `scaly-fuchsia-wasp-4e4745a26240-20220113-103534` |
| 图片帧 | `[13328, 13332, 13336, 13340]` |
| 目标动作区间 | `[13340, 13344]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 13328**

![history_to_future_action_000006 frame 13328](images/history_to_future_action_000006_00.jpg)

**图 2，帧 13332**

![history_to_future_action_000006 frame 13332](images/history_to_future_action_000006_01.jpg)

**图 3，帧 13336**

![history_to_future_action_000006 frame 13336](images/history_to_future_action_000006_02.jpg)

**图 4，帧 13340**

![history_to_future_action_000006 frame 13340](images/history_to_future_action_000006_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; Mouse 1 4 W space ; Mouse -1 7 W space ; Mouse 0 7 W space <|action_end|>
```

## history_to_future_action_000007

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220225-154320` |
| 图片帧 | `[4083, 4087, 4091, 4095]` |
| 目标动作区间 | `[4095, 4099]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4083**

![history_to_future_action_000007 frame 4083](images/history_to_future_action_000007_00.jpg)

**图 2，帧 4087**

![history_to_future_action_000007 frame 4087](images/history_to_future_action_000007_01.jpg)

**图 3，帧 4091**

![history_to_future_action_000007 frame 4091](images/history_to_future_action_000007_02.jpg)

**图 4，帧 4095**

![history_to_future_action_000007 frame 4095](images/history_to_future_action_000007_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W A ctrl MouseRight ; W A ctrl MouseRight ; W A ctrl MouseRight ; W A ctrl MouseRight <|action_end|>
```

## history_to_future_action_000008

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-81ac59c6a2ef-20220113-124305` |
| 图片帧 | `[345, 349, 353, 357]` |
| 目标动作区间 | `[357, 361]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 345**

![history_to_future_action_000008 frame 345](images/history_to_future_action_000008_00.jpg)

**图 2，帧 349**

![history_to_future_action_000008 frame 349](images/history_to_future_action_000008_01.jpg)

**图 3，帧 353**

![history_to_future_action_000008 frame 353](images/history_to_future_action_000008_02.jpg)

**图 4，帧 357**

![history_to_future_action_000008 frame 357](images/history_to_future_action_000008_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ; W space ; W space <|action_end|>
```

## history_to_future_action_000009

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-02537371d0f5-20220226-193645` |
| 图片帧 | `[6239, 6243, 6247, 6251]` |
| 目标动作区间 | `[6251, 6255]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6239**

![history_to_future_action_000009 frame 6239](images/history_to_future_action_000009_00.jpg)

**图 2，帧 6243**

![history_to_future_action_000009 frame 6243](images/history_to_future_action_000009_01.jpg)

**图 3，帧 6247**

![history_to_future_action_000009 frame 6247](images/history_to_future_action_000009_02.jpg)

**图 4，帧 6251**

![history_to_future_action_000009 frame 6251](images/history_to_future_action_000009_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D ; Mouse 0 -1 W D ; W D ; W D <|action_end|>
```

## history_to_future_action_000010

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `cheeky-cornflower-setter-22cff0c6900a-20220114-130718` |
| 图片帧 | `[5607, 5611, 5615, 5619]` |
| 目标动作区间 | `[5619, 5623]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5607**

![history_to_future_action_000010 frame 5607](images/history_to_future_action_000010_00.jpg)

**图 2，帧 5611**

![history_to_future_action_000010 frame 5611](images/history_to_future_action_000010_01.jpg)

**图 3，帧 5615**

![history_to_future_action_000010 frame 5615](images/history_to_future_action_000010_02.jpg)

**图 4，帧 5619**

![history_to_future_action_000010 frame 5619](images/history_to_future_action_000010_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 22 -8 MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000011

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `tasty-brass-devil-11af2aaacde4-20220304-012055` |
| 图片帧 | `[7765, 7769, 7773, 7777]` |
| 目标动作区间 | `[7777, 7781]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7765**

![history_to_future_action_000011 frame 7765](images/history_to_future_action_000011_00.jpg)

**图 2，帧 7769**

![history_to_future_action_000011 frame 7769](images/history_to_future_action_000011_01.jpg)

**图 3，帧 7773**

![history_to_future_action_000011 frame 7773](images/history_to_future_action_000011_02.jpg)

**图 4，帧 7777**

![history_to_future_action_000011 frame 7777](images/history_to_future_action_000011_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W ; W ; Mouse 0 2 W <|action_end|>
```

## history_to_future_action_000012

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-d334c303998b-20220306-081615` |
| 图片帧 | `[12873, 12877, 12881, 12885]` |
| 目标动作区间 | `[12885, 12889]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12873**

![history_to_future_action_000012 frame 12873](images/history_to_future_action_000012_00.jpg)

**图 2，帧 12877**

![history_to_future_action_000012 frame 12877](images/history_to_future_action_000012_01.jpg)

**图 3，帧 12881**

![history_to_future_action_000012 frame 12881](images/history_to_future_action_000012_02.jpg)

**图 4，帧 12885**

![history_to_future_action_000012 frame 12885](images/history_to_future_action_000012_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000013

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-38d98a78547a-20220310-200442` |
| 图片帧 | `[13252, 13256, 13260, 13264]` |
| 目标动作区间 | `[13264, 13268]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 13252**

![history_to_future_action_000013 frame 13252](images/history_to_future_action_000013_00.jpg)

**图 2，帧 13256**

![history_to_future_action_000013 frame 13256](images/history_to_future_action_000013_01.jpg)

**图 3，帧 13260**

![history_to_future_action_000013 frame 13260](images/history_to_future_action_000013_02.jpg)

**图 4，帧 13264**

![history_to_future_action_000013 frame 13264](images/history_to_future_action_000013_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; Mouse 13 -33 space ctrl ; Mouse 1 -114 space ctrl ; Mouse 7 -116 W space ctrl <|action_end|>
```

## history_to_future_action_000014

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-7d0e77def0a6-20220211-074759` |
| 图片帧 | `[526, 530, 534, 538]` |
| 目标动作区间 | `[538, 542]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 526**

![history_to_future_action_000014 frame 526](images/history_to_future_action_000014_00.jpg)

**图 2，帧 530**

![history_to_future_action_000014 frame 530](images/history_to_future_action_000014_01.jpg)

**图 3，帧 534**

![history_to_future_action_000014 frame 534](images/history_to_future_action_000014_02.jpg)

**图 4，帧 538**

![history_to_future_action_000014 frame 538](images/history_to_future_action_000014_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -1 MouseLeft ; MouseLeft ; Mouse 1 0 ; Mouse 34 18 <|action_end|>
```

## history_to_future_action_000015

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20211229-000335` |
| 图片帧 | `[10704, 10708, 10712, 10716]` |
| 目标动作区间 | `[10716, 10720]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10704**

![history_to_future_action_000015 frame 10704](images/history_to_future_action_000015_00.jpg)

**图 2，帧 10708**

![history_to_future_action_000015 frame 10708](images/history_to_future_action_000015_01.jpg)

**图 3，帧 10712**

![history_to_future_action_000015 frame 10712](images/history_to_future_action_000015_02.jpg)

**图 4，帧 10716**

![history_to_future_action_000015 frame 10716](images/history_to_future_action_000015_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 3 MouseLeft ; Mouse 0 2 MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000016

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220210-220716` |
| 图片帧 | `[682, 686, 690, 694]` |
| 目标动作区间 | `[694, 698]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 682**

![history_to_future_action_000016 frame 682](images/history_to_future_action_000016_00.jpg)

**图 2，帧 686**

![history_to_future_action_000016 frame 686](images/history_to_future_action_000016_01.jpg)

**图 3，帧 690**

![history_to_future_action_000016 frame 690](images/history_to_future_action_000016_02.jpg)

**图 4，帧 694**

![history_to_future_action_000016 frame 694](images/history_to_future_action_000016_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 0 W D ; W D ; W D ; W D <|action_end|>
```

## history_to_future_action_000017

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `pokey-cyan-spitz-f148f280ecf5-20220209-224017` |
| 图片帧 | `[10191, 10195, 10199, 10203]` |
| 目标动作区间 | `[10203, 10207]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10191**

![history_to_future_action_000017 frame 10191](images/history_to_future_action_000017_00.jpg)

**图 2，帧 10195**

![history_to_future_action_000017 frame 10195](images/history_to_future_action_000017_01.jpg)

**图 3，帧 10199**

![history_to_future_action_000017 frame 10199](images/history_to_future_action_000017_02.jpg)

**图 4，帧 10203**

![history_to_future_action_000017 frame 10203](images/history_to_future_action_000017_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 105 -6 ; Mouse 101 -10 ; Mouse 16 -2 ; Mouse 4 -1 <|action_end|>
```

## history_to_future_action_000018

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220112-163316` |
| 图片帧 | `[2241, 2245, 2249, 2253]` |
| 目标动作区间 | `[2253, 2257]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2241**

![history_to_future_action_000018 frame 2241](images/history_to_future_action_000018_00.jpg)

**图 2，帧 2245**

![history_to_future_action_000018 frame 2245](images/history_to_future_action_000018_01.jpg)

**图 3，帧 2249**

![history_to_future_action_000018 frame 2249](images/history_to_future_action_000018_02.jpg)

**图 4，帧 2253**

![history_to_future_action_000018 frame 2253](images/history_to_future_action_000018_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 20 -6 ; Mouse 6 -1 W ; Mouse 2 0 W ; Mouse 11 -1 W A <|action_end|>
```

## history_to_future_action_000019

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-f153ac423f61-20220219-231103` |
| 图片帧 | `[12582, 12586, 12590, 12594]` |
| 目标动作区间 | `[12594, 12598]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12582**

![history_to_future_action_000019 frame 12582](images/history_to_future_action_000019_00.jpg)

**图 2，帧 12586**

![history_to_future_action_000019 frame 12586](images/history_to_future_action_000019_01.jpg)

**图 3，帧 12590**

![history_to_future_action_000019 frame 12590](images/history_to_future_action_000019_02.jpg)

**图 4，帧 12594**

![history_to_future_action_000019 frame 12594](images/history_to_future_action_000019_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -1 ; W ; W ; W <|action_end|>
```

## history_to_future_action_000020

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-d202f9abd85a-20220215-110356` |
| 图片帧 | `[1623, 1627, 1631, 1635]` |
| 目标动作区间 | `[1635, 1639]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1623**

![history_to_future_action_000020 frame 1623](images/history_to_future_action_000020_00.jpg)

**图 2，帧 1627**

![history_to_future_action_000020 frame 1627](images/history_to_future_action_000020_01.jpg)

**图 3，帧 1631**

![history_to_future_action_000020 frame 1631](images/history_to_future_action_000020_02.jpg)

**图 4，帧 1635**

![history_to_future_action_000020 frame 1635](images/history_to_future_action_000020_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -19 2 ; Mouse -41 0 ; Mouse -38 1 ; Mouse -23 2 <|action_end|>
```

## history_to_future_action_000021

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `tasty-brass-devil-89d8eb8d6ef9-20220128-001239` |
| 图片帧 | `[10094, 10098, 10102, 10106]` |
| 目标动作区间 | `[10106, 10110]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10094**

![history_to_future_action_000021 frame 10094](images/history_to_future_action_000021_00.jpg)

**图 2，帧 10098**

![history_to_future_action_000021 frame 10098](images/history_to_future_action_000021_01.jpg)

**图 3，帧 10102**

![history_to_future_action_000021 frame 10102](images/history_to_future_action_000021_02.jpg)

**图 4，帧 10106**

![history_to_future_action_000021 frame 10106](images/history_to_future_action_000021_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -4 62 ; Mouse 0 52 ; Mouse -2 71 ; Mouse -1 28 <|action_end|>
```

## history_to_future_action_000022

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220118-205219` |
| 图片帧 | `[1860, 1864, 1868, 1872]` |
| 目标动作区间 | `[1872, 1876]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1860**

![history_to_future_action_000022 frame 1860](images/history_to_future_action_000022_00.jpg)

**图 2，帧 1864**

![history_to_future_action_000022 frame 1864](images/history_to_future_action_000022_01.jpg)

**图 3，帧 1868**

![history_to_future_action_000022 frame 1868](images/history_to_future_action_000022_02.jpg)

**图 4，帧 1872**

![history_to_future_action_000022 frame 1872](images/history_to_future_action_000022_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 17 -42 W ; Mouse -6 -42 W ; Mouse -5 -11 W ; W <|action_end|>
```

## history_to_future_action_000023

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player562-f153ac423f61-20220211-210520` |
| 图片帧 | `[15542, 15546, 15550, 15554]` |
| 目标动作区间 | `[15554, 15558]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 15542**

![history_to_future_action_000023 frame 15542](images/history_to_future_action_000023_00.jpg)

**图 2，帧 15546**

![history_to_future_action_000023 frame 15546](images/history_to_future_action_000023_01.jpg)

**图 3，帧 15550**

![history_to_future_action_000023 frame 15550](images/history_to_future_action_000023_02.jpg)

**图 4，帧 15554**

![history_to_future_action_000023 frame 15554](images/history_to_future_action_000023_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## history_to_future_action_000024

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220222-171726` |
| 图片帧 | `[2858, 2862, 2866, 2870]` |
| 目标动作区间 | `[2870, 2874]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2858**

![history_to_future_action_000024 frame 2858](images/history_to_future_action_000024_00.jpg)

**图 2，帧 2862**

![history_to_future_action_000024 frame 2862](images/history_to_future_action_000024_01.jpg)

**图 3，帧 2866**

![history_to_future_action_000024 frame 2866](images/history_to_future_action_000024_02.jpg)

**图 4，帧 2870**

![history_to_future_action_000024 frame 2870](images/history_to_future_action_000024_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -39 9 W D space ; Mouse -26 25 W D space ; Mouse -48 29 W D space ; Mouse -39 17 W D space <|action_end|>
```

## history_to_future_action_000025

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-14eaa0bcd945-20220220-224348` |
| 图片帧 | `[5066, 5070, 5074, 5078]` |
| 目标动作区间 | `[5078, 5082]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5066**

![history_to_future_action_000025 frame 5066](images/history_to_future_action_000025_00.jpg)

**图 2，帧 5070**

![history_to_future_action_000025 frame 5070](images/history_to_future_action_000025_01.jpg)

**图 3，帧 5074**

![history_to_future_action_000025 frame 5074](images/history_to_future_action_000025_02.jpg)

**图 4，帧 5078**

![history_to_future_action_000025 frame 5078](images/history_to_future_action_000025_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 S MouseLeft ; S MouseLeft ; S MouseLeft ; Mouse 16 -9 S <|action_end|>
```

## history_to_future_action_000026

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player565-f153ac423f61-20220204-215303` |
| 图片帧 | `[2755, 2759, 2763, 2767]` |
| 目标动作区间 | `[2767, 2771]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2755**

![history_to_future_action_000026 frame 2755](images/history_to_future_action_000026_00.jpg)

**图 2，帧 2759**

![history_to_future_action_000026 frame 2759](images/history_to_future_action_000026_01.jpg)

**图 3，帧 2763**

![history_to_future_action_000026 frame 2763](images/history_to_future_action_000026_02.jpg)

**图 4，帧 2767**

![history_to_future_action_000026 frame 2767](images/history_to_future_action_000026_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; S space ; S space ; S space ; S space <|action_end|>
```

## history_to_future_action_000027

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220118-205219` |
| 图片帧 | `[712, 716, 720, 724]` |
| 目标动作区间 | `[724, 728]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 712**

![history_to_future_action_000027 frame 712](images/history_to_future_action_000027_00.jpg)

**图 2，帧 716**

![history_to_future_action_000027 frame 716](images/history_to_future_action_000027_01.jpg)

**图 3，帧 720**

![history_to_future_action_000027 frame 720](images/history_to_future_action_000027_02.jpg)

**图 4，帧 724**

![history_to_future_action_000027 frame 724](images/history_to_future_action_000027_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse -2 -3 MouseLeft <|action_end|>
```

## history_to_future_action_000028

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `hazy-thistle-chipmunk-1000da24b354-20220130-184251` |
| 图片帧 | `[51, 55, 59, 63]` |
| 目标动作区间 | `[63, 67]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 51**

![history_to_future_action_000028 frame 51](images/history_to_future_action_000028_00.jpg)

**图 2，帧 55**

![history_to_future_action_000028 frame 55](images/history_to_future_action_000028_01.jpg)

**图 3，帧 59**

![history_to_future_action_000028 frame 59](images/history_to_future_action_000028_02.jpg)

**图 4，帧 63**

![history_to_future_action_000028 frame 63](images/history_to_future_action_000028_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; Mouse 0 7 space ctrl ; Mouse -1 6 space ctrl <|action_end|>
```

## history_to_future_action_000029

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220222-230327` |
| 图片帧 | `[2768, 2772, 2776, 2780]` |
| 目标动作区间 | `[2780, 2784]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2768**

![history_to_future_action_000029 frame 2768](images/history_to_future_action_000029_00.jpg)

**图 2，帧 2772**

![history_to_future_action_000029 frame 2772](images/history_to_future_action_000029_01.jpg)

**图 3，帧 2776**

![history_to_future_action_000029 frame 2776](images/history_to_future_action_000029_02.jpg)

**图 4，帧 2780**

![history_to_future_action_000029 frame 2780](images/history_to_future_action_000029_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 4 -3 W ; Mouse 8 -13 W ; Mouse 27 -18 W ; Mouse 61 -19 W <|action_end|>
```

## history_to_future_action_000030

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `wiggy-aquamarine-tapir-167c21c6a7b9-20220116-235053` |
| 图片帧 | `[1431, 1435, 1439, 1443]` |
| 目标动作区间 | `[1443, 1447]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1431**

![history_to_future_action_000030 frame 1431](images/history_to_future_action_000030_00.jpg)

**图 2，帧 1435**

![history_to_future_action_000030 frame 1435](images/history_to_future_action_000030_01.jpg)

**图 3，帧 1439**

![history_to_future_action_000030 frame 1439](images/history_to_future_action_000030_02.jpg)

**图 4，帧 1443**

![history_to_future_action_000030 frame 1443](images/history_to_future_action_000030_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -8 10 MouseLeft ; Mouse -7 8 MouseLeft ; Mouse -9 5 MouseLeft ; Mouse -2 2 MouseLeft <|action_end|>
```

## history_to_future_action_000031

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-8cac881af70d-20220203-180347` |
| 图片帧 | `[98, 102, 106, 110]` |
| 目标动作区间 | `[110, 114]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 98**

![history_to_future_action_000031 frame 98](images/history_to_future_action_000031_00.jpg)

**图 2，帧 102**

![history_to_future_action_000031 frame 102](images/history_to_future_action_000031_01.jpg)

**图 3，帧 106**

![history_to_future_action_000031 frame 106](images/history_to_future_action_000031_02.jpg)

**图 4，帧 110**

![history_to_future_action_000031 frame 110](images/history_to_future_action_000031_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 -3 W space ctrl ; Mouse 1 -1 W space ctrl ; W space ctrl ; Mouse 2 -3 W space ctrl <|action_end|>
```

## history_to_future_action_000032

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `trippy-red-llama-43e98777af21-20220304-151450` |
| 图片帧 | `[1442, 1446, 1450, 1454]` |
| 目标动作区间 | `[1454, 1458]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1442**

![history_to_future_action_000032 frame 1442](images/history_to_future_action_000032_00.jpg)

**图 2，帧 1446**

![history_to_future_action_000032 frame 1446](images/history_to_future_action_000032_01.jpg)

**图 3，帧 1450**

![history_to_future_action_000032 frame 1450](images/history_to_future_action_000032_02.jpg)

**图 4，帧 1454**

![history_to_future_action_000032 frame 1454](images/history_to_future_action_000032_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -3 ; Mouse 2 -4 ; MouseRight ; Mouse 4 0 <|action_end|>
```

## history_to_future_action_000033

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player38-f153ac423f61-20211221-210043` |
| 图片帧 | `[50375, 50379, 50383, 50387]` |
| 目标动作区间 | `[50387, 50391]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 50375**

![history_to_future_action_000033 frame 50375](images/history_to_future_action_000033_00.jpg)

**图 2，帧 50379**

![history_to_future_action_000033 frame 50379](images/history_to_future_action_000033_01.jpg)

**图 3，帧 50383**

![history_to_future_action_000033 frame 50383](images/history_to_future_action_000033_02.jpg)

**图 4，帧 50387**

![history_to_future_action_000033 frame 50387](images/history_to_future_action_000033_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 6 4 MouseLeft ; Mouse 0 1 MouseLeft ; Mouse 2 2 MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000034

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220116-170244` |
| 图片帧 | `[3410, 3414, 3418, 3422]` |
| 目标动作区间 | `[3422, 3426]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3410**

![history_to_future_action_000034 frame 3410](images/history_to_future_action_000034_00.jpg)

**图 2，帧 3414**

![history_to_future_action_000034 frame 3414](images/history_to_future_action_000034_01.jpg)

**图 3，帧 3418**

![history_to_future_action_000034 frame 3418](images/history_to_future_action_000034_02.jpg)

**图 4，帧 3422**

![history_to_future_action_000034 frame 3422](images/history_to_future_action_000034_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000035

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-3877b94f878c-20220130-080015` |
| 图片帧 | `[22433, 22437, 22441, 22445]` |
| 目标动作区间 | `[22445, 22449]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 22433**

![history_to_future_action_000035 frame 22433](images/history_to_future_action_000035_00.jpg)

**图 2，帧 22437**

![history_to_future_action_000035 frame 22437](images/history_to_future_action_000035_01.jpg)

**图 3，帧 22441**

![history_to_future_action_000035 frame 22441](images/history_to_future_action_000035_02.jpg)

**图 4，帧 22445**

![history_to_future_action_000035 frame 22445](images/history_to_future_action_000035_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 6 -7 S shift MouseRight ; Mouse 67 -7 S shift ; Mouse 86 5 S shift ; Mouse 48 -2 S shift <|action_end|>
```

## history_to_future_action_000036

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220122-160112` |
| 图片帧 | `[28399, 28403, 28407, 28411]` |
| 目标动作区间 | `[28411, 28415]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 28399**

![history_to_future_action_000036 frame 28399](images/history_to_future_action_000036_00.jpg)

**图 2，帧 28403**

![history_to_future_action_000036 frame 28403](images/history_to_future_action_000036_01.jpg)

**图 3，帧 28407**

![history_to_future_action_000036 frame 28407](images/history_to_future_action_000036_02.jpg)

**图 4，帧 28411**

![history_to_future_action_000036 frame 28411](images/history_to_future_action_000036_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; Mouse -24 0 W ; Mouse -41 0 W ; Mouse -15 0 W <|action_end|>
```

## history_to_future_action_000037

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-c0e904a2014d-20220201-165846` |
| 图片帧 | `[2569, 2573, 2577, 2581]` |
| 目标动作区间 | `[2581, 2585]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2569**

![history_to_future_action_000037 frame 2569](images/history_to_future_action_000037_00.jpg)

**图 2，帧 2573**

![history_to_future_action_000037 frame 2573](images/history_to_future_action_000037_01.jpg)

**图 3，帧 2577**

![history_to_future_action_000037 frame 2577](images/history_to_future_action_000037_02.jpg)

**图 4，帧 2581**

![history_to_future_action_000037 frame 2581](images/history_to_future_action_000037_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -17 0 W ctrl ; W ctrl ; Mouse -1 -1 W ctrl ; Mouse 10 -2 W <|action_end|>
```

## history_to_future_action_000038

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `whiny-ecru-cougar-fa758463ff72-20220220-212526` |
| 图片帧 | `[2474, 2478, 2482, 2486]` |
| 目标动作区间 | `[2486, 2490]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2474**

![history_to_future_action_000038 frame 2474](images/history_to_future_action_000038_00.jpg)

**图 2，帧 2478**

![history_to_future_action_000038 frame 2478](images/history_to_future_action_000038_01.jpg)

**图 3，帧 2482**

![history_to_future_action_000038 frame 2482](images/history_to_future_action_000038_02.jpg)

**图 4，帧 2486**

![history_to_future_action_000038 frame 2486](images/history_to_future_action_000038_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 3 -6 ; Mouse 3 -7 ; Mouse 4 -12 ; Mouse 3 -11 <|action_end|>
```

## history_to_future_action_000039

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-d561894beec3-20220306-072039` |
| 图片帧 | `[8776, 8780, 8784, 8788]` |
| 目标动作区间 | `[8788, 8792]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8776**

![history_to_future_action_000039 frame 8776](images/history_to_future_action_000039_00.jpg)

**图 2，帧 8780**

![history_to_future_action_000039 frame 8780](images/history_to_future_action_000039_01.jpg)

**图 3，帧 8784**

![history_to_future_action_000039 frame 8784](images/history_to_future_action_000039_02.jpg)

**图 4，帧 8788**

![history_to_future_action_000039 frame 8788](images/history_to_future_action_000039_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse -1 0 W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## history_to_future_action_000040

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220126-220739` |
| 图片帧 | `[3371, 3375, 3379, 3383]` |
| 目标动作区间 | `[3383, 3387]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3371**

![history_to_future_action_000040 frame 3371](images/history_to_future_action_000040_00.jpg)

**图 2，帧 3375**

![history_to_future_action_000040 frame 3375](images/history_to_future_action_000040_01.jpg)

**图 3，帧 3379**

![history_to_future_action_000040 frame 3379](images/history_to_future_action_000040_02.jpg)

**图 4，帧 3383**

![history_to_future_action_000040 frame 3383](images/history_to_future_action_000040_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 40 -8 W space ; Mouse 21 -3 W space ; Mouse 14 -1 W space ; Mouse 1 0 W space <|action_end|>
```

## history_to_future_action_000041

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-f153ac423f61-20220123-044045` |
| 图片帧 | `[3471, 3475, 3479, 3483]` |
| 目标动作区间 | `[3483, 3487]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3471**

![history_to_future_action_000041 frame 3471](images/history_to_future_action_000041_00.jpg)

**图 2，帧 3475**

![history_to_future_action_000041 frame 3475](images/history_to_future_action_000041_01.jpg)

**图 3，帧 3479**

![history_to_future_action_000041 frame 3479](images/history_to_future_action_000041_02.jpg)

**图 4，帧 3483**

![history_to_future_action_000041 frame 3483](images/history_to_future_action_000041_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## history_to_future_action_000042

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220214-140704` |
| 图片帧 | `[17181, 17185, 17189, 17193]` |
| 目标动作区间 | `[17193, 17197]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 17181**

![history_to_future_action_000042 frame 17181](images/history_to_future_action_000042_00.jpg)

**图 2，帧 17185**

![history_to_future_action_000042 frame 17185](images/history_to_future_action_000042_01.jpg)

**图 3，帧 17189**

![history_to_future_action_000042 frame 17189](images/history_to_future_action_000042_02.jpg)

**图 4，帧 17193**

![history_to_future_action_000042 frame 17193](images/history_to_future_action_000042_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 -1 ; Mouse -10 -2 ; Mouse 6 2 ; Mouse 5 0 <|action_end|>
```

## history_to_future_action_000043

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `jumpy-denim-lion-f153ac423f61-20220321-000020` |
| 图片帧 | `[6354, 6358, 6362, 6366]` |
| 目标动作区间 | `[6366, 6370]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6354**

![history_to_future_action_000043 frame 6354](images/history_to_future_action_000043_00.jpg)

**图 2，帧 6358**

![history_to_future_action_000043 frame 6358](images/history_to_future_action_000043_01.jpg)

**图 3，帧 6362**

![history_to_future_action_000043 frame 6362](images/history_to_future_action_000043_02.jpg)

**图 4，帧 6366**

![history_to_future_action_000043 frame 6366](images/history_to_future_action_000043_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseRight ; shift ; shift ; Mouse 0 8 shift <|action_end|>
```

## history_to_future_action_000044

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `tasty-brass-devil-11af2aaacde4-20220304-012055` |
| 图片帧 | `[10882, 10886, 10890, 10894]` |
| 目标动作区间 | `[10894, 10898]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10882**

![history_to_future_action_000044 frame 10882](images/history_to_future_action_000044_00.jpg)

**图 2，帧 10886**

![history_to_future_action_000044 frame 10886](images/history_to_future_action_000044_01.jpg)

**图 3，帧 10890**

![history_to_future_action_000044 frame 10890](images/history_to_future_action_000044_02.jpg)

**图 4，帧 10894**

![history_to_future_action_000044 frame 10894](images/history_to_future_action_000044_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 10 MouseLeft ; Mouse 0 1 MouseLeft ; Mouse 0 15 MouseLeft ; Mouse 0 21 MouseLeft <|action_end|>
```

## history_to_future_action_000045

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `snippy-chartreuse-mastiff-f153ac423f61-20220204-180034` |
| 图片帧 | `[5234, 5238, 5242, 5246]` |
| 目标动作区间 | `[5246, 5250]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5234**

![history_to_future_action_000045 frame 5234](images/history_to_future_action_000045_00.jpg)

**图 2，帧 5238**

![history_to_future_action_000045 frame 5238](images/history_to_future_action_000045_01.jpg)

**图 3，帧 5242**

![history_to_future_action_000045 frame 5242](images/history_to_future_action_000045_02.jpg)

**图 4，帧 5246**

![history_to_future_action_000045 frame 5246](images/history_to_future_action_000045_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D ; W D ; W D ; W D <|action_end|>
```

## history_to_future_action_000046

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-a805c585b1a4-20220304-044453` |
| 图片帧 | `[5429, 5433, 5437, 5441]` |
| 目标动作区间 | `[5441, 5445]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5429**

![history_to_future_action_000046 frame 5429](images/history_to_future_action_000046_00.jpg)

**图 2，帧 5433**

![history_to_future_action_000046 frame 5433](images/history_to_future_action_000046_01.jpg)

**图 3，帧 5437**

![history_to_future_action_000046 frame 5437](images/history_to_future_action_000046_02.jpg)

**图 4，帧 5441**

![history_to_future_action_000046 frame 5441](images/history_to_future_action_000046_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 0 W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## history_to_future_action_000047

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-14eaa0bcd945-20220220-230409` |
| 图片帧 | `[887, 891, 895, 899]` |
| 目标动作区间 | `[899, 903]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 887**

![history_to_future_action_000047 frame 887](images/history_to_future_action_000047_00.jpg)

**图 2，帧 891**

![history_to_future_action_000047 frame 891](images/history_to_future_action_000047_01.jpg)

**图 3，帧 895**

![history_to_future_action_000047 frame 895](images/history_to_future_action_000047_02.jpg)

**图 4，帧 899**

![history_to_future_action_000047 frame 899](images/history_to_future_action_000047_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -16 -2 S ; Mouse -36 -3 S MouseLeft ; Mouse -7 0 S MouseLeft ; Mouse -8 0 MouseLeft <|action_end|>
```

## history_to_future_action_000048

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `scaly-fuchsia-wasp-919ac0f2ca9f-20220308-130420` |
| 图片帧 | `[23342, 23346, 23350, 23354]` |
| 目标动作区间 | `[23354, 23358]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 23342**

![history_to_future_action_000048 frame 23342](images/history_to_future_action_000048_00.jpg)

**图 2，帧 23346**

![history_to_future_action_000048 frame 23346](images/history_to_future_action_000048_01.jpg)

**图 3，帧 23350**

![history_to_future_action_000048 frame 23350](images/history_to_future_action_000048_02.jpg)

**图 4，帧 23354**

![history_to_future_action_000048 frame 23354](images/history_to_future_action_000048_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 260 -45 S D shift ; Mouse 192 -21 D shift ; Mouse 49 -37 D shift ; Mouse -4 -8 D shift <|action_end|>
```

## history_to_future_action_000049

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-3569dd869bf6-20220302-202134` |
| 图片帧 | `[2673, 2677, 2681, 2685]` |
| 目标动作区间 | `[2685, 2689]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2673**

![history_to_future_action_000049 frame 2673](images/history_to_future_action_000049_00.jpg)

**图 2，帧 2677**

![history_to_future_action_000049 frame 2677](images/history_to_future_action_000049_01.jpg)

**图 3，帧 2681**

![history_to_future_action_000049 frame 2681](images/history_to_future_action_000049_02.jpg)

**图 4，帧 2685**

![history_to_future_action_000049 frame 2685](images/history_to_future_action_000049_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 1 W A space ctrl ; Mouse -2 1 W A space ctrl ; Mouse -4 2 W space ctrl ; Mouse -1 1 W space ctrl <|action_end|>
```

## history_to_future_action_000050

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220124-204258` |
| 图片帧 | `[1580, 1584, 1588, 1592]` |
| 目标动作区间 | `[1592, 1596]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1580**

![history_to_future_action_000050 frame 1580](images/history_to_future_action_000050_00.jpg)

**图 2，帧 1584**

![history_to_future_action_000050 frame 1584](images/history_to_future_action_000050_01.jpg)

**图 3，帧 1588**

![history_to_future_action_000050 frame 1588](images/history_to_future_action_000050_02.jpg)

**图 4，帧 1592**

![history_to_future_action_000050 frame 1592](images/history_to_future_action_000050_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -14 5 W 8 ; Mouse -4 0 W ; Mouse -1 1 W ; W <|action_end|>
```

## history_to_future_action_000051

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `scaly-fuchsia-wasp-6475cd9e99ad-20211224-154001` |
| 图片帧 | `[6327, 6331, 6335, 6339]` |
| 目标动作区间 | `[6339, 6343]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6327**

![history_to_future_action_000051 frame 6327](images/history_to_future_action_000051_00.jpg)

**图 2，帧 6331**

![history_to_future_action_000051 frame 6331](images/history_to_future_action_000051_01.jpg)

**图 3，帧 6335**

![history_to_future_action_000051 frame 6335](images/history_to_future_action_000051_02.jpg)

**图 4，帧 6339**

![history_to_future_action_000051 frame 6339](images/history_to_future_action_000051_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ; Mouse 13 6 ; Mouse 138 44 ; Mouse 214 4 <|action_end|>
```

## history_to_future_action_000052

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `tasty-brass-devil-0a138c1ac9cd-20220224-122626` |
| 图片帧 | `[3341, 3345, 3349, 3353]` |
| 目标动作区间 | `[3353, 3357]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3341**

![history_to_future_action_000052 frame 3341](images/history_to_future_action_000052_00.jpg)

**图 2，帧 3345**

![history_to_future_action_000052 frame 3345](images/history_to_future_action_000052_01.jpg)

**图 3，帧 3349**

![history_to_future_action_000052 frame 3349](images/history_to_future_action_000052_02.jpg)

**图 4，帧 3353**

![history_to_future_action_000052 frame 3353](images/history_to_future_action_000052_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## history_to_future_action_000053

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-12a51edaab3d-20220125-082417` |
| 图片帧 | `[2904, 2908, 2912, 2916]` |
| 目标动作区间 | `[2916, 2920]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2904**

![history_to_future_action_000053 frame 2904](images/history_to_future_action_000053_00.jpg)

**图 2，帧 2908**

![history_to_future_action_000053 frame 2908](images/history_to_future_action_000053_01.jpg)

**图 3，帧 2912**

![history_to_future_action_000053 frame 2912](images/history_to_future_action_000053_02.jpg)

**图 4，帧 2916**

![history_to_future_action_000053 frame 2916](images/history_to_future_action_000053_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 72 15 ; Mouse 122 25 ; Mouse 145 36 ; Mouse 123 22 <|action_end|>
```

## history_to_future_action_000054

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-d07890fc21c9-20220219-174156` |
| 图片帧 | `[3796, 3800, 3804, 3808]` |
| 目标动作区间 | `[3808, 3812]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3796**

![history_to_future_action_000054 frame 3796](images/history_to_future_action_000054_00.jpg)

**图 2，帧 3800**

![history_to_future_action_000054 frame 3800](images/history_to_future_action_000054_01.jpg)

**图 3，帧 3804**

![history_to_future_action_000054 frame 3804](images/history_to_future_action_000054_02.jpg)

**图 4，帧 3808**

![history_to_future_action_000054 frame 3808](images/history_to_future_action_000054_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; Mouse 5 1 W space ctrl ; Mouse 30 3 W space ctrl <|action_end|>
```

## history_to_future_action_000055

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-3a220760b0b1-20220130-103123` |
| 图片帧 | `[10873, 10877, 10881, 10885]` |
| 目标动作区间 | `[10885, 10889]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10873**

![history_to_future_action_000055 frame 10873](images/history_to_future_action_000055_00.jpg)

**图 2，帧 10877**

![history_to_future_action_000055 frame 10877](images/history_to_future_action_000055_01.jpg)

**图 3，帧 10881**

![history_to_future_action_000055 frame 10881](images/history_to_future_action_000055_02.jpg)

**图 4，帧 10885**

![history_to_future_action_000055 frame 10885](images/history_to_future_action_000055_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -6 0 W space ctrl ; Mouse -5 0 W space ctrl ; Mouse -2 0 W space ctrl ; Mouse -3 0 W space ctrl <|action_end|>
```

## history_to_future_action_000056

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player92-f153ac423f61-20220116-011114` |
| 图片帧 | `[4956, 4960, 4964, 4968]` |
| 目标动作区间 | `[4968, 4972]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4956**

![history_to_future_action_000056 frame 4956](images/history_to_future_action_000056_00.jpg)

**图 2，帧 4960**

![history_to_future_action_000056 frame 4960](images/history_to_future_action_000056_01.jpg)

**图 3，帧 4964**

![history_to_future_action_000056 frame 4964](images/history_to_future_action_000056_02.jpg)

**图 4，帧 4968**

![history_to_future_action_000056 frame 4968](images/history_to_future_action_000056_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -7 2 D ; Mouse -18 9 D ; Mouse -29 15 ; Mouse -27 21 <|action_end|>
```

## history_to_future_action_000057

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220226-093505` |
| 图片帧 | `[3972, 3976, 3980, 3984]` |
| 目标动作区间 | `[3984, 3988]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3972**

![history_to_future_action_000057 frame 3972](images/history_to_future_action_000057_00.jpg)

**图 2，帧 3976**

![history_to_future_action_000057 frame 3976](images/history_to_future_action_000057_01.jpg)

**图 3，帧 3980**

![history_to_future_action_000057 frame 3980](images/history_to_future_action_000057_02.jpg)

**图 4，帧 3984**

![history_to_future_action_000057 frame 3984](images/history_to_future_action_000057_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

## history_to_future_action_000058

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220302-170137` |
| 图片帧 | `[1597, 1601, 1605, 1609]` |
| 目标动作区间 | `[1609, 1613]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1597**

![history_to_future_action_000058 frame 1597](images/history_to_future_action_000058_00.jpg)

**图 2，帧 1601**

![history_to_future_action_000058 frame 1601](images/history_to_future_action_000058_01.jpg)

**图 3，帧 1605**

![history_to_future_action_000058 frame 1605](images/history_to_future_action_000058_02.jpg)

**图 4，帧 1609**

![history_to_future_action_000058 frame 1609](images/history_to_future_action_000058_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -10 W MouseLeft ; Mouse 0 -2 MouseLeft ; Mouse 2 -9 MouseLeft ; Mouse 0 -4 MouseLeft <|action_end|>
```

## history_to_future_action_000059

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `whiny-ecru-cougar-1252293a6a75-20220102-002736` |
| 图片帧 | `[22124, 22128, 22132, 22136]` |
| 目标动作区间 | `[22136, 22140]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 22124**

![history_to_future_action_000059 frame 22124](images/history_to_future_action_000059_00.jpg)

**图 2，帧 22128**

![history_to_future_action_000059 frame 22128](images/history_to_future_action_000059_01.jpg)

**图 3，帧 22132**

![history_to_future_action_000059 frame 22132](images/history_to_future_action_000059_02.jpg)

**图 4，帧 22136**

![history_to_future_action_000059 frame 22136](images/history_to_future_action_000059_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 2 1 MouseLeft ; Mouse 2 1 MouseLeft ; Mouse 1 1 MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000060

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-eef151dc62a0-20211224-172441` |
| 图片帧 | `[1449, 1453, 1457, 1461]` |
| 目标动作区间 | `[1461, 1465]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1449**

![history_to_future_action_000060 frame 1449](images/history_to_future_action_000060_00.jpg)

**图 2，帧 1453**

![history_to_future_action_000060 frame 1453](images/history_to_future_action_000060_01.jpg)

**图 3，帧 1457**

![history_to_future_action_000060 frame 1457](images/history_to_future_action_000060_02.jpg)

**图 4，帧 1461**

![history_to_future_action_000060 frame 1461](images/history_to_future_action_000060_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 10 0 W ctrl ; Mouse 0 1 W ctrl ; Mouse -21 6 W space ctrl ; Mouse -47 7 W space ctrl <|action_end|>
```

## history_to_future_action_000061

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-adf130039d63-20220402-155444` |
| 图片帧 | `[222, 226, 230, 234]` |
| 目标动作区间 | `[234, 238]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 222**

![history_to_future_action_000061 frame 222](images/history_to_future_action_000061_00.jpg)

**图 2，帧 226**

![history_to_future_action_000061 frame 226](images/history_to_future_action_000061_01.jpg)

**图 3，帧 230**

![history_to_future_action_000061 frame 230](images/history_to_future_action_000061_02.jpg)

**图 4，帧 234**

![history_to_future_action_000061 frame 234](images/history_to_future_action_000061_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 14 0 W space ctrl ; Mouse 9 1 W space ctrl ; Mouse 3 2 W space ctrl ; Mouse 2 1 W space <|action_end|>
```

## history_to_future_action_000062

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-f20c680680e3-20220206-144724` |
| 图片帧 | `[6119, 6123, 6127, 6131]` |
| 目标动作区间 | `[6131, 6135]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6119**

![history_to_future_action_000062 frame 6119](images/history_to_future_action_000062_00.jpg)

**图 2，帧 6123**

![history_to_future_action_000062 frame 6123](images/history_to_future_action_000062_01.jpg)

**图 3，帧 6127**

![history_to_future_action_000062 frame 6127](images/history_to_future_action_000062_02.jpg)

**图 4，帧 6131**

![history_to_future_action_000062 frame 6131](images/history_to_future_action_000062_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; Mouse -2 -1 ; Mouse -4 -6 ; Mouse -5 -10 <|action_end|>
```

## history_to_future_action_000063

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-c9ec9ad8ca63-20220118-130021` |
| 图片帧 | `[2182, 2186, 2190, 2194]` |
| 目标动作区间 | `[2194, 2198]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2182**

![history_to_future_action_000063 frame 2182](images/history_to_future_action_000063_00.jpg)

**图 2，帧 2186**

![history_to_future_action_000063 frame 2186](images/history_to_future_action_000063_01.jpg)

**图 3，帧 2190**

![history_to_future_action_000063 frame 2190](images/history_to_future_action_000063_02.jpg)

**图 4，帧 2194**

![history_to_future_action_000063 frame 2194](images/history_to_future_action_000063_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; Mouse 1 0 shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## history_to_future_action_000064

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player52-f153ac423f61-20220118-192143` |
| 图片帧 | `[3179, 3183, 3187, 3191]` |
| 目标动作区间 | `[3191, 3195]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3179**

![history_to_future_action_000064 frame 3179](images/history_to_future_action_000064_00.jpg)

**图 2，帧 3183**

![history_to_future_action_000064 frame 3183](images/history_to_future_action_000064_01.jpg)

**图 3，帧 3187**

![history_to_future_action_000064 frame 3187](images/history_to_future_action_000064_02.jpg)

**图 4，帧 3191**

![history_to_future_action_000064 frame 3191](images/history_to_future_action_000064_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; Mouse 1 0 MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000065

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `wiggy-aquamarine-tapir-f153ac423f61-20220108-182933` |
| 图片帧 | `[5116, 5120, 5124, 5128]` |
| 目标动作区间 | `[5128, 5132]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5116**

![history_to_future_action_000065 frame 5116](images/history_to_future_action_000065_00.jpg)

**图 2，帧 5120**

![history_to_future_action_000065 frame 5120](images/history_to_future_action_000065_01.jpg)

**图 3，帧 5124**

![history_to_future_action_000065 frame 5124](images/history_to_future_action_000065_02.jpg)

**图 4，帧 5128**

![history_to_future_action_000065 frame 5128](images/history_to_future_action_000065_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 4 ; Mouse 0 1 MouseRight ; MouseRight ; MouseRight <|action_end|>
```

## history_to_future_action_000066

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220125-215542` |
| 图片帧 | `[1011, 1015, 1019, 1023]` |
| 目标动作区间 | `[1023, 1027]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1011**

![history_to_future_action_000066 frame 1011](images/history_to_future_action_000066_00.jpg)

**图 2，帧 1015**

![history_to_future_action_000066 frame 1015](images/history_to_future_action_000066_01.jpg)

**图 3，帧 1019**

![history_to_future_action_000066 frame 1019](images/history_to_future_action_000066_02.jpg)

**图 4，帧 1023**

![history_to_future_action_000066 frame 1023](images/history_to_future_action_000066_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 4 -2 MouseRight ; MouseRight ; Mouse 0 -2 MouseRight ; Mouse -6 -20 <|action_end|>
```

## history_to_future_action_000067

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `pokey-cyan-spitz-f153ac423f61-20220124-224229` |
| 图片帧 | `[8693, 8697, 8701, 8705]` |
| 目标动作区间 | `[8705, 8709]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8693**

![history_to_future_action_000067 frame 8693](images/history_to_future_action_000067_00.jpg)

**图 2，帧 8697**

![history_to_future_action_000067 frame 8697](images/history_to_future_action_000067_01.jpg)

**图 3，帧 8701**

![history_to_future_action_000067 frame 8701](images/history_to_future_action_000067_02.jpg)

**图 4，帧 8705**

![history_to_future_action_000067 frame 8705](images/history_to_future_action_000067_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 7 -26 W ctrl MouseLeft ; Mouse 2 -10 W ctrl MouseLeft ; Mouse 0 -2 W ctrl MouseLeft ; W ctrl MouseLeft <|action_end|>
```

## history_to_future_action_000068

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220125-205043` |
| 图片帧 | `[466, 470, 474, 478]` |
| 目标动作区间 | `[478, 482]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 466**

![history_to_future_action_000068 frame 466](images/history_to_future_action_000068_00.jpg)

**图 2，帧 470**

![history_to_future_action_000068 frame 470](images/history_to_future_action_000068_01.jpg)

**图 3，帧 474**

![history_to_future_action_000068 frame 474](images/history_to_future_action_000068_02.jpg)

**图 4，帧 478**

![history_to_future_action_000068 frame 478](images/history_to_future_action_000068_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## history_to_future_action_000069

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `whiny-ecru-cougar-e9e1b6f7a159-20220121-150844` |
| 图片帧 | `[455, 459, 463, 467]` |
| 目标动作区间 | `[467, 471]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 455**

![history_to_future_action_000069 frame 455](images/history_to_future_action_000069_00.jpg)

**图 2，帧 459**

![history_to_future_action_000069 frame 459](images/history_to_future_action_000069_01.jpg)

**图 3，帧 463**

![history_to_future_action_000069 frame 463](images/history_to_future_action_000069_02.jpg)

**图 4，帧 467**

![history_to_future_action_000069 frame 467](images/history_to_future_action_000069_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000070

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `snippy-chartreuse-mastiff-f16c982e7e60-20220104-202249` |
| 图片帧 | `[259, 263, 267, 271]` |
| 目标动作区间 | `[271, 275]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 259**

![history_to_future_action_000070 frame 259](images/history_to_future_action_000070_00.jpg)

**图 2，帧 263**

![history_to_future_action_000070 frame 263](images/history_to_future_action_000070_01.jpg)

**图 3，帧 267**

![history_to_future_action_000070 frame 267](images/history_to_future_action_000070_02.jpg)

**图 4，帧 271**

![history_to_future_action_000070 frame 271](images/history_to_future_action_000070_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 18 3 W ; Mouse 6 1 W ; W ; W <|action_end|>
```

## history_to_future_action_000071

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player402-f153ac423f61-20211117-165816` |
| 图片帧 | `[681, 685, 689, 693]` |
| 目标动作区间 | `[693, 697]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 681**

![history_to_future_action_000071 frame 681](images/history_to_future_action_000071_00.jpg)

**图 2，帧 685**

![history_to_future_action_000071 frame 685](images/history_to_future_action_000071_01.jpg)

**图 3，帧 689**

![history_to_future_action_000071 frame 689](images/history_to_future_action_000071_02.jpg)

**图 4，帧 693**

![history_to_future_action_000071 frame 693](images/history_to_future_action_000071_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## history_to_future_action_000072

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `jumpy-denim-lion-1c4403d4ac27-20220301-060256` |
| 图片帧 | `[23523, 23527, 23531, 23535]` |
| 目标动作区间 | `[23535, 23539]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 23523**

![history_to_future_action_000072 frame 23523](images/history_to_future_action_000072_00.jpg)

**图 2，帧 23527**

![history_to_future_action_000072 frame 23527](images/history_to_future_action_000072_01.jpg)

**图 3，帧 23531**

![history_to_future_action_000072 frame 23531](images/history_to_future_action_000072_02.jpg)

**图 4，帧 23535**

![history_to_future_action_000072 frame 23535](images/history_to_future_action_000072_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

## history_to_future_action_000073

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220113-001334` |
| 图片帧 | `[10995, 10999, 11003, 11007]` |
| 目标动作区间 | `[11007, 11011]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10995**

![history_to_future_action_000073 frame 10995](images/history_to_future_action_000073_00.jpg)

**图 2，帧 10999**

![history_to_future_action_000073 frame 10999](images/history_to_future_action_000073_01.jpg)

**图 3，帧 11003**

![history_to_future_action_000073 frame 11003](images/history_to_future_action_000073_02.jpg)

**图 4，帧 11007**

![history_to_future_action_000073 frame 11007](images/history_to_future_action_000073_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 8 MouseLeft ; Mouse -1 19 MouseLeft ; Mouse 2 34 MouseLeft ; Mouse 2 21 <|action_end|>
```

## history_to_future_action_000074

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player402-f153ac423f61-20211117-170331` |
| 图片帧 | `[11932, 11936, 11940, 11944]` |
| 目标动作区间 | `[11944, 11948]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 11932**

![history_to_future_action_000074 frame 11932](images/history_to_future_action_000074_00.jpg)

**图 2，帧 11936**

![history_to_future_action_000074 frame 11936](images/history_to_future_action_000074_01.jpg)

**图 3，帧 11940**

![history_to_future_action_000074 frame 11940](images/history_to_future_action_000074_02.jpg)

**图 4，帧 11944**

![history_to_future_action_000074 frame 11944](images/history_to_future_action_000074_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000075

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-f89105d72ed4-20220227-124818` |
| 图片帧 | `[3035, 3039, 3043, 3047]` |
| 目标动作区间 | `[3047, 3051]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3035**

![history_to_future_action_000075 frame 3035](images/history_to_future_action_000075_00.jpg)

**图 2，帧 3039**

![history_to_future_action_000075 frame 3039](images/history_to_future_action_000075_01.jpg)

**图 3，帧 3043**

![history_to_future_action_000075 frame 3043](images/history_to_future_action_000075_02.jpg)

**图 4，帧 3047**

![history_to_future_action_000075 frame 3047](images/history_to_future_action_000075_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 2 21 S ; Mouse -3 8 S ; Mouse -5 9 S ; Mouse -1 2 S <|action_end|>
```

## history_to_future_action_000076

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player69-ee57f91f0d02-20211122-150103` |
| 图片帧 | `[204, 208, 212, 216]` |
| 目标动作区间 | `[216, 220]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 204**

![history_to_future_action_000076 frame 204](images/history_to_future_action_000076_00.jpg)

**图 2，帧 208**

![history_to_future_action_000076 frame 208](images/history_to_future_action_000076_01.jpg)

**图 3，帧 212**

![history_to_future_action_000076 frame 212](images/history_to_future_action_000076_02.jpg)

**图 4，帧 216**

![history_to_future_action_000076 frame 216](images/history_to_future_action_000076_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 7 -3 W space ctrl ; Mouse 10 -1 W space ctrl ; Mouse 1 -1 W space ctrl ; Mouse 29 4 W space ctrl <|action_end|>
```

## history_to_future_action_000077

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-f17a1bd047a7-20220129-120033` |
| 图片帧 | `[31798, 31802, 31806, 31810]` |
| 目标动作区间 | `[31810, 31814]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 31798**

![history_to_future_action_000077 frame 31798](images/history_to_future_action_000077_00.jpg)

**图 2，帧 31802**

![history_to_future_action_000077 frame 31802](images/history_to_future_action_000077_01.jpg)

**图 3，帧 31806**

![history_to_future_action_000077 frame 31806](images/history_to_future_action_000077_02.jpg)

**图 4，帧 31810**

![history_to_future_action_000077 frame 31810](images/history_to_future_action_000077_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 8 -12 ; Mouse 6 -6 ; Mouse 0 -1 ;  <|action_end|>
```

## history_to_future_action_000078

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `lovely-persimmon-angora-f153ac423f61-20220212-160351` |
| 图片帧 | `[1858, 1862, 1866, 1870]` |
| 目标动作区间 | `[1870, 1874]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1858**

![history_to_future_action_000078 frame 1858](images/history_to_future_action_000078_00.jpg)

**图 2，帧 1862**

![history_to_future_action_000078 frame 1862](images/history_to_future_action_000078_01.jpg)

**图 3，帧 1866**

![history_to_future_action_000078 frame 1866](images/history_to_future_action_000078_02.jpg)

**图 4，帧 1870**

![history_to_future_action_000078 frame 1870](images/history_to_future_action_000078_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D space ; W D space ; W D space ; W D space <|action_end|>
```

## history_to_future_action_000079

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220202-134426` |
| 图片帧 | `[1759, 1763, 1767, 1771]` |
| 目标动作区间 | `[1771, 1775]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1759**

![history_to_future_action_000079 frame 1759](images/history_to_future_action_000079_00.jpg)

**图 2，帧 1763**

![history_to_future_action_000079 frame 1763](images/history_to_future_action_000079_01.jpg)

**图 3，帧 1767**

![history_to_future_action_000079 frame 1767](images/history_to_future_action_000079_02.jpg)

**图 4，帧 1771**

![history_to_future_action_000079 frame 1771](images/history_to_future_action_000079_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; Mouse -2 0 W space <|action_end|>
```

## history_to_future_action_000080

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `thirsty-lavender-koala-e51495cbc96b-20220126-160855` |
| 图片帧 | `[9580, 9584, 9588, 9592]` |
| 目标动作区间 | `[9592, 9596]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9580**

![history_to_future_action_000080 frame 9580](images/history_to_future_action_000080_00.jpg)

**图 2，帧 9584**

![history_to_future_action_000080 frame 9584](images/history_to_future_action_000080_01.jpg)

**图 3，帧 9588**

![history_to_future_action_000080 frame 9588](images/history_to_future_action_000080_02.jpg)

**图 4，帧 9592**

![history_to_future_action_000080 frame 9592](images/history_to_future_action_000080_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000081

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-b0e45b1500d8-20220306-084415` |
| 图片帧 | `[320, 324, 328, 332]` |
| 目标动作区间 | `[332, 336]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 320**

![history_to_future_action_000081 frame 320](images/history_to_future_action_000081_00.jpg)

**图 2，帧 324**

![history_to_future_action_000081 frame 324](images/history_to_future_action_000081_01.jpg)

**图 3，帧 328**

![history_to_future_action_000081 frame 328](images/history_to_future_action_000081_02.jpg)

**图 4，帧 332**

![history_to_future_action_000081 frame 332](images/history_to_future_action_000081_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 2 1 W space ctrl ; W space ctrl ; Mouse 0 6 W space ctrl ; Mouse -3 10 W space ctrl <|action_end|>
```

## history_to_future_action_000082

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `pokey-cyan-spitz-8677d03577d3-20220309-223019` |
| 图片帧 | `[16856, 16860, 16864, 16868]` |
| 目标动作区间 | `[16868, 16872]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 16856**

![history_to_future_action_000082 frame 16856](images/history_to_future_action_000082_00.jpg)

**图 2，帧 16860**

![history_to_future_action_000082 frame 16860](images/history_to_future_action_000082_01.jpg)

**图 3，帧 16864**

![history_to_future_action_000082 frame 16864](images/history_to_future_action_000082_02.jpg)

**图 4，帧 16868**

![history_to_future_action_000082 frame 16868](images/history_to_future_action_000082_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -6 3 W space ; Mouse -10 3 W ; Mouse -12 4 W ; Mouse -20 6 W <|action_end|>
```

## history_to_future_action_000083

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `thirsty-lavender-koala-f153ac423f61-20220108-223948` |
| 图片帧 | `[10767, 10771, 10775, 10779]` |
| 目标动作区间 | `[10779, 10783]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 10767**

![history_to_future_action_000083 frame 10767](images/history_to_future_action_000083_00.jpg)

**图 2，帧 10771**

![history_to_future_action_000083 frame 10771](images/history_to_future_action_000083_01.jpg)

**图 3，帧 10775**

![history_to_future_action_000083 frame 10775](images/history_to_future_action_000083_02.jpg)

**图 4，帧 10779**

![history_to_future_action_000083 frame 10779](images/history_to_future_action_000083_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -10 -2 W space ; W space ; W ; Mouse 0 2 W <|action_end|>
```

## history_to_future_action_000084

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220119-080157` |
| 图片帧 | `[4551, 4555, 4559, 4563]` |
| 目标动作区间 | `[4563, 4567]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4551**

![history_to_future_action_000084 frame 4551](images/history_to_future_action_000084_00.jpg)

**图 2，帧 4555**

![history_to_future_action_000084 frame 4555](images/history_to_future_action_000084_01.jpg)

**图 3，帧 4559**

![history_to_future_action_000084 frame 4559](images/history_to_future_action_000084_02.jpg)

**图 4，帧 4563**

![history_to_future_action_000084 frame 4563](images/history_to_future_action_000084_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ; W ; Mouse -3 2 W <|action_end|>
```

## history_to_future_action_000085

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220207-200909` |
| 图片帧 | `[1753, 1757, 1761, 1765]` |
| 目标动作区间 | `[1765, 1769]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1753**

![history_to_future_action_000085 frame 1753](images/history_to_future_action_000085_00.jpg)

**图 2，帧 1757**

![history_to_future_action_000085 frame 1757](images/history_to_future_action_000085_01.jpg)

**图 3，帧 1761**

![history_to_future_action_000085 frame 1761](images/history_to_future_action_000085_02.jpg)

**图 4，帧 1765**

![history_to_future_action_000085 frame 1765](images/history_to_future_action_000085_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; Mouse -19 -23 W ; Mouse 0 -1 W ; MouseRight <|action_end|>
```

## history_to_future_action_000086

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20211226-230601` |
| 图片帧 | `[2765, 2769, 2773, 2777]` |
| 目标动作区间 | `[2777, 2781]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2765**

![history_to_future_action_000086 frame 2765](images/history_to_future_action_000086_00.jpg)

**图 2，帧 2769**

![history_to_future_action_000086 frame 2769](images/history_to_future_action_000086_01.jpg)

**图 3，帧 2773**

![history_to_future_action_000086 frame 2773](images/history_to_future_action_000086_02.jpg)

**图 4，帧 2777**

![history_to_future_action_000086 frame 2777](images/history_to_future_action_000086_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 0 shift ; Mouse -5 -4 shift ; Mouse -23 -23 A shift ; Mouse -34 -32 A shift <|action_end|>
```

## history_to_future_action_000087

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-0533cc0fa099-20220225-010404` |
| 图片帧 | `[9844, 9848, 9852, 9856]` |
| 目标动作区间 | `[9856, 9860]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9844**

![history_to_future_action_000087 frame 9844](images/history_to_future_action_000087_00.jpg)

**图 2，帧 9848**

![history_to_future_action_000087 frame 9848](images/history_to_future_action_000087_01.jpg)

**图 3，帧 9852**

![history_to_future_action_000087 frame 9852](images/history_to_future_action_000087_02.jpg)

**图 4，帧 9856**

![history_to_future_action_000087 frame 9856](images/history_to_future_action_000087_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

## history_to_future_action_000088

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `jumpy-denim-lion-2fa86b2f37df-20211227-000119` |
| 图片帧 | `[197, 201, 205, 209]` |
| 目标动作区间 | `[209, 213]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 197**

![history_to_future_action_000088 frame 197](images/history_to_future_action_000088_00.jpg)

**图 2，帧 201**

![history_to_future_action_000088 frame 201](images/history_to_future_action_000088_01.jpg)

**图 3，帧 205**

![history_to_future_action_000088 frame 205](images/history_to_future_action_000088_02.jpg)

**图 4，帧 209**

![history_to_future_action_000088 frame 209](images/history_to_future_action_000088_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W MouseRight ; W MouseRight ; W ; W <|action_end|>
```

## history_to_future_action_000089

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220202-142929` |
| 图片帧 | `[2876, 2880, 2884, 2888]` |
| 目标动作区间 | `[2888, 2892]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2876**

![history_to_future_action_000089 frame 2876](images/history_to_future_action_000089_00.jpg)

**图 2，帧 2880**

![history_to_future_action_000089 frame 2880](images/history_to_future_action_000089_01.jpg)

**图 3，帧 2884**

![history_to_future_action_000089 frame 2884](images/history_to_future_action_000089_02.jpg)

**图 4，帧 2888**

![history_to_future_action_000089 frame 2888](images/history_to_future_action_000089_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W space <|action_end|>
```

## history_to_future_action_000090

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `tasty-brass-devil-f153ac423f61-20220208-110347` |
| 图片帧 | `[26250, 26254, 26258, 26262]` |
| 目标动作区间 | `[26262, 26266]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 26250**

![history_to_future_action_000090 frame 26250](images/history_to_future_action_000090_00.jpg)

**图 2，帧 26254**

![history_to_future_action_000090 frame 26254](images/history_to_future_action_000090_01.jpg)

**图 3，帧 26258**

![history_to_future_action_000090 frame 26258](images/history_to_future_action_000090_02.jpg)

**图 4，帧 26262**

![history_to_future_action_000090 frame 26262](images/history_to_future_action_000090_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## history_to_future_action_000091

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-df1077b506d8-20220125-121018` |
| 图片帧 | `[3001, 3005, 3009, 3013]` |
| 目标动作区间 | `[3013, 3017]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3001**

![history_to_future_action_000091 frame 3001](images/history_to_future_action_000091_00.jpg)

**图 2，帧 3005**

![history_to_future_action_000091 frame 3005](images/history_to_future_action_000091_01.jpg)

**图 3，帧 3009**

![history_to_future_action_000091 frame 3009](images/history_to_future_action_000091_02.jpg)

**图 4，帧 3013**

![history_to_future_action_000091 frame 3013](images/history_to_future_action_000091_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -23 -76 D ; Mouse -10 -49 D ; Mouse -13 -32 D ; Mouse -11 -18 D <|action_end|>
```

## history_to_future_action_000092

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-pink-molly-9c1feba8a470-20220130-112802` |
| 图片帧 | `[3259, 3263, 3267, 3271]` |
| 目标动作区间 | `[3271, 3275]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3259**

![history_to_future_action_000092 frame 3259](images/history_to_future_action_000092_00.jpg)

**图 2，帧 3263**

![history_to_future_action_000092 frame 3263](images/history_to_future_action_000092_01.jpg)

**图 3，帧 3267**

![history_to_future_action_000092 frame 3267](images/history_to_future_action_000092_02.jpg)

**图 4，帧 3271**

![history_to_future_action_000092 frame 3271](images/history_to_future_action_000092_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift ; shift ; shift ; shift <|action_end|>
```

## history_to_future_action_000093

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `woozy-ruby-ostrich-f52befc9df4e-20220228-014358` |
| 图片帧 | `[11, 15, 19, 23]` |
| 目标动作区间 | `[23, 27]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 11**

![history_to_future_action_000093 frame 11](images/history_to_future_action_000093_00.jpg)

**图 2，帧 15**

![history_to_future_action_000093 frame 15](images/history_to_future_action_000093_01.jpg)

**图 3，帧 19**

![history_to_future_action_000093 frame 19](images/history_to_future_action_000093_02.jpg)

**图 4，帧 23**

![history_to_future_action_000093 frame 23](images/history_to_future_action_000093_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W A ; W ; W ; W <|action_end|>
```

## history_to_future_action_000094

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `thirsty-lavender-koala-f153ac423f61-20220110-101251` |
| 图片帧 | `[8707, 8711, 8715, 8719]` |
| 目标动作区间 | `[8719, 8723]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8707**

![history_to_future_action_000094 frame 8707](images/history_to_future_action_000094_00.jpg)

**图 2，帧 8711**

![history_to_future_action_000094 frame 8711](images/history_to_future_action_000094_01.jpg)

**图 3，帧 8715**

![history_to_future_action_000094 frame 8715](images/history_to_future_action_000094_02.jpg)

**图 4，帧 8719**

![history_to_future_action_000094 frame 8719](images/history_to_future_action_000094_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 0 ; Mouse -2 1 ; Mouse -1 0 S ; S <|action_end|>
```

## history_to_future_action_000095

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `wiggy-aquamarine-tapir-208670055f1c-20220130-225012` |
| 图片帧 | `[3526, 3530, 3534, 3538]` |
| 目标动作区间 | `[3538, 3542]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3526**

![history_to_future_action_000095 frame 3526](images/history_to_future_action_000095_00.jpg)

**图 2，帧 3530**

![history_to_future_action_000095 frame 3530](images/history_to_future_action_000095_01.jpg)

**图 3，帧 3534**

![history_to_future_action_000095 frame 3534](images/history_to_future_action_000095_02.jpg)

**图 4，帧 3538**

![history_to_future_action_000095 frame 3538](images/history_to_future_action_000095_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 -1 W A shift ; W A shift ; Mouse 2 -4 W A shift ; Mouse 8 -6 W A shift <|action_end|>
```

## history_to_future_action_000096

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `Player198-e5241690c374-20211123-001638` |
| 图片帧 | `[946, 950, 954, 958]` |
| 目标动作区间 | `[958, 962]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 946**

![history_to_future_action_000096 frame 946](images/history_to_future_action_000096_00.jpg)

**图 2，帧 950**

![history_to_future_action_000096 frame 950](images/history_to_future_action_000096_01.jpg)

**图 3，帧 954**

![history_to_future_action_000096 frame 954](images/history_to_future_action_000096_02.jpg)

**图 4，帧 958**

![history_to_future_action_000096 frame 958](images/history_to_future_action_000096_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 2 W space ; Mouse 0 1 W ; Mouse 1 0 W ; Mouse 1 0 W <|action_end|>
```

## history_to_future_action_000097

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-29cc0a442269-20220219-173231` |
| 图片帧 | `[1583, 1587, 1591, 1595]` |
| 目标动作区间 | `[1595, 1599]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1583**

![history_to_future_action_000097 frame 1583](images/history_to_future_action_000097_00.jpg)

**图 2，帧 1587**

![history_to_future_action_000097 frame 1587](images/history_to_future_action_000097_01.jpg)

**图 3，帧 1591**

![history_to_future_action_000097 frame 1591](images/history_to_future_action_000097_02.jpg)

**图 4，帧 1595**

![history_to_future_action_000097 frame 1595](images/history_to_future_action_000097_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## history_to_future_action_000098

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220121-162619` |
| 图片帧 | `[4932, 4936, 4940, 4944]` |
| 目标动作区间 | `[4944, 4948]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4932**

![history_to_future_action_000098 frame 4932](images/history_to_future_action_000098_00.jpg)

**图 2，帧 4936**

![history_to_future_action_000098 frame 4936](images/history_to_future_action_000098_01.jpg)

**图 3，帧 4940**

![history_to_future_action_000098 frame 4940](images/history_to_future_action_000098_02.jpg)

**图 4，帧 4944**

![history_to_future_action_000098 frame 4944](images/history_to_future_action_000098_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 12 6 ; Mouse 11 2 ; Mouse 2 0 ; Mouse -8 -1 <|action_end|>
```

## history_to_future_action_000099

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `wiggy-aquamarine-tapir-a6bd1142348c-20220129-173601` |
| 图片帧 | `[3765, 3769, 3773, 3777]` |
| 目标动作区间 | `[3777, 3781]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3765**

![history_to_future_action_000099 frame 3765](images/history_to_future_action_000099_00.jpg)

**图 2，帧 3769**

![history_to_future_action_000099 frame 3769](images/history_to_future_action_000099_01.jpg)

**图 3，帧 3773**

![history_to_future_action_000099 frame 3773](images/history_to_future_action_000099_02.jpg)

**图 4，帧 3777**

![history_to_future_action_000099 frame 3777](images/history_to_future_action_000099_03.jpg)

### 问题

The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 24 28 ; Mouse 0 1 ; Mouse -8 -4 ; Mouse -27 -19 <|action_end|>
```

## single_frame_intent_to_action_000000

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-dbe110142113-20220209-195346` |
| 图片帧 | `[2621]` |
| 目标动作区间 | `[2621, 2625]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2621**

![single_frame_intent_to_action_000000 frame 2621](images/single_frame_intent_to_action_000000_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 6 -4 ; Mouse 6 -8 ; Mouse 0 -6 ; Mouse -1 -1 <|action_end|>
```

## single_frame_intent_to_action_000001

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player701-9a4b2b341d68-20220114-151954` |
| 图片帧 | `[313]` |
| 目标动作区间 | `[313, 317]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 313**

![single_frame_intent_to_action_000001 frame 313](images/single_frame_intent_to_action_000001_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000002

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220212-144445` |
| 图片帧 | `[8391]` |
| 目标动作区间 | `[8391, 8395]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8391**

![single_frame_intent_to_action_000002 frame 8391](images/single_frame_intent_to_action_000002_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ;  ; Mouse 0 7 ; Mouse -3 11 <|action_end|>
```

## single_frame_intent_to_action_000003

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-dbe110142113-20220209-195346` |
| 图片帧 | `[3135]` |
| 目标动作区间 | `[3135, 3139]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3135**

![single_frame_intent_to_action_000003 frame 3135](images/single_frame_intent_to_action_000003_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 14 -1 ; Mouse 6 2 ; Mouse 4 4 ; Mouse 4 10 <|action_end|>
```

## single_frame_intent_to_action_000004

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220309-192013` |
| 图片帧 | `[5384]` |
| 目标动作区间 | `[5384, 5388]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5384**

![single_frame_intent_to_action_000004 frame 5384](images/single_frame_intent_to_action_000004_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 14 -5 shift ; Mouse 8 -2 shift ; Mouse 2 0 D shift ; D shift <|action_end|>
```

## single_frame_intent_to_action_000005

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-d41bfb5ca7bb-20220227-082331` |
| 图片帧 | `[270]` |
| 目标动作区间 | `[270, 274]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 270**

![single_frame_intent_to_action_000005 frame 270](images/single_frame_intent_to_action_000005_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 10 39 ; Mouse 209 86 ; Mouse 118 17 ; Mouse 162 12 <|action_end|>
```

## single_frame_intent_to_action_000006

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-pink-molly-81ac59c6a2ef-20220113-121309` |
| 图片帧 | `[3672]` |
| 目标动作区间 | `[3672, 3676]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3672**

![single_frame_intent_to_action_000006 frame 3672](images/single_frame_intent_to_action_000006_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 38 15 A ; Mouse 32 11 A ; Mouse 33 11 A ; Mouse 29 11 A <|action_end|>
```

## single_frame_intent_to_action_000007

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `thirsty-lavender-koala-b343a535c597-20220224-203207` |
| 图片帧 | `[28489]` |
| 目标动作区间 | `[28489, 28493]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 28489**

![single_frame_intent_to_action_000007 frame 28489](images/single_frame_intent_to_action_000007_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -229 49 ; Mouse -63 36 ; Mouse -4 12 ; Mouse 8 -8 <|action_end|>
```

## single_frame_intent_to_action_000008

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `pokey-cyan-spitz-a80588c26213-20220108-191056` |
| 图片帧 | `[2702]` |
| 目标动作区间 | `[2702, 2706]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2702**

![single_frame_intent_to_action_000008 frame 2702](images/single_frame_intent_to_action_000008_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; Mouse 1 -2 W ; Mouse 0 -2 W ; Mouse -11 19 <|action_end|>
```

## single_frame_intent_to_action_000009

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-39e4e3c2309d-20220224-153226` |
| 图片帧 | `[36]` |
| 目标动作区间 | `[36, 40]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 36**

![single_frame_intent_to_action_000009 frame 36](images/single_frame_intent_to_action_000009_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 12 ; Mouse -7 14 ; Mouse -9 12 ; Mouse -20 11 <|action_end|>
```

## single_frame_intent_to_action_000010

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-5d9ab504739c-20220213-022859` |
| 图片帧 | `[31162]` |
| 目标动作区间 | `[31162, 31166]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 31162**

![single_frame_intent_to_action_000010 frame 31162](images/single_frame_intent_to_action_000010_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 4 shift ; Mouse 0 1 shift ; shift ; Mouse 0 1 shift MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000011

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player565-f153ac423f61-20220204-214251` |
| 图片帧 | `[5021]` |
| 目标动作区间 | `[5021, 5025]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5021**

![single_frame_intent_to_action_000011 frame 5021](images/single_frame_intent_to_action_000011_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; S space ; Mouse 8 3 S space ; Mouse 92 44 S D ; Mouse 12 11 S D <|action_end|>
```

## single_frame_intent_to_action_000012

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220205-122419` |
| 图片帧 | `[4115]` |
| 目标动作区间 | `[4115, 4119]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4115**

![single_frame_intent_to_action_000012 frame 4115](images/single_frame_intent_to_action_000012_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; Mouse 3 0 W <|action_end|>
```

## single_frame_intent_to_action_000013

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-0d2bec37e4e8-20220131-140232` |
| 图片帧 | `[12631]` |
| 目标动作区间 | `[12631, 12635]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12631**

![single_frame_intent_to_action_000013 frame 12631](images/single_frame_intent_to_action_000013_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 8 W space ; Mouse 4 5 W ; Mouse 5 8 ; Mouse 1 10 <|action_end|>
```

## single_frame_intent_to_action_000014

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player886-f153ac423f61-20220219-014603` |
| 图片帧 | `[5119]` |
| 目标动作区间 | `[5119, 5123]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5119**

![single_frame_intent_to_action_000014 frame 5119](images/single_frame_intent_to_action_000014_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 9 4 W D ; W ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000015

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `tasty-brass-devil-f153ac423f61-20220109-234930` |
| 图片帧 | `[1861]` |
| 目标动作区间 | `[1861, 1865]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1861**

![single_frame_intent_to_action_000015 frame 1861](images/single_frame_intent_to_action_000015_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000016

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220117-210843` |
| 图片帧 | `[3162]` |
| 目标动作区间 | `[3162, 3166]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3162**

![single_frame_intent_to_action_000016 frame 3162](images/single_frame_intent_to_action_000016_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 58 -6 W MouseRight ; W MouseRight ; W MouseRight ; W MouseRight <|action_end|>
```

## single_frame_intent_to_action_000017

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-400005035101-20220206-063620` |
| 图片帧 | `[4955]` |
| 目标动作区间 | `[4955, 4959]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4955**

![single_frame_intent_to_action_000017 frame 4955](images/single_frame_intent_to_action_000017_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -14 -36 W D space ctrl ; Mouse -17 -45 W D space ctrl ; Mouse -16 -38 W D space ctrl ; Mouse -5 -14 W D space ctrl <|action_end|>
```

## single_frame_intent_to_action_000018

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-560a75b81c57-20220301-060717` |
| 图片帧 | `[12922]` |
| 目标动作区间 | `[12922, 12926]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12922**

![single_frame_intent_to_action_000018 frame 12922](images/single_frame_intent_to_action_000018_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse -6 -4 W ctrl MouseRight ; Mouse 2 -17 W ctrl MouseRight ; Mouse 16 -12 W ctrl <|action_end|>
```

## single_frame_intent_to_action_000019

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-pink-molly-f153ac423f61-20220127-123957` |
| 图片帧 | `[3132]` |
| 目标动作区间 | `[3132, 3136]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3132**

![single_frame_intent_to_action_000019 frame 3132](images/single_frame_intent_to_action_000019_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000020

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `trippy-red-llama-f153ac423f61-20220203-160137` |
| 图片帧 | `[9470]` |
| 目标动作区间 | `[9470, 9474]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9470**

![single_frame_intent_to_action_000020 frame 9470](images/single_frame_intent_to_action_000020_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 W ; Mouse 1 1 W ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000021

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-3074e7f751e9-20220123-194843` |
| 图片帧 | `[4439]` |
| 目标动作区间 | `[4439, 4443]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4439**

![single_frame_intent_to_action_000021 frame 4439](images/single_frame_intent_to_action_000021_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 5 W D ; Mouse -6 5 W D ; Mouse -3 7 W ; Mouse -1 9 W <|action_end|>
```

## single_frame_intent_to_action_000022

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `tasty-brass-devil-a47f39f57c24-20220305-194222` |
| 图片帧 | `[4140]` |
| 目标动作区间 | `[4140, 4144]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4140**

![single_frame_intent_to_action_000022 frame 4140](images/single_frame_intent_to_action_000022_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 10 0 ; Mouse 10 -1 ; Mouse 2 -1 ; Mouse 1 -1 7 <|action_end|>
```

## single_frame_intent_to_action_000023

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-1466c62bae85-20220222-163515` |
| 图片帧 | `[7789]` |
| 目标动作区间 | `[7789, 7793]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7789**

![single_frame_intent_to_action_000023 frame 7789](images/single_frame_intent_to_action_000023_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -4 0 W space ctrl ; W space ctrl ; Mouse -15 9 W space ctrl ; Mouse -80 29 W space ctrl <|action_end|>
```

## single_frame_intent_to_action_000024

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-eef151dc62a0-20211224-180136` |
| 图片帧 | `[15207]` |
| 目标动作区间 | `[15207, 15211]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 15207**

![single_frame_intent_to_action_000024 frame 15207](images/single_frame_intent_to_action_000024_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 50 20 ; Mouse 49 13 ; Mouse 20 4 ; Mouse 3 0 <|action_end|>
```

## single_frame_intent_to_action_000025

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-cee19105e3a0-20211229-161244` |
| 图片帧 | `[4541]` |
| 目标动作区间 | `[4541, 4545]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4541**

![single_frame_intent_to_action_000025 frame 4541](images/single_frame_intent_to_action_000025_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 -2 W ; W ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000026

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-f153ac423f61-20220228-011458` |
| 图片帧 | `[124]` |
| 目标动作区间 | `[124, 128]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 124**

![single_frame_intent_to_action_000026 frame 124](images/single_frame_intent_to_action_000026_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000027

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-cb0d36f9697e-20220206-060440` |
| 图片帧 | `[315]` |
| 目标动作区间 | `[315, 319]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 315**

![single_frame_intent_to_action_000027 frame 315](images/single_frame_intent_to_action_000027_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; S ctrl ; S ctrl ; S ; S <|action_end|>
```

## single_frame_intent_to_action_000028

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-f153ac423f61-20220224-145318` |
| 图片帧 | `[3138]` |
| 目标动作区间 | `[3138, 3142]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3138**

![single_frame_intent_to_action_000028 frame 3138](images/single_frame_intent_to_action_000028_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 2 -1 MouseLeft ; Mouse 21 -2 MouseLeft ; Mouse 45 -2 MouseLeft ; Mouse 72 -5 W MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000029

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220118-221055` |
| 图片帧 | `[6710]` |
| 目标动作区间 | `[6710, 6714]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6710**

![single_frame_intent_to_action_000029 frame 6710](images/single_frame_intent_to_action_000029_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 15 -1 W 5 ; Mouse 7 -1 W 5 ; Mouse 2 0 W ; Mouse 8 0 W <|action_end|>
```

## single_frame_intent_to_action_000030

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-f153ac423f61-20220228-215821` |
| 图片帧 | `[1745]` |
| 目标动作区间 | `[1745, 1749]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1745**

![single_frame_intent_to_action_000030 frame 1745](images/single_frame_intent_to_action_000030_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 40 4 ; Mouse 20 0 ; Mouse 4 0 ; shift <|action_end|>
```

## single_frame_intent_to_action_000031

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20211226-230601` |
| 图片帧 | `[4638]` |
| 目标动作区间 | `[4638, 4642]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4638**

![single_frame_intent_to_action_000031 frame 4638](images/single_frame_intent_to_action_000031_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 52 -117 W D ; Mouse 9 -136 W ; Mouse 5 -15 W ; Mouse -5 3 W <|action_end|>
```

## single_frame_intent_to_action_000032

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player294-f153ac423f61-20211222-213746` |
| 图片帧 | `[6590]` |
| 目标动作区间 | `[6590, 6594]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6590**

![single_frame_intent_to_action_000032 frame 6590](images/single_frame_intent_to_action_000032_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## single_frame_intent_to_action_000033

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player509-f153ac423f61-20220113-124444` |
| 图片帧 | `[640]` |
| 目标动作区间 | `[640, 644]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 640**

![single_frame_intent_to_action_000033 frame 640](images/single_frame_intent_to_action_000033_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 6 W MouseRight ; Mouse 3 12 W space ; Mouse 3 31 W space ; Mouse -13 49 W space <|action_end|>
```

## single_frame_intent_to_action_000034

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220206-215848` |
| 图片帧 | `[5034]` |
| 目标动作区间 | `[5034, 5038]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5034**

![single_frame_intent_to_action_000034 frame 5034](images/single_frame_intent_to_action_000034_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000035

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-2dc1a49559f5-20220128-120017` |
| 图片帧 | `[2989]` |
| 目标动作区间 | `[2989, 2993]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2989**

![single_frame_intent_to_action_000035 frame 2989](images/single_frame_intent_to_action_000035_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -17 W ; Mouse 3 -22 W ; Mouse 8 -28 ; Mouse 9 -23 <|action_end|>
```

## single_frame_intent_to_action_000036

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-f153ac423f61-20220215-153342` |
| 图片帧 | `[14036]` |
| 目标动作区间 | `[14036, 14040]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 14036**

![single_frame_intent_to_action_000036 frame 14036](images/single_frame_intent_to_action_000036_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 11 0 W ; W ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000037

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-90a647e39947-20220308-092828` |
| 图片帧 | `[7780]` |
| 目标动作区间 | `[7780, 7784]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7780**

![single_frame_intent_to_action_000037 frame 7780](images/single_frame_intent_to_action_000037_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; A ; A MouseLeft ; MouseLeft ; Mouse 5 -2 MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000038

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `jumpy-denim-lion-20f4188b8a75-20220225-061816` |
| 图片帧 | `[3714]` |
| 目标动作区间 | `[3714, 3718]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3714**

![single_frame_intent_to_action_000038 frame 3714](images/single_frame_intent_to_action_000038_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift ; Mouse 2 0 shift ; Mouse 2 1 shift <|action_end|>
```

## single_frame_intent_to_action_000039

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `hazy-thistle-chipmunk-f153ac423f61-20220102-214129` |
| 图片帧 | `[1891]` |
| 目标动作区间 | `[1891, 1895]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1891**

![single_frame_intent_to_action_000039 frame 1891](images/single_frame_intent_to_action_000039_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift ; shift ; shift <|action_end|>
```

## single_frame_intent_to_action_000040

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `pokey-cyan-spitz-f153ac423f61-20220310-230249` |
| 图片帧 | `[4912]` |
| 目标动作区间 | `[4912, 4916]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4912**

![single_frame_intent_to_action_000040 frame 4912](images/single_frame_intent_to_action_000040_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## single_frame_intent_to_action_000041

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-04e11164aa4c-20220214-001337` |
| 图片帧 | `[37]` |
| 目标动作区间 | `[37, 41]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 37**

![single_frame_intent_to_action_000041 frame 37](images/single_frame_intent_to_action_000041_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -2 W ; W ; Mouse -1 0 W ; W <|action_end|>
```

## single_frame_intent_to_action_000042

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `tasty-brass-devil-304bb5ac4e17-20220126-004940` |
| 图片帧 | `[2330]` |
| 目标动作区间 | `[2330, 2334]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2330**

![single_frame_intent_to_action_000042 frame 2330](images/single_frame_intent_to_action_000042_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 -2 ; Mouse -5 -1 ; Mouse -3 -2 ; Mouse -4 -3 4 <|action_end|>
```

## single_frame_intent_to_action_000043

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `tasty-brass-devil-f153ac423f61-20220224-115401` |
| 图片帧 | `[9303]` |
| 目标动作区间 | `[9303, 9307]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9303**

![single_frame_intent_to_action_000043 frame 9303](images/single_frame_intent_to_action_000043_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000044

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `pokey-cyan-spitz-f153ac423f61-20220107-230310` |
| 图片帧 | `[24195]` |
| 目标动作区间 | `[24195, 24199]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 24195**

![single_frame_intent_to_action_000044 frame 24195](images/single_frame_intent_to_action_000044_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W MouseRight ; MouseRight ; Mouse -1 -2 ; Mouse -1 3 <|action_end|>
```

## single_frame_intent_to_action_000045

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-a02fa6166716-20220115-073752` |
| 图片帧 | `[1630]` |
| 目标动作区间 | `[1630, 1634]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1630**

![single_frame_intent_to_action_000045 frame 1630](images/single_frame_intent_to_action_000045_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 43 -12 ; Mouse 86 -15 ; Mouse 97 -24 ; Mouse 33 -27 <|action_end|>
```

## single_frame_intent_to_action_000046

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `jumpy-denim-lion-146c08be38eb-20220124-034039` |
| 图片帧 | `[18768]` |
| 目标动作区间 | `[18768, 18772]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 18768**

![single_frame_intent_to_action_000046 frame 18768](images/single_frame_intent_to_action_000046_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -28 -4 shift ; Mouse -20 12 shift ; Mouse -3 4 shift ; Mouse 3 4 shift <|action_end|>
```

## single_frame_intent_to_action_000047

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220124-170210` |
| 图片帧 | `[2981]` |
| 目标动作区间 | `[2981, 2985]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2981**

![single_frame_intent_to_action_000047 frame 2981](images/single_frame_intent_to_action_000047_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift ; shift MouseRight ; shift MouseRight ; shift MouseRight <|action_end|>
```

## single_frame_intent_to_action_000048

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220308-132032` |
| 图片帧 | `[1563]` |
| 目标动作区间 | `[1563, 1567]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1563**

![single_frame_intent_to_action_000048 frame 1563](images/single_frame_intent_to_action_000048_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ; W space ; W space ; W space <|action_end|>
```

## single_frame_intent_to_action_000049

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player433-f153ac423f61-20211127-232918` |
| 图片帧 | `[5052]` |
| 目标动作区间 | `[5052, 5056]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 5052**

![single_frame_intent_to_action_000049 frame 5052](images/single_frame_intent_to_action_000049_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -17 15 W ctrl ; Mouse -13 10 W ctrl ; Mouse -9 3 W ctrl ; Mouse -3 1 W space ctrl <|action_end|>
```

## single_frame_intent_to_action_000050

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `wiggy-aquamarine-tapir-cbdf9b72f292-20220303-220411` |
| 图片帧 | `[11211]` |
| 目标动作区间 | `[11211, 11215]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 11211**

![single_frame_intent_to_action_000050 frame 11211](images/single_frame_intent_to_action_000050_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W D ctrl ; Mouse -15 -1 W D ctrl ; Mouse -28 -13 W D ctrl ; Mouse -43 -17 W D ctrl <|action_end|>
```

## single_frame_intent_to_action_000051

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-1ce2c0153e89-20220113-143515` |
| 图片帧 | `[2270]` |
| 目标动作区间 | `[2270, 2274]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2270**

![single_frame_intent_to_action_000051 frame 2270](images/single_frame_intent_to_action_000051_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 15 27 MouseLeft ; Mouse 9 43 MouseLeft ; Mouse 5 40 ; Mouse 7 13 <|action_end|>
```

## single_frame_intent_to_action_000052

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `jumpy-denim-lion-1c4403d4ac27-20220301-060256` |
| 图片帧 | `[14172]` |
| 目标动作区间 | `[14172, 14176]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 14172**

![single_frame_intent_to_action_000052 frame 14172](images/single_frame_intent_to_action_000052_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W A MouseLeft ; W A MouseLeft ; W A MouseLeft ; W A MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000053

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `tasty-brass-devil-f153ac423f61-20220228-000003` |
| 图片帧 | `[3161]` |
| 目标动作区间 | `[3161, 3165]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3161**

![single_frame_intent_to_action_000053 frame 3161](images/single_frame_intent_to_action_000053_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ;  ;  ;  <|action_end|>
```

## single_frame_intent_to_action_000054

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player543-2fa34fc69e53-20220212-145145` |
| 图片帧 | `[9605]` |
| 目标动作区间 | `[9605, 9609]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9605**

![single_frame_intent_to_action_000054 frame 9605](images/single_frame_intent_to_action_000054_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 134 -25 W A D ; Mouse -3 -2 W D ; Mouse -5 3 W D ; W D <|action_end|>
```

## single_frame_intent_to_action_000055

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-1837ac1029d9-20220102-010429` |
| 图片帧 | `[2333]` |
| 目标动作区间 | `[2333, 2337]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2333**

![single_frame_intent_to_action_000055 frame 2333](images/single_frame_intent_to_action_000055_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 2 MouseLeft ; Mouse 4 3 W MouseLeft ; Mouse 16 8 W MouseLeft ; Mouse 27 2 W <|action_end|>
```

## single_frame_intent_to_action_000056

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `bumpy-pumpkin-dunker-f153ac423f61-20220117-223356` |
| 图片帧 | `[4433]` |
| 目标动作区间 | `[4433, 4437]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4433**

![single_frame_intent_to_action_000056 frame 4433](images/single_frame_intent_to_action_000056_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 0 -3 W ; Mouse 0 -7 W ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000057

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-cee19105e3a0-20211229-161244` |
| 图片帧 | `[843]` |
| 目标动作区间 | `[843, 847]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 843**

![single_frame_intent_to_action_000057 frame 843](images/single_frame_intent_to_action_000057_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -69 13 ; Mouse -16 -2 ; W ; W <|action_end|>
```

## single_frame_intent_to_action_000058

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `jumpy-denim-lion-629b6234c3ab-20220207-171416` |
| 图片帧 | `[1601]` |
| 目标动作区间 | `[1601, 1605]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1601**

![single_frame_intent_to_action_000058 frame 1601](images/single_frame_intent_to_action_000058_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -8 0 W space ; Mouse -16 0 W ; Mouse -30 0 W ; Mouse -34 0 W <|action_end|>
```

## single_frame_intent_to_action_000059

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `trippy-red-llama-ad39205a77f9-20220201-195516` |
| 图片帧 | `[1742]` |
| 目标动作区间 | `[1742, 1746]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1742**

![single_frame_intent_to_action_000059 frame 1742](images/single_frame_intent_to_action_000059_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 1 W space ; W space ; W space ; W space <|action_end|>
```

## single_frame_intent_to_action_000060

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-24c6b77bc4c8-20220215-053356` |
| 图片帧 | `[487]` |
| 目标动作区间 | `[487, 491]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 487**

![single_frame_intent_to_action_000060 frame 487](images/single_frame_intent_to_action_000060_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## single_frame_intent_to_action_000061

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player325-f153ac423f61-20211205-212153` |
| 图片帧 | `[12350]` |
| 目标动作区间 | `[12350, 12354]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12350**

![single_frame_intent_to_action_000061 frame 12350](images/single_frame_intent_to_action_000061_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 0 ; 3 ; 3 ; 3 <|action_end|>
```

## single_frame_intent_to_action_000062

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-45c942cb5f8b-20220201-040454` |
| 图片帧 | `[8521]` |
| 目标动作区间 | `[8521, 8525]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8521**

![single_frame_intent_to_action_000062 frame 8521](images/single_frame_intent_to_action_000062_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 0 A ; A ; A ; A <|action_end|>
```

## single_frame_intent_to_action_000063

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220302-170137` |
| 图片帧 | `[2937]` |
| 目标动作区间 | `[2937, 2941]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2937**

![single_frame_intent_to_action_000063 frame 2937](images/single_frame_intent_to_action_000063_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -27 -20 MouseLeft ; Mouse -44 -31 MouseLeft ; Mouse -145 -74 ; Mouse -37 -9 <|action_end|>
```

## single_frame_intent_to_action_000064

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220222-225135` |
| 图片帧 | `[2317]` |
| 目标动作区间 | `[2317, 2321]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2317**

![single_frame_intent_to_action_000064 frame 2317](images/single_frame_intent_to_action_000064_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 102 2 MouseLeft ; Mouse 52 0 MouseLeft ; Mouse 2 -1 MouseLeft ; MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000065

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220228-215934` |
| 图片帧 | `[4573]` |
| 目标动作区间 | `[4573, 4577]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4573**

![single_frame_intent_to_action_000065 frame 4573](images/single_frame_intent_to_action_000065_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -3 0 W D ; Mouse -11 -2 W D ; Mouse -24 -7 W ; Mouse -45 -15 W <|action_end|>
```

## single_frame_intent_to_action_000066

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-viridian-beaver-f153ac423f61-20220221-182757` |
| 图片帧 | `[6047]` |
| 目标动作区间 | `[6047, 6051]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6047**

![single_frame_intent_to_action_000066 frame 6047](images/single_frame_intent_to_action_000066_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 5 -3 W ; Mouse 15 -1 W ; Mouse 14 4 W ; Mouse 48 9 W <|action_end|>
```

## single_frame_intent_to_action_000067

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220103-221314` |
| 图片帧 | `[8338]` |
| 目标动作区间 | `[8338, 8342]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8338**

![single_frame_intent_to_action_000067 frame 8338](images/single_frame_intent_to_action_000067_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 23 0 ; Mouse 6 0 ; Mouse 18 -4 ; Mouse 7 -3 <|action_end|>
```

## single_frame_intent_to_action_000068

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-7a214a94ed96-20211231-005857` |
| 图片帧 | `[2156]` |
| 目标动作区间 | `[2156, 2160]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2156**

![single_frame_intent_to_action_000068 frame 2156](images/single_frame_intent_to_action_000068_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 0 W ctrl ; W ctrl ; W ctrl ; W ctrl <|action_end|>
```

## single_frame_intent_to_action_000069

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-a554ba8a0d0c-20220308-072403` |
| 图片帧 | `[2459]` |
| 目标动作区间 | `[2459, 2463]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2459**

![single_frame_intent_to_action_000069 frame 2459](images/single_frame_intent_to_action_000069_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 -2 MouseLeft ; Mouse -1 0 MouseLeft ; Mouse -1 0 MouseLeft ; Mouse -4 0 MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000070

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `woozy-ruby-ostrich-2394f3d72b16-20220211-033846` |
| 图片帧 | `[7406]` |
| 目标动作区间 | `[7406, 7410]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 7406**

![single_frame_intent_to_action_000070 frame 7406](images/single_frame_intent_to_action_000070_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000071

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player427-f153ac423f61-20211222-214609` |
| 图片帧 | `[696]` |
| 目标动作区间 | `[696, 700]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 696**

![single_frame_intent_to_action_000071 frame 696](images/single_frame_intent_to_action_000071_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ; MouseRight ; MouseRight ; space <|action_end|>
```

## single_frame_intent_to_action_000072

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `snippy-chartreuse-mastiff-c636490d741c-20220227-150130` |
| 图片帧 | `[9801]` |
| 目标动作区间 | `[9801, 9805]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 9801**

![single_frame_intent_to_action_000072 frame 9801](images/single_frame_intent_to_action_000072_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ; MouseRight ; MouseRight ; MouseRight <|action_end|>
```

## single_frame_intent_to_action_000073

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `wiggy-aquamarine-tapir-7d1389103c14-20220222-183334` |
| 图片帧 | `[778]` |
| 目标动作区间 | `[778, 782]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 778**

![single_frame_intent_to_action_000073 frame 778](images/single_frame_intent_to_action_000073_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 28 19 shift ; Mouse 88 169 shift ; Mouse 74 148 shift ; Mouse 16 87 shift <|action_end|>
```

## single_frame_intent_to_action_000074

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-e70ba6e3e3ed-20211224-130730` |
| 图片帧 | `[2117]` |
| 目标动作区间 | `[2117, 2121]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2117**

![single_frame_intent_to_action_000074 frame 2117](images/single_frame_intent_to_action_000074_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W space ; W space ; W space ; W <|action_end|>
```

## single_frame_intent_to_action_000075

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player543-2fa34fc69e53-20220212-143959` |
| 图片帧 | `[827]` |
| 目标动作区间 | `[827, 831]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 827**

![single_frame_intent_to_action_000075 frame 827](images/single_frame_intent_to_action_000075_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -87 10 W MouseRight ; Mouse -33 5 W ; W ; Mouse 37 -4 W <|action_end|>
```

## single_frame_intent_to_action_000076

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `squeaky-magnolia-ocelot-f153ac423f61-20220302-154945` |
| 图片帧 | `[250]` |
| 目标动作区间 | `[250, 254]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 250**

![single_frame_intent_to_action_000076 frame 250](images/single_frame_intent_to_action_000076_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift ; shift ; shift ; shift <|action_end|>
```

## single_frame_intent_to_action_000077

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-pink-molly-df6191006049-20220111-113652` |
| 图片帧 | `[178]` |
| 目标动作区间 | `[178, 182]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 178**

![single_frame_intent_to_action_000077 frame 178](images/single_frame_intent_to_action_000077_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 2 9 ; Mouse 0 7 ; Mouse 0 9 ; Mouse 0 6 <|action_end|>
```

## single_frame_intent_to_action_000078

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-9cf5531ec360-20211223-005027` |
| 图片帧 | `[2845]` |
| 目标动作区间 | `[2845, 2849]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2845**

![single_frame_intent_to_action_000078 frame 2845](images/single_frame_intent_to_action_000078_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse -1 0 MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000079

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `pokey-cyan-spitz-f153ac423f61-20211229-202427` |
| 图片帧 | `[52]` |
| 目标动作区间 | `[52, 56]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 52**

![single_frame_intent_to_action_000079 frame 52](images/single_frame_intent_to_action_000079_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -8 1 ; Mouse -5 3 ; Mouse -15 9 ; Mouse -7 4 <|action_end|>
```

## single_frame_intent_to_action_000080

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-55fad1d6ca5a-20220109-162933` |
| 图片帧 | `[12053]` |
| 目标动作区间 | `[12053, 12057]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 12053**

![single_frame_intent_to_action_000080 frame 12053](images/single_frame_intent_to_action_000080_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000081

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-pink-molly-4e97c3451821-20220125-141450` |
| 图片帧 | `[1419]` |
| 目标动作区间 | `[1419, 1423]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 1419**

![single_frame_intent_to_action_000081 frame 1419](images/single_frame_intent_to_action_000081_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -13 19 ; Mouse -10 25 ; Mouse -9 23 ; Mouse -5 29 <|action_end|>
```

## single_frame_intent_to_action_000082

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220115-140242` |
| 图片帧 | `[575]` |
| 目标动作区间 | `[575, 579]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 575**

![single_frame_intent_to_action_000082 frame 575](images/single_frame_intent_to_action_000082_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 6 5 ; Mouse 60 27 ; Mouse 69 29 ; Mouse 23 9 <|action_end|>
```

## single_frame_intent_to_action_000083

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `pokey-cyan-spitz-1ad85ee2e4ea-20220219-105517` |
| 图片帧 | `[412]` |
| 目标动作区间 | `[412, 416]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 412**

![single_frame_intent_to_action_000083 frame 412](images/single_frame_intent_to_action_000083_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W MouseLeft ; Mouse -4 -3 W MouseLeft ; Mouse -25 -28 W MouseLeft ; Mouse -53 -53 W <|action_end|>
```

## single_frame_intent_to_action_000084

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `wiggy-aquamarine-tapir-c5c7d9334fd2-20220303-222439` |
| 图片帧 | `[355]` |
| 目标动作区间 | `[355, 359]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 355**

![single_frame_intent_to_action_000084 frame 355](images/single_frame_intent_to_action_000084_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; Mouse -1 3 ; Mouse 0 -4 ; Mouse -1 -10 <|action_end|>
```

## single_frame_intent_to_action_000085

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `squeaky-magnolia-ocelot-11b9cc7e1fff-20220227-140852` |
| 图片帧 | `[8090]` |
| 目标动作区间 | `[8090, 8094]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 8090**

![single_frame_intent_to_action_000085 frame 8090](images/single_frame_intent_to_action_000085_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -15 7 MouseLeft ; MouseLeft ; MouseLeft ; Mouse 0 1 A <|action_end|>
```

## single_frame_intent_to_action_000086

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `tasty-brass-devil-de14d5f0a376-20220126-000020` |
| 图片帧 | `[406]` |
| 目标动作区间 | `[406, 410]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 406**

![single_frame_intent_to_action_000086 frame 406](images/single_frame_intent_to_action_000086_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000087

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-f10cfa71108f-20220128-000815` |
| 图片帧 | `[68]` |
| 目标动作区间 | `[68, 72]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 68**

![single_frame_intent_to_action_000087 frame 68](images/single_frame_intent_to_action_000087_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -20 10 ; Mouse 15 -12 ; Mouse 87 -60 ; Mouse 164 -80 <|action_end|>
```

## single_frame_intent_to_action_000088

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `whiny-ecru-cougar-a6470cea1776-20220121-153254` |
| 图片帧 | `[2857]` |
| 目标动作区间 | `[2857, 2861]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 2857**

![single_frame_intent_to_action_000088 frame 2857](images/single_frame_intent_to_action_000088_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -25 25 MouseLeft ; Mouse -8 7 MouseLeft ; MouseLeft ; Mouse 0 1 MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000089

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `Player757-f153ac423f61-20211202-221257` |
| 图片帧 | `[13511]` |
| 目标动作区间 | `[13511, 13515]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 13511**

![single_frame_intent_to_action_000089 frame 13511](images/single_frame_intent_to_action_000089_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 185 14 ; Mouse 347 -51 ; Mouse 412 -52 ; Mouse 188 -21 <|action_end|>
```

## single_frame_intent_to_action_000090

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-4f33abd7d22d-20220129-162059` |
| 图片帧 | `[4749]` |
| 目标动作区间 | `[4749, 4753]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4749**

![single_frame_intent_to_action_000090 frame 4749](images/single_frame_intent_to_action_000090_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

## single_frame_intent_to_action_000091

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-5cf3df68db9c-20220309-205307` |
| 图片帧 | `[4809]` |
| 目标动作区间 | `[4809, 4813]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4809**

![single_frame_intent_to_action_000091 frame 4809](images/single_frame_intent_to_action_000091_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 4 13 W ctrl MouseRight ; W ctrl ; Mouse 1 -5 W ctrl ; Mouse 6 -27 W ctrl <|action_end|>
```

## single_frame_intent_to_action_000092

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-74a57b1ea9e6-20220206-142015` |
| 图片帧 | `[6504]` |
| 目标动作区间 | `[6504, 6508]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 6504**

![single_frame_intent_to_action_000092 frame 6504](images/single_frame_intent_to_action_000092_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -1 14 W space MouseRight ; Mouse -6 28 W space MouseRight ; Mouse -6 11 W space MouseRight ; Mouse -49 15 W space MouseRight <|action_end|>
```

## single_frame_intent_to_action_000093

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-d208b0c3df45-20220224-035358` |
| 图片帧 | `[3]` |
| 目标动作区间 | `[3, 7]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3**

![single_frame_intent_to_action_000093 frame 3](images/single_frame_intent_to_action_000093_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -33 0 ; Mouse -126 0 ; Mouse -166 0 ; Mouse -82 0 <|action_end|>
```

## single_frame_intent_to_action_000094

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20220226-172113` |
| 图片帧 | `[237]` |
| 目标动作区间 | `[237, 241]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 237**

![single_frame_intent_to_action_000094 frame 237](images/single_frame_intent_to_action_000094_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift ; shift ; shift <|action_end|>
```

## single_frame_intent_to_action_000095

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `scaly-fuchsia-wasp-f153ac423f61-20220126-224310` |
| 图片帧 | `[4272]` |
| 目标动作区间 | `[4272, 4276]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 4272**

![single_frame_intent_to_action_000095 frame 4272](images/single_frame_intent_to_action_000095_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; W <|action_end|>
```

## single_frame_intent_to_action_000096

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `cheeky-cornflower-setter-f153ac423f61-20220226-213918` |
| 图片帧 | `[3306]` |
| 目标动作区间 | `[3306, 3310]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3306**

![single_frame_intent_to_action_000096 frame 3306](images/single_frame_intent_to_action_000096_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 15 26 MouseRight ; Mouse 19 14 MouseRight ; Mouse 74 3 MouseRight ; Mouse 44 -11 MouseRight <|action_end|>
```

## single_frame_intent_to_action_000097

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `shabby-pink-molly-58b7f75a14ce-20220405-130305` |
| 图片帧 | `[17353]` |
| 目标动作区间 | `[17353, 17357]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 17353**

![single_frame_intent_to_action_000097 frame 17353](images/single_frame_intent_to_action_000097_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 4 0 ; Mouse 1 -1 MouseRight ; Mouse 1 1 MouseRight ; Mouse 21 0 <|action_end|>
```

## single_frame_intent_to_action_000098

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-f153ac423f61-20211225-224732` |
| 图片帧 | `[3038]` |
| 目标动作区间 | `[3038, 3042]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 3038**

![single_frame_intent_to_action_000098 frame 3038](images/single_frame_intent_to_action_000098_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseRight ;  ; Mouse -1 0 ; Mouse -12 0 <|action_end|>
```

## single_frame_intent_to_action_000099

| 字段 | 内容 |
|---|---|
| 题型 | `single_frame_intent_to_action` |
| 来源 episode | `gimpy-jade-panda-345e83b833df-20211226-121717` |
| 图片帧 | `[17368]` |
| 目标动作区间 | `[17368, 17372]` |
| 初始训练准入 | `False` |
| 结构审核 | `pending` |

### 图片

**图 1，帧 17368**

![single_frame_intent_to_action_000099 frame 17368](images/single_frame_intent_to_action_000099_00.jpg)

### 问题

The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the next 200 ms that advances this intent. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```
