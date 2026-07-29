# MineStudio 轨迹训练题生成报告

> 本报告由出题流程自动生成。图片与参考动作来自真实 MineStudio 轨迹。
> 参考轨迹是一种人类示范，不是唯一正确答案。`answer_key.jsonl` 不应交给做题模型。

## 汇总

| 项目 | 数量 |
|---|---:|
| 候选题目 | 9 |
| 结构审核完成 | 9 |
| 结构审核通过 | 9 |

## demonstration_optimization_000000

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `snippy-chartreuse-mastiff-68b5723ce118-20220418-220340` |
| 图片帧 | `[2864, 2868, 2872, 2876]` |
| 目标动作区间 | `[2864, 2880]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 2864**

![demonstration_optimization_000000 frame 2864](images/demonstration_optimization_000000_00.jpg)

**图 2，帧 2868**

![demonstration_optimization_000000 frame 2868](images/demonstration_optimization_000000_01.jpg)

**图 3，帧 2872**

![demonstration_optimization_000000 frame 2872](images/demonstration_optimization_000000_02.jpg)

**图 4，帧 2876**

![demonstration_optimization_000000 frame 2876](images/demonstration_optimization_000000_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite the action sequence into a cleaner demonstration while preserving the visible intent and causal order. Remove isolated control noise, keep necessary movement and interaction, and return only a JSON array of valid action blocks.

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
<|action_start|> ; W MouseLeft ; W MouseLeft ; W MouseLeft ; W MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 6 W MouseLeft ; Mouse 1 22 W MouseLeft ; Mouse 0 26 MouseLeft ; Mouse 0 33 MouseLeft <|action_end|>
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
<|action_start|> ; W MouseLeft ; W MouseLeft ; W MouseLeft ; W MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 0 6 W MouseLeft ; Mouse 1 22 W MouseLeft ; Mouse 0 26 MouseLeft ; Mouse 0 33 MouseLeft <|action_end|>
```

### 结构校验结果

```json
{
  "id": "demonstration_optimization_000000",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## demonstration_optimization_000001

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `lanky-flax-dormouse-f569f2c0c2df-20220422-170154` |
| 图片帧 | `[8745, 8749, 8753, 8757]` |
| 目标动作区间 | `[8745, 8761]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 8745**

![demonstration_optimization_000001 frame 8745](images/demonstration_optimization_000001_00.jpg)

**图 2，帧 8749**

![demonstration_optimization_000001 frame 8749](images/demonstration_optimization_000001_01.jpg)

**图 3，帧 8753**

![demonstration_optimization_000001 frame 8753](images/demonstration_optimization_000001_02.jpg)

**图 4，帧 8757**

![demonstration_optimization_000001 frame 8757](images/demonstration_optimization_000001_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite the action sequence into a cleaner demonstration while preserving the visible intent and causal order. Remove isolated control noise, keep necessary movement and interaction, and return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 1 6 MouseRight ; MouseRight ; Mouse 1 -1 MouseRight ; Mouse 5 -12 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 10 -28 ; Mouse 14 -35 ; Mouse 16 -33 ; Mouse 12 -25 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 5 -14 W ; Mouse 3 -9 W ; Mouse 1 -8 W ; Mouse -3 -3 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -3 0 W ; Mouse -6 7 MouseLeft ; Mouse -1 1 MouseLeft ; S MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 6 MouseRight ; MouseRight ; Mouse 1 -1 MouseRight ; Mouse 5 -12 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 10 -28 ; Mouse 14 -35 ; Mouse 16 -33 ; Mouse 12 -25 W <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 5 -14 W ; Mouse 3 -9 W ; Mouse 1 -8 W ; Mouse -3 -3 W <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -3 0 W ; Mouse -6 7 MouseLeft ; Mouse -1 1 MouseLeft ; S MouseLeft <|action_end|>
```

### 结构校验结果

```json
{
  "id": "demonstration_optimization_000001",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## demonstration_optimization_000002

| 字段 | 内容 |
|---|---|
| 题型 | `demonstration_optimization` |
| 来源 episode | `gimpy-jade-panda-3f668b9b9a16-20220417-181524` |
| 图片帧 | `[554, 558, 562, 566]` |
| 目标动作区间 | `[554, 570]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 554**

![demonstration_optimization_000002 frame 554](images/demonstration_optimization_000002_00.jpg)

**图 2，帧 558**

![demonstration_optimization_000002 frame 558](images/demonstration_optimization_000002_01.jpg)

**图 3，帧 562**

![demonstration_optimization_000002 frame 562](images/demonstration_optimization_000002_02.jpg)

**图 4，帧 566**

![demonstration_optimization_000002 frame 566](images/demonstration_optimization_000002_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite the action sequence into a cleaner demonstration while preserving the visible intent and causal order. Remove isolated control noise, keep necessary movement and interaction, and return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 6 MouseLeft ; Mouse -2 6 ; Mouse -5 7 ; Mouse -20 12 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -12 6 ; Mouse 3 -7 ; Mouse 16 -18 ; Mouse 8 -5 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 2 -1 MouseLeft ; Mouse 1 -2 ;  ; Mouse -4 7 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -14 10 ; Mouse -23 10 ; Mouse -11 2 ; Mouse -18 -2 <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 6 MouseLeft ; Mouse -2 6 ; Mouse -5 7 ; Mouse -20 12 <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -12 6 ; Mouse 3 -7 ; Mouse 16 -18 ; Mouse 8 -5 <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 2 -1 MouseLeft ; Mouse 1 -2 ;  ; Mouse -4 7 <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse -14 10 ; Mouse -23 10 ; Mouse -11 2 ; Mouse -18 -2 <|action_end|>
```

### 结构校验结果

```json
{
  "id": "demonstration_optimization_000002",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## image_sequence_to_action_000000

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `squeaky-magnolia-ocelot-d4a075adc507-20220421-210034` |
| 图片帧 | `[1152, 1153, 1154, 1155, 1156]` |
| 目标动作区间 | `[1152, 1156]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 1152**

![image_sequence_to_action_000000 frame 1152](images/image_sequence_to_action_000000_00.jpg)

**图 2，帧 1153**

![image_sequence_to_action_000000 frame 1153](images/image_sequence_to_action_000000_01.jpg)

**图 3，帧 1154**

![image_sequence_to_action_000000 frame 1154](images/image_sequence_to_action_000000_02.jpg)

**图 4，帧 1155**

![image_sequence_to_action_000000 frame 1155](images/image_sequence_to_action_000000_03.jpg)

**图 5，帧 1156**

![image_sequence_to_action_000000 frame 1156](images/image_sequence_to_action_000000_04.jpg)

### 问题

The five images are consecutive Minecraft observations in chronological order across 200 ms. No action labels are provided. Infer one reasonable action sequence that could have produced the observed transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 1 4 W space ctrl ; Mouse 2 1 W space ctrl ; Mouse 1 0 W space ctrl ; Mouse 2 1 W space ctrl <|action_end|>
```

### 结构校验结果

```json
{
  "id": "image_sequence_to_action_000000",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## image_sequence_to_action_000001

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `lovely-persimmon-angora-e118fb40d762-20220414-220127` |
| 图片帧 | `[4606, 4607, 4608, 4609, 4610]` |
| 目标动作区间 | `[4606, 4610]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 4606**

![image_sequence_to_action_000001 frame 4606](images/image_sequence_to_action_000001_00.jpg)

**图 2，帧 4607**

![image_sequence_to_action_000001 frame 4607](images/image_sequence_to_action_000001_01.jpg)

**图 3，帧 4608**

![image_sequence_to_action_000001 frame 4608](images/image_sequence_to_action_000001_02.jpg)

**图 4，帧 4609**

![image_sequence_to_action_000001 frame 4609](images/image_sequence_to_action_000001_03.jpg)

**图 5，帧 4610**

![image_sequence_to_action_000001 frame 4610](images/image_sequence_to_action_000001_04.jpg)

### 问题

The five images are consecutive Minecraft observations in chronological order across 200 ms. No action labels are provided. Infer one reasonable action sequence that could have produced the observed transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -438 -186 MouseLeft ; Mouse -242 -137 MouseLeft ; Mouse 0 -10 MouseLeft ; Mouse 2 -11 MouseLeft <|action_end|>
```

### 结构校验结果

```json
{
  "id": "image_sequence_to_action_000001",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## image_sequence_to_action_000002

| 字段 | 内容 |
|---|---|
| 题型 | `image_sequence_to_action` |
| 来源 episode | `shabby-viridian-beaver-92ea1f7fac67-20220419-182607` |
| 图片帧 | `[1945, 1946, 1947, 1948, 1949]` |
| 目标动作区间 | `[1945, 1949]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 1945**

![image_sequence_to_action_000002 frame 1945](images/image_sequence_to_action_000002_00.jpg)

**图 2，帧 1946**

![image_sequence_to_action_000002 frame 1946](images/image_sequence_to_action_000002_01.jpg)

**图 3，帧 1947**

![image_sequence_to_action_000002 frame 1947](images/image_sequence_to_action_000002_02.jpg)

**图 4，帧 1948**

![image_sequence_to_action_000002 frame 1948](images/image_sequence_to_action_000002_03.jpg)

**图 5，帧 1949**

![image_sequence_to_action_000002 frame 1949](images/image_sequence_to_action_000002_04.jpg)

### 问题

The five images are consecutive Minecraft observations in chronological order across 200 ms. No action labels are provided. Infer one reasonable action sequence that could have produced the observed transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 12 4 ; Mouse 44 8 ; Mouse 76 4 ; Mouse 24 1 <|action_end|>
```

### 结构校验结果

```json
{
  "id": "image_sequence_to_action_000002",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## history_to_future_action_000000

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `gimpy-jade-panda-45dec39be8a7-20220423-102039` |
| 图片帧 | `[4418, 4422, 4426, 4430]` |
| 目标动作区间 | `[4430, 4434]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 4418**

![history_to_future_action_000000 frame 4418](images/history_to_future_action_000000_00.jpg)

**图 2，帧 4422**

![history_to_future_action_000000 frame 4422](images/history_to_future_action_000000_01.jpg)

**图 3，帧 4426**

![history_to_future_action_000000 frame 4426](images/history_to_future_action_000000_02.jpg)

**图 4，帧 4430**

![history_to_future_action_000000 frame 4430](images/history_to_future_action_000000_03.jpg)

### 问题

The images are past observations in chronological order and contain no action labels. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 结构校验结果

```json
{
  "id": "history_to_future_action_000000",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## history_to_future_action_000001

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `lovely-persimmon-angora-743ac0c64519-20220416-212236` |
| 图片帧 | `[2334, 2338, 2342, 2346]` |
| 目标动作区间 | `[2346, 2350]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 2334**

![history_to_future_action_000001 frame 2334](images/history_to_future_action_000001_00.jpg)

**图 2，帧 2338**

![history_to_future_action_000001 frame 2338](images/history_to_future_action_000001_01.jpg)

**图 3，帧 2342**

![history_to_future_action_000001 frame 2342](images/history_to_future_action_000001_02.jpg)

**图 4，帧 2346**

![history_to_future_action_000001 frame 2346](images/history_to_future_action_000001_03.jpg)

### 问题

The images are past observations in chronological order and contain no action labels. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

### 结构校验结果

```json
{
  "id": "history_to_future_action_000001",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

## history_to_future_action_000002

| 字段 | 内容 |
|---|---|
| 题型 | `history_to_future_action` |
| 来源 episode | `shabby-viridian-beaver-922bac605e0c-20220419-172533` |
| 图片帧 | `[8205, 8209, 8213, 8217]` |
| 目标动作区间 | `[8217, 8221]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 8205**

![history_to_future_action_000002 frame 8205](images/history_to_future_action_000002_00.jpg)

**图 2，帧 8209**

![history_to_future_action_000002 frame 8209](images/history_to_future_action_000002_01.jpg)

**图 3，帧 8213**

![history_to_future_action_000002 frame 8213](images/history_to_future_action_000002_02.jpg)

**图 4，帧 8217**

![history_to_future_action_000002 frame 8217](images/history_to_future_action_000002_03.jpg)

### 问题

The images are past observations in chronological order and contain no action labels. Infer one reasonable action sequence for the next 200 ms. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 40 -12 ; Mouse 26 -11 ; Mouse 1 0 MouseLeft ; S MouseLeft <|action_end|>
```

### 结构校验结果

```json
{
  "id": "history_to_future_action_000002",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

