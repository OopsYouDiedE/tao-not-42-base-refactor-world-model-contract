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
| 来源 episode | `sleepy-sangria-bat-fa0de4ef2478-20220423-082844` |
| 图片帧 | `[9796, 9800, 9804, 9808]` |
| 目标动作区间 | `[9796, 9812]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 9796**

![demonstration_optimization_000000 frame 9796](images/demonstration_optimization_000000_00.jpg)

**图 2，帧 9800**

![demonstration_optimization_000000 frame 9800](images/demonstration_optimization_000000_01.jpg)

**图 3，帧 9804**

![demonstration_optimization_000000 frame 9804](images/demonstration_optimization_000000_02.jpg)

**图 4，帧 9808**

![demonstration_optimization_000000 frame 9808](images/demonstration_optimization_000000_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite the action sequence into a cleaner demonstration while preserving the visible intent and causal order. Remove isolated control noise, keep necessary movement and interaction, and return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse -2 44 MouseLeft ; Mouse -3 24 MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; W MouseLeft ; W MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W MouseLeft ; W MouseLeft ; Mouse 0 -9 W MouseLeft ; Mouse 9 -26 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 8 -26 MouseLeft ; Mouse 2 -22 MouseLeft ; Mouse 0 -14 MouseLeft ; Mouse 0 -6 MouseLeft <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -2 44 MouseLeft ; Mouse -3 24 MouseLeft ; MouseLeft ; MouseLeft <|action_end|>
```

动作块 2：

```text
<|action_start|> ; MouseLeft ; MouseLeft ; W MouseLeft ; W MouseLeft <|action_end|>
```

动作块 3：

```text
<|action_start|> ; W MouseLeft ; W MouseLeft ; Mouse 0 -9 W MouseLeft ; Mouse 9 -26 MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; Mouse 8 -26 MouseLeft ; Mouse 2 -22 MouseLeft ; Mouse 0 -14 MouseLeft ; Mouse 0 -6 MouseLeft <|action_end|>
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
| 来源 episode | `gimpy-jade-panda-48eadb8bd054-20220413-210137` |
| 图片帧 | `[4165, 4169, 4173, 4177]` |
| 目标动作区间 | `[4165, 4181]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 4165**

![demonstration_optimization_000001 frame 4165](images/demonstration_optimization_000001_00.jpg)

**图 2，帧 4169**

![demonstration_optimization_000001 frame 4169](images/demonstration_optimization_000001_01.jpg)

**图 3，帧 4173**

![demonstration_optimization_000001 frame 4173](images/demonstration_optimization_000001_02.jpg)

**图 4，帧 4177**

![demonstration_optimization_000001 frame 4177](images/demonstration_optimization_000001_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite the action sequence into a cleaner demonstration while preserving the visible intent and causal order. Remove isolated control noise, keep necessary movement and interaction, and return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse -3 -2 W ctrl ; Mouse 22 21 W ctrl ; Mouse 0 1 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 W space ctrl ; Mouse -8 -6 W space ctrl ; Mouse 0 1 W ctrl ; Mouse -11 0 W ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -21 12 W ctrl ; Mouse -52 13 W space ctrl ; Mouse -36 6 W space ctrl ; Mouse -10 3 W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ctrl ; Mouse 0 1 W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; W ctrl ; Mouse -3 -2 W ctrl ; Mouse 22 21 W ctrl ; Mouse 0 1 W ctrl <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse -1 0 W space ctrl ; Mouse -8 -6 W space ctrl ; Mouse 0 1 W ctrl ; Mouse -11 0 W ctrl <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse -21 12 W ctrl ; Mouse -52 13 W space ctrl ; Mouse -36 6 W space ctrl ; Mouse -10 3 W ctrl <|action_end|>
```

动作块 4：

```text
<|action_start|> ; W space ctrl ; Mouse 0 1 W space ctrl ; W space ctrl ; W space ctrl <|action_end|>
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
| 来源 episode | `shabby-pink-molly-f35fc91db4d3-20220414-081351` |
| 图片帧 | `[2764, 2768, 2772, 2776]` |
| 目标动作区间 | `[2764, 2780]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 2764**

![demonstration_optimization_000002 frame 2764](images/demonstration_optimization_000002_00.jpg)

**图 2，帧 2768**

![demonstration_optimization_000002 frame 2768](images/demonstration_optimization_000002_01.jpg)

**图 3，帧 2772**

![demonstration_optimization_000002 frame 2772](images/demonstration_optimization_000002_02.jpg)

**图 4，帧 2776**

![demonstration_optimization_000002 frame 2776](images/demonstration_optimization_000002_03.jpg)

### 问题

The images and raw action blocks form one chronological Minecraft demonstration. Rewrite the action sequence into a cleaner demonstration while preserving the visible intent and causal order. Remove isolated control noise, keep necessary movement and interaction, and return only a JSON array of valid action blocks.

### 待优化的原始动作序列

动作块 1：

```text
<|action_start|> ; Mouse 8 -2 ; Mouse 15 -8 ; Mouse 8 -8 ; Mouse 5 -7 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 -8 shift ; Mouse 0 -8 shift ; Mouse 0 -2 shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 0 -1 shift ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift ; shift ; shift ; shift <|action_end|>
```

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse 8 -2 ; Mouse 15 -8 ; Mouse 8 -8 ; Mouse 5 -7 shift <|action_end|>
```

动作块 2：

```text
<|action_start|> ; Mouse 2 -8 shift ; Mouse 0 -8 shift ; Mouse 0 -2 shift ; shift <|action_end|>
```

动作块 3：

```text
<|action_start|> ; Mouse 0 -1 shift ; shift MouseLeft ; shift MouseLeft ; shift MouseLeft <|action_end|>
```

动作块 4：

```text
<|action_start|> ; shift ; shift ; shift ; shift <|action_end|>
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
| 来源 episode | `cheeky-cornflower-setter-6b3831c95bf8-20220417-112336` |
| 图片帧 | `[18433, 18434, 18435, 18436, 18437]` |
| 目标动作区间 | `[18433, 18437]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 18433**

![image_sequence_to_action_000000 frame 18433](images/image_sequence_to_action_000000_00.jpg)

**图 2，帧 18434**

![image_sequence_to_action_000000 frame 18434](images/image_sequence_to_action_000000_01.jpg)

**图 3，帧 18435**

![image_sequence_to_action_000000 frame 18435](images/image_sequence_to_action_000000_02.jpg)

**图 4，帧 18436**

![image_sequence_to_action_000000 frame 18436](images/image_sequence_to_action_000000_03.jpg)

**图 5，帧 18437**

![image_sequence_to_action_000000 frame 18437](images/image_sequence_to_action_000000_04.jpg)

### 问题

The five images are consecutive Minecraft observations in chronological order across 200 ms. No action labels are provided. Infer one reasonable action sequence that could have produced the observed transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; MouseLeft ; Mouse 0 12 MouseLeft ; Mouse -9 49 MouseLeft ; Mouse -10 40 MouseLeft <|action_end|>
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
| 来源 episode | `lovely-persimmon-angora-e10920a66232-20220416-140321` |
| 图片帧 | `[9213, 9214, 9215, 9216, 9217]` |
| 目标动作区间 | `[9213, 9217]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 9213**

![image_sequence_to_action_000001 frame 9213](images/image_sequence_to_action_000001_00.jpg)

**图 2，帧 9214**

![image_sequence_to_action_000001 frame 9214](images/image_sequence_to_action_000001_01.jpg)

**图 3，帧 9215**

![image_sequence_to_action_000001 frame 9215](images/image_sequence_to_action_000001_02.jpg)

**图 4，帧 9216**

![image_sequence_to_action_000001 frame 9216](images/image_sequence_to_action_000001_03.jpg)

**图 5，帧 9217**

![image_sequence_to_action_000001 frame 9217](images/image_sequence_to_action_000001_04.jpg)

### 问题

The five images are consecutive Minecraft observations in chronological order across 200 ms. No action labels are provided. Infer one reasonable action sequence that could have produced the observed transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; Mouse -5 1 MouseLeft ; Mouse -4 1 MouseLeft ; Mouse -2 0 MouseLeft ; MouseLeft <|action_end|>
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
| 来源 episode | `shabby-viridian-beaver-922bac605e0c-20220419-172533` |
| 图片帧 | `[972, 973, 974, 975, 976]` |
| 目标动作区间 | `[972, 976]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 972**

![image_sequence_to_action_000002 frame 972](images/image_sequence_to_action_000002_00.jpg)

**图 2，帧 973**

![image_sequence_to_action_000002 frame 973](images/image_sequence_to_action_000002_01.jpg)

**图 3，帧 974**

![image_sequence_to_action_000002 frame 974](images/image_sequence_to_action_000002_02.jpg)

**图 4，帧 975**

![image_sequence_to_action_000002 frame 975](images/image_sequence_to_action_000002_03.jpg)

**图 5，帧 976**

![image_sequence_to_action_000002 frame 976](images/image_sequence_to_action_000002_04.jpg)

### 问题

The five images are consecutive Minecraft observations in chronological order across 200 ms. No action labels are provided. Infer one reasonable action sequence that could have produced the observed transition. Return only a JSON array containing one valid action block.

### 参考答案轨迹

参考类型：真实人类演示；该轨迹不是唯一合理答案。

动作块 1：

```text
<|action_start|> ; shift MouseLeft ; shift ; shift ; shift <|action_end|>
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
| 来源 episode | `lovely-persimmon-angora-738809e79cc7-20220416-150446` |
| 图片帧 | `[9339, 9343, 9347, 9351]` |
| 目标动作区间 | `[9351, 9355]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 9339**

![history_to_future_action_000001 frame 9339](images/history_to_future_action_000001_00.jpg)

**图 2，帧 9343**

![history_to_future_action_000001 frame 9343](images/history_to_future_action_000001_01.jpg)

**图 3，帧 9347**

![history_to_future_action_000001 frame 9347](images/history_to_future_action_000001_02.jpg)

**图 4，帧 9351**

![history_to_future_action_000001 frame 9351](images/history_to_future_action_000001_03.jpg)

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
| 来源 episode | `shabby-viridian-beaver-90117de50fa9-20220417-153308` |
| 图片帧 | `[16410, 16414, 16418, 16422]` |
| 目标动作区间 | `[16422, 16426]` |
| 初始训练准入 | `False` |
| 结构审核 | `pass` |

### 图片

**图 1，帧 16410**

![history_to_future_action_000002 frame 16410](images/history_to_future_action_000002_00.jpg)

**图 2，帧 16414**

![history_to_future_action_000002 frame 16414](images/history_to_future_action_000002_01.jpg)

**图 3，帧 16418**

![history_to_future_action_000002 frame 16418](images/history_to_future_action_000002_02.jpg)

**图 4，帧 16422**

![history_to_future_action_000002 frame 16422](images/history_to_future_action_000002_03.jpg)

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
  "id": "history_to_future_action_000002",
  "reviewer": "deterministic_structure_audit_v1",
  "decision": "pass",
  "hard_rejection": false,
  "reasons": []
}
```

