# 模型盲测包

`requests.jsonl` 每行是一道独立视觉题。模型必须读取 `images` 中按顺序排列的全部图片，
按照 `prompt` 作答，并把结果写入 `responses.jsonl`。回答格式与
`responses_template.jsonl` 相同：`id` 原样返回，`answer` 是一个或多个变长动作块组成的数组。

本目录不包含参考答案。不要向做题模型提供上级目录中的 `answer_key.jsonl`、生成报告或审核文件。

完成答题后运行：

```bash
python -m datasets.minestudio_finetune.test_answers   --dataset-dir datasets/minestudio_finetune/luna_eval   --responses datasets/minestudio_finetune/luna_eval/blind/responses.jsonl   --output datasets/minestudio_finetune/luna_eval/luna_results.jsonl
```

报告三个口径：动作协议格式通过率、与人类参考轨迹的相似度分布、视觉语义审核通过率。
由于动作存在多解，最终正确率采用视觉语义审核通过率；参考相似度只作为诊断指标。
