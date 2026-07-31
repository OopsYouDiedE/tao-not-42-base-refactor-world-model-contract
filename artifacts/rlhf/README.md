# RLHF 过程归档

本目录保存 RLHF 的可审计过程元数据，不保存 4 GB 级 LoRA 权重。权重由 Hugging Face 仓库保存，下面的 JSON 保留输入轨迹、训练超参数、指标和复现关系。

| 流程 | 轨迹 | 结果 | 权重 |
|---|---|---|---|
| 树冠动作 RLHF | `tree-trajectory-rollout.json` | `tree-trajectory-training-result.json` | `unjustify/minestudio-gemma4-26b-a4b-trajectory-rlhf` |
| 四题型审核 rollout | `reviewer-rollout.json` | 尚未执行审核 LoRA 更新 | `unjustify/minestudio-gemma4-26b-a4b-reviewer-rlhf`（待训练） |

## 复现

```bash
python -m train.gemma_vision_rlhf \
  --adapter unjustify/minestudio-gemma4-26b-a4b-trajectory-lora \
  --execution runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/execution.json \
  --output-dir runs/gemma4-26b-a4b-trajectory-rlhf-retry10 \
  --epochs 1 \
  --hf-repo unjustify/minestudio-gemma4-26b-a4b-trajectory-rlhf

python -m train.gemma_vision_review_rlhf rollout \
  --adapter unjustify/minestudio-gemma4-26b-a4b-trajectory-lora \
  --archive runs/datasets/reviewer-rlhf-source/minestudio-trajectory-sft-237.h5 \
  --output-dir runs/gemma4-26b-a4b-reviewer-rollouts-20260731
```

审核 rollout 的瓶颈已经记录在 `reviewer-rollout.json`：模型能够输出合法审核 JSON，但对合成 `unsupported_key` 候选存在系统性错误批准，需在此基础上执行审核能力 RLHF。
