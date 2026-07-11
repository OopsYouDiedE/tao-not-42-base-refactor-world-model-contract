#!/usr/bin/env bash
# 教师打标循环:每 300s 对 pool+holdout 增量打标(vpt_teacher manifest 幂等跳过已标段)。
# L4 实测 666fps,一段 6000 帧 ≈ 9s,与 BC 训练共卡无压力。
# 用法: nohup bash scripts/label_loop.sh > runs/label_loop.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
while true; do
  python -m train.minecraft.vpt_teacher \
    --pool runs/data/vpt_early runs/data/vpt_holdout \
    --out runs/data/vpt_labels --device cuda 2>&1 | grep -v UserWarning
  sleep 300
done
