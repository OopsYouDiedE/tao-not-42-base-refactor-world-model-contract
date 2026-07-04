---
name: fovea-twotower-step1-handoff
description: Step1 交接书(Colab→本机3090):环境、数据编码、R1/GateA/R2 执行命令、通过判据、已冒烟验证的状态
metadata:
  type: project
---

# Step 1 交接书:本机 3090 执行手册

**背景**(3 句话):我们在验证凹视双塔架构([[fovea-twotower-design]])的地基——
"策略只靠读一个**冻结**世界模型的 GDN(Mamba 族)状态获得全部历史"是否成立,
实验设计见 [[fovea-twotower-step1]]。代码已在 Colab L4 上全链路冒烟通过,
本机只需:编码数据 → 依序跑 3 条命令 → 对照判据读数。

## 0. 环境

```bash
pip install flash-linear-attention h5py opencv-python scikit-learn
# torch≥2.5+cu 已装的前提下;fla 纯 triton 免编译。
# DINOv2-S 首跑自动经 torch.hub 下载(~85MB,需外网);HF 数据集公开,无需 token。
```

已验证组合:torch 2.11+cu128 / flash-linear-attention(TileLang 内核,首次调用编译 ~1min)。

## 1. 数据编码(~1-1.5h,产出 ~4-5GB)

```bash
PYTHONPATH=. python3 tests/encode_gaming500_hdf5.py \
  --games valorant --n 45 --hz 10 --scale-h 90 --quality 85 \
  --no-upload --parallel 8 \
  --out runs/data/g500_160p --raw runs/data/gaming500_raw
```

- 断点续跑:进度在 `--out/manifest.json`,重启自动跳过完成会话;
- **编够 ~6 个会话(≈2.5h 数据)即可先启 R1**,后续分片落地后重启训练即可吸收;
- 磁盘峰值 ~40GB(原片滚动删);`--parallel` 按带宽/核数调。
- Colab 上已编好 6 会话半成品(2.47h,shard_0000.h5 435MB)可直接拷来续跑或冒烟。

## 2. 三条命令与判据

### R1 世界塔(一晚)

```bash
PYTHONPATH=. python3 train/fovea_twotower/train_r1.py \
  --data runs/data/g500_160p --out runs/ftt_r1 --steps 12000 --bs 8 --seq 64
```

期望:eval_loss 稳定下降;ckpt 每 1000 步覆写 `runs/ftt_r1/ckpt.pt`。
3090 上若 VRAM 富余可 `--bs 16`(步数减半到 6000)。

### Gate A 探针(半天内,先决判据)

```bash
PYTHONPATH=. python3 train/fovea_twotower/probe_a.py \
  --data runs/data/g500_160p --ckpt runs/ftt_r1/ckpt.pt --n 2000
```

- **PASS**:`auc_attack_FULL ≥ auc_attack_FRAME + 0.05`(脚本自动打 verdict);
- FAIL 处置(step1 §5):世界目标没把控制信息压进状态——R1 加辅助目标重试一次;
  重试仍败 → S2 判负,不必跑 R2,回 Colab 会话议降级方案。

### R2/B1 对比(各 4-6h,可串行)

```bash
PYTHONPATH=. python3 train/fovea_twotower/train_r2.py \
  --data runs/data/g500_160p --ctx runs/ftt_r1/ckpt.pt \
  --out runs/ftt_r2_seed --seed 1 --steps 6000
PYTHONPATH=. python3 train/fovea_twotower/train_r2.py \
  --data runs/data/g500_160p --ctx runs/ftt_r1/ckpt.pt \
  --out runs/ftt_r2_zero --seed 0 --steps 6000
```

对比两组最终 EVAL 行(log.jsonl 尾部):`f1_attack / f1_keys / r2_mouse`。
- **S3a PASS**:seed 组主指标相对 zero 组 ≥ +10%;
- FAIL → 播种通道无效,退回"共享近窗 KV"降级方案(见 step1 §5),双塔仍在。

## 3. 代码事实(Colab 冒烟已验)

| 事实 | 值 |
|---|---|
| 塔参数量 | 各 58.6M(GDN 内部扩展所致,大于纸面 15-25M 估;容量只多不少) |
| 流形态 | 每帧 81 DINO token(126×126 方形,纵横比压扁,全数据一致)+ 1 动作 token |
| 状态 | 9 个 GDN 层 × {recurrent_state [B,6,512,256] + conv_state ×3} |
| 播种实现 | fla `Cache.update(recurrent_state=, conv_state=, layer_idx=, offset=0)` |
| 因果性 | Action 塔保持因果(nemotron 笔记 Table 2:双向化 Mamba 无收益) |
| 精度 | 纯 bf16,无 GradScaler;梯度裁剪 1.0 |

## 4. 坑与注意

- 三个脚本 `holdout_frac=0.1` 必须一致(按段名 md5 决定性切分,改了就时间泄漏);
- 编码中断的 0 帧段是正常残留,加载器自动跳过;半成品分片可读(h5 追加式写);
- fla 首次前向 TileLang 编译 ~1min,不是卡死;
- probe/r2 的 `--seq 64` 与 R1 训练一致(状态分布对齐);
- 全部脚本日志双写 stdout + `<out>/log.jsonl`,判据从 jsonl 尾部读。

## 5. 完成后

把三个 verdict(GateA JSON、两组 R2 final EVAL)连同 log.jsonl 提交入库
(建议 `knowledge/` 下按现有 R2a 探针 JSON 的习语),推送后在任意会话喊
Claude 读 [[fovea-twotower-step1]] §5 对照处置表定下一步。
