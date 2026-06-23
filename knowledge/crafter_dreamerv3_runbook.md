# Crafter × DreamerV3 训练手册（L4 实测）

> 本文记录平台迁移所需的操作步骤与本轮训练结论。权威来源仍是代码（`train/crafter/`、`net/dreamerv3/`）与 `knowledge/dreamer.md`，本文只补充可复用的运行方法与本轮观测结果。

## 1. 本轮训练结论（small，跑到 75.8k / 200k 步后手动停止）

- **世界模型：正常收敛**。`wm_total 240 → ~35`、`image 237 → ~26`，单调下降后趋于平台，无 NaN（曲线见 `training_curve.png` 右栏）。这说明训练管线运行正常，世界模型有在学习。
- **策略：40k 步到峰后回落**。`ep_rew` 从早期 1.0 升到峰值约 2.4（@40k），随后回落至约 1.8（@75k）；`ach/ep` 趋势相同，峰值约 3.3（@40k），回落至约 2.7。actor 熵从约 2.3 降至约 1.0，策略逐渐收窄，在有限预算内没有突破更难的成就。
- **最佳检查点：`checkpoints/ckpt_00040000.pt`**（不是最新的）。评测、可视化、热启续训均建议使用此检查点。
- **参照尺度**：Crafter 原生奖励 = 每首次解锁成就 +1，加生命值 ±0.1 塑形（`env.py` 未缩放）。随机基线约 2.1；DreamerV3 在完整 1M 步预算下约 10–11（成就 8–9）。本轮 200k 步是完整预算的 1/5，加之 40k 后策略退化，绝对分偏低属于正常。

## 2. 从零开始训练（任意 Linux + CUDA 平台）

```bash
# 0) 进入项目根目录，后续命令均从根目录执行
cd <repo>

# 1) 安装依赖。requirements.txt 不含 crafter，需单独安装（会附带 opensimplex / ruamel.yaml）
pip install -r requirements.txt
pip install crafter

# 2) 检查 GPU（确认 torch 能识别到显卡）
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi

# 3) 冒烟测试（tiny，约 4k 步，几分钟，验证整条链路无 NaN）
python -m train.crafter.train_dreamerv3 --size tiny --total-steps 4000 \
    --prefill 500 --run-dir runs/crafter_smoke
# 看到 wm/img 损失下降、有 upd= 日志、最后打印"训练完成"即通过

# 4) 正式训练（small，后台运行 + 行缓冲日志）
nohup python -m train.crafter.train_dreamerv3 --size small --total-steps 200000 \
    --run-dir runs/crafter_dreamerv3 > runs/crafter_dreamerv3/train.log 2>&1 &

# 5) 观测
tail -f runs/crafter_dreamerv3/train.log              # 指标
watch -n1 nvidia-smi                                  # GPU（利用率有突发，需多次采样看均值）
```

效率开关（`net/` 保持不变，仅在训练入口 `_enable_fast_math()` 中设置）**已默认开启**，无需手动配置：TF32 + `cudnn.benchmark` + 关闭 `torch.distributions` 参数校验（RSSM observe 的纯 CPU 校验开销）。默认使用大 batch 短序列：`batch=48 seq=32 updates_per=2 train_every=1`（train ratio 384）。

## 3. 并行环境数（n_envs）与 GPU / CPU 的关系

**架构说明**：`VecCrafterEnv` 在单进程内顺序步进 n_envs 个 Crafter 实例（为 Colab 兼容，未使用多进程）。Crafter 世界生成是纯 Python，受 GIL 约束，因此环境步进是 CPU 串行的，墙钟时间与 n_envs 成正比；增加 CPU 核数不会自动加速（单进程无法利用多核）。

三个资源各自负责不同部分：

| 资源 | 负责 | 受什么主导 |
|---|---|---|
| **CPU** | 环境步进（串行）+ RSSM `observe` 沿 T 的 Python 逐步调度 | n_envs、seq_len；是实际吞吐上限（overhead-bound） |
| **GPU** | 世界模型前向/反向 + 想象 actor-critic 的张量计算 | batch × seq × updates_per；与 n_envs 无关 |
| 关系纽带 | `train ratio = updates_per × batch × seq /(train_every × n_envs)` | 调整此值以平衡数据供给与墙钟时间 |

**实测（L4，small）**：GPU 利用率突发 30–94%（均值约 70%），显存 5.5/23 GB，GPU 不是瓶颈，瓶颈在于 RSSM 的逐步 Python 循环。

**n_envs 建议**：
- **当前单进程 env（本仓现状）**：`n_envs=8` 是较优选择（配合 train ratio ≈ 384/512）。继续增加 n_envs 会线性增加 env 墙钟，样本复用也会下降，超过 16 基本只会更慢，不建议。
- **若换更强 GPU（A100/H100）**：GPU 更不是瓶颈，优先增大 batch（48→64–96，RSSM overhead 与 batch 几乎无关，边际成本低），或将 `--size` 从 `small` 改为 `default`；n_envs 仍保持 8–16。显存参考：B=64，T=48 峰值约 20GB（24GB 卡的安全上限），B≤64，T≤32 < 14GB。
- **若想利用多核 CPU**：需将 env 改为多进程/子进程 VecEnv（本仓未实现）。改完后 n_envs ≈ 物理核数，env 吞吐可接近线性扩展——这是单机最大的提速方向，但需要先实现并行 env。
- **经验规则**：`n_envs ≈ min(可并行的物理核数, 16)`，**仅当 env 真正并行时成立**；否则固定 8。GPU 空闲（利用率低）时提高 `updates_per`/`batch`；CPU 满载（env 步进是墙钟瓶颈）时考虑并行 env 或降低 n_envs。

## 4. 用最佳检查点做评测 / 可视化（不依赖继续训练）

`ckpt_00040000.pt` 包含 `{"total_steps", "model_state", "ep_rewards"}`；用 `net.dreamerv3.build_dreamerv3(..., **SIZE_PRESETS["small"])` 重建模型后 `load_state_dict(ckpt["model_state"])`，再用 `agent.policy()` 在 `VecCrafterEnv` 上跑 rollout，即可渲染帧或统计成就。

## 5. 续训（到 500k–1M 步才有望接近基准）

本仓训练脚本目前**没有 `--resume` 入口**（只存检查点，不读取）。有两条路：
1. 直接以 `--total-steps 1000000` 从头重训（L4 约需 1 天；建议在更强的卡 + 更大 batch 上运行）。
2. 给 `train_dreamerv3.py` 增加加载检查点的 `--resume` 参数，从 `ckpt_00040000.pt` 热启（需修改代码）。

针对策略退化问题，可同时提高 actor 熵正则系数或调整 `--ac-lr`，以缓解熵过早塌缩。
