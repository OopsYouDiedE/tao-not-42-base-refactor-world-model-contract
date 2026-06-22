# Crafter × DreamerV3 训练交接 / 运行手册（L4 实测）

> 本文是平台迁移用的操作手册 + 本轮训练总结。SSOT 仍是代码（`train/crafter/`、`net/dreamerv3/`）
> 与 `knowledge/dreamer.md`，本文只记可复用的运行方法与本轮结论。

## 1. 本轮训练总结（small，跑到 75.8k / 200k 步后手动停）

- **世界模型：收敛良好**。`wm_total 240 → ~35`、`image 237 → ~26`,单调下降后平台,无 NaN（曲线 `training_curve.png` 右栏）。这是管线跑通 + 世界模型在学的明确证据。
- **策略：40k 步达峰后回落**。`ep_rew` 1.0(早期)→ **峰值 ~2.4 @ 40k** → 回落 ~1.8 @ 75k;`ach/ep` 同形,峰值 ~3.3 @ 40k → ~2.7。随策略熵从 ~2.3 降到 ~1.0,actor 收敛到偏窄的行为模式(熵塌缩),小预算下没能突破更难的成就。
- **最佳检查点：`checkpoints/ckpt_00040000.pt`**(不是最新的)。做评测/可视化/续训热启都用它。
- **参照尺度**:Crafter 原生 reward = 每首次解锁成就 +1 + 生命值 ±0.1 塑形(env.py 未缩放)。随机基线 ~2.1;DreamerV3 完整 **1M 步**预算下约 10–11(成就 8–9)。本轮 200k 是 1/5 预算,且 40k 后退化,故绝对分偏低属预期。

## 2. 从零到训练(任意 Linux + CUDA 平台)

```bash
# 0) 取项目(repo 根目录,下面命令都从根目录跑)
cd <repo>

# 1) 依赖。requirements.txt 未含 crafter,需单独装(会带 opensimplex / ruamel.yaml)
pip install -r requirements.txt
pip install crafter

# 2) GPU 自检(确认 torch 看得到卡)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi

# 3) 冒烟(tiny,~4k 步,几分钟,验证全链路 + 无 NaN)
python -m train.crafter.train_dreamerv3 --size tiny --total-steps 4000 \
    --prefill 500 --run-dir runs/crafter_smoke
# 看到 wm/img 损失下降、有 upd= 日志、最后打印"训练完成"即通过

# 4) 正式训练(small,后台 + 行缓冲日志)
nohup python -m train.crafter.train_dreamerv3 --size small --total-steps 200000 \
    --run-dir runs/crafter_dreamerv3 > runs/crafter_dreamerv3/train.log 2>&1 &

# 5) 观测
tail -f runs/crafter_dreamerv3/train.log              # 指标
watch -n1 nvidia-smi                                  # GPU(突发利用,需多次采样看均值)
```

效率开关(`net/` 保持纯净,只在训练入口 `_enable_fast_math()` 设)**已默认开启**,无需手动:
TF32 + `cudnn.benchmark` + 关闭 `torch.distributions` 参数校验(RSSM observe 的纯 CPU 校验开销)。
默认大 batch 短 seq:`batch=48 seq=32 updates_per=2 train_every=1`(train ratio 384)。

## 3. 并行环境数(n_envs)与 GPU / CPU 的关系 —— 调参核心

**架构事实**:`VecCrafterEnv` 在**单进程内顺序**步进 n_envs 个 Crafter 实例(为 Colab 兼容,不用多进程)。
Crafter 世界生成是纯 Python,受 GIL 约束 ⇒ **env 步进是 CPU 串行的,墙钟 ∝ n_envs**;加 CPU 核数**不会**自动加速(单进程跑不满多核)。

三者各管一段:

| 资源 | 负责 | 受什么主导 |
|---|---|---|
| **CPU** | 环境步进(串行)+ RSSM `observe` 沿 T 的 Python 逐步调度 | n_envs、seq_len;**真正的吞吐天花板**(overhead-bound) |
| **GPU** | 世界模型前向/反向 + 想象 actor-critic 的张量计算 | batch × seq × updates_per;**与 n_envs 无关** |
| 关系纽带 | `train ratio = updates_per × batch × seq /(train_every × n_envs)` | 调它平衡"喂数据" vs "墙钟" |

**实测(L4,small)**:GPU 突发利用 30–94%(均值 ~70%),显存 5.5/23 GB ⇒ GPU 不是瓶颈,瓶颈是 RSSM 的逐步 Python 循环。

**推荐 n_envs**:
- **当前单进程 env(本仓现状)**:`n_envs=8` 是甜点(配 train ratio≈384/512)。再加 n_envs 会**线性增加 env 墙钟**而样本复用下降,>16 基本只是更慢,不推荐。
- **若换更强 GPU(A100/H100)**:GPU 更不是瓶颈 ⇒ 优先**加大 batch(48→64–96,RSSM overhead 与 batch 几乎无关 ⇒ 近乎免费多喂数据)**,或 `--size small→default`;n_envs 仍保持 8–16。显存参考:B=64,T=48 峰值 ~20GB(24GB 卡安全上限),B≤64,T≤32 < 14GB。
- **若想真正吃满多核 CPU**:需要给 env 改成**多进程/子进程 VecEnv**(本仓未实现)。改完后 n_envs ≈ 物理核数,env 吞吐可近似 ×核数 —— 这是单机最大的提速点,但要先写并行 env。
- **一句话经验法则**:`n_envs ≈ min(可并行的物理核数, 16)`,**当且仅当 env 真并行**;否则固定 8。GPU 空(利用率低)就抬 `updates_per`/`batch`;CPU 满(env 步进是墙)就并行 env 或降 n_envs。

## 4. 用最佳检查点做评测 / 可视化(不依赖继续训练)

`ckpt_00040000.pt` 含 `{"total_steps", "model_state", "ep_rewards"}`;
`net.dreamerv3.build_dreamerv3(..., **SIZE_PRESETS["small"])` 重建后 `load_state_dict(ckpt["model_state"])`,
再用 `agent.policy()` 在 `VecCrafterEnv` 上跑 rollout 即可渲染帧 / 统计成就。

## 5. 续训(到 500k–1M 才有望逼近基准)

本仓训练脚本当前**没有 `--resume` 入口**(只存不读检查点)。两条路:
1. 直接 `--total-steps 1000000` 从头重训(L4 约 1 天;建议在更强卡 + 更大 batch 上跑)。
2. 给 `train_dreamerv3.py` 加载检查点的 `--resume` 旋钮后热启 `ckpt_00040000.pt`(需改代码)。
对策略退化,可一并提高 actor 熵正则 / 调 `--ac-lr`,缓解熵塌缩。
