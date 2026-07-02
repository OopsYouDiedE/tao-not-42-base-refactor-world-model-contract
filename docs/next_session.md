# 下一次 Colab 会话交接单(2026-07-02 收尾时写)

> 给下一个会话的助手/自己:当前状态、待办与一键命令。主线目标不变:
> 世界模型 → 动作先验 → CraftGround 在线 mine_stone(见 MEMORY 与 knowledge/)。

## 环境自举(GPU 机,L4)

```bash
git clone https://github.com/OopsYouDiedE/tao-not-42-base-refactor-world-model-contract.git /content/repo && cd /content/repo
python install_env.py --dreamer --ppo-ad --dev   # 跳过 minerl(不兼容,见 activity_log)
# ⚠ apt 交互卡死风险:DEBIAN_FRONTEND=noninteractive(install_env 待修)
hf auth login                                     # 需 write token(旧 token 建议已吊销换新)
```

## 状态快照(截至上会话结束)

- **128² findcave 基线(W)已定论**:同帧预算三口径全面胜 64²,EV+0.229/开环+0.74dB,
  结论 knowledge/conclusion_minecraft_dreamer4_run.md §7;checkpoint 未保留(VM 亡)。
- **hard_weight=1.0 负结果**(§8):OHEM 破坏早期流匹配收敛,重试须 ≤0.3+warmup。
- **g500 数据变量实验中止于 8.2k/10k**(§9 补):初步信号为三口径全面差于 findcave,
  下轮建议**数据混合而非替换**。
- **HDF5 归档管线代码就绪**(tests/encode_gaming500_hdf5.py):并行 5、5GB 分片、
  167 游戏轮转交错、多机 --shard-prefix;HF 仓库 unjustify/gaming500-360p-hdf5
  已建但**尚无分片**(首片未攒满即收尾)。
- IG 恒负遗留;解码器平铺 Linear(604M)换卷积头的优先级已上移;
  效率账见 knowledge/analysis_efficiency_levers.md(4.3% MFU ⇒ 架构侧优化优先)。

## 待办(按优先级)

1. **重启 HDF5 编码上传**(GPU 机跑前半游戏,CPU 高内存机跑后半,见下);
2. 数据混合实验:findcave+gaming500_mc 合池,128² 基线配方,对照 §7/§9;
3. 解码器卷积头改造(省 ~7GB 显存 + 质量),之后 batch 可翻倍;
4. IG 恒负专项诊断(v1 学习式挑帧的前置);
5. C 判别头原型(真/生成二分类,训练副产品级代价)。

## 一键命令

GPU 机编码(前半游戏,与 CPU 机分工):
```bash
GAMES=$(python -c "
import requests
t = requests.get('https://huggingface.co/api/datasets/markov-ai/gaming-500-hours/tree/main', timeout=60).json()
gs = sorted(x['path'] for x in t if x['type']=='directory')
print(','.join(g for g in gs if g < 'g'))")
PYTHONPATH=. nohup python tests/encode_gaming500_hdf5.py \
  --games "$GAMES" --n 999 --parallel 3 --shard-gb 5 --shard-prefix gpu_ \
  --out runs/data/g500_h5 > runs/logs/g500_h5.log 2>&1 &
```

CPU 高内存机编码(后半游戏,无训练争核可 --parallel 6):
```bash
# 同上,把 g < 'g' 改为 g >= 'g',--shard-prefix cpu_,--parallel 6
```

数据混合训练(待办 2,GPU 机):
```bash
mkdir -p runs/data/mixed && cd runs/data/mixed && ln -s ../vpt_findcave/* ../gaming500_mc/* . ; cd /content/repo
nohup python -m train.minecraft.train_dreamer4 \
  --data_dir runs/data/mixed --camera_scale 32 --img_size 128 \
  --seq_len 16 --batch_size 16 --token_dim 384 --dyn_layers 8 --enc_base 48 \
  --motion_sample 4 --amp bf16 --workers 6 --clip_cache 4 --clip_max_frames 9000 \
  --total_steps 10000 --eval_interval 500 --seed 42 \
  --run_dir runs/mc_d4_b128_mix > runs/logs/mc_d4_b128_mix.log 2>&1 &
```
(数据下载:`PYTHONPATH=. python tests/download_vpt_data.py --index find-cave-Jul-28 --n 32
--out runs/data/vpt_findcave`;gaming500_mc 用 tests/convert_gaming500.py --n 30
--match "surviv|single|tutorial|explor" --crop-stream --purge-raw。)

## 惯例

- 每 10 分钟 cron:有更改按 AGENTS.md 中文 commit + push;结论入 knowledge/,
  过程入 docs/activity_log.md;runs/ 不入库。
- 长跑必设 --clip_max_frames 9000(超长段 OOM 教训)与 RAM 水位监控;
  pkill 模式串会自匹配包装 shell,用拆串技巧(P="encode_""gaming500")。

## 训练监控结论(2026-07-02 实战沉淀,详见 memory: training-run-ops-lessons)

1. 帧率是第一健康指标:~330 帧/s 正常(GPU-bound);66-200 数据饥饿;无 step 行+GPU 低
   多半是缓存预热(gaming500 段大,首批 3-6 分钟),别误杀。
2. 启动前算内存账:workers×clip_cache×段帧数×49KB(128²)+ ckpt 保存 CPU 尖峰 ~8GB。
3. 止损判据:连续 3 次评估全口径大幅落后且无收敛迹象才杀,单次评估差不算数。
4. LR 无退火 ⇒ 中途收割零损失;同帧预算点是最干净收割点,看门狗盯日志自动杀。
5. 评估里程碑随出随入库(checkpoint 是易失品,活下来的只有 git 里的数字)。
