# 交接文档(2026-07-05 14:52,供切换会话/模型后接手)

**任务性质(用户指令,不可降级)**:证明导向——找多项证据支撑愿景;遇完全否定性
证伪 → 换方向重做,**判据仍必须先于看结果预登记**(纪律不变,变的只是败处置)。
慢塔**必须用预训练权重**。需要 sudo / 重下大文件 → 先告知用户。遇难解兼容性问题
→ 已授权直接全新安装环境(甚至换 CUDA 版本)。**不许 git push**(用户亲自推)。

## 1. 证据台账(截至现在)

| 证据 | 判定 | 关键数字 | 出处 |
|---|---|---|---|
| 播种机制(Step1) | ✓ | keys 显著 + 错配归因 | step1 doc,已提交 20ece0b |
| S4a 一档(免费搭车) | ✗ | Δ方位 +3.1pt<5pt,age R² 跨零 | step2 §7 |
| S4b/S4c 一档 | ✗(S4b 显著反向) | M1 0.568<M0 0.588 | step2 §7 |
| S5a/S5b 记忆/前瞻视界 | ✗ | k=10 STATE −0.24 vs FRAME +0.01 | step2 §8,runs/ftt_s5_360.json |
| **S5c 可控性** | **✓** | Δerr(flip−true) CI [+0.0014,+0.0081] | 同上;一步模拟器实证 |
| **S4a 二档(W1b, aux-msg 1.0)** | **✓** | Δ方位 +5.2pt≥5pt;age R² 0.275 CI [0.21,0.34] | step2 §7.1,runs/ftt_w1b/probe_b.json |

结论口径:带时记忆**可形成但需显式目标**;状态是程序性(动作→后果)不是情景性
(场景内容);分界线论(用户 2026-07-05 重构)是所有判读的总纲。

## 2. 在途进程(接手第一件事:逐一确认)

| 进程 | 状态@14:52 | 产出哨兵 | 异常处置 |
|---|---|---|---|
| eval_s5 on W1b(PID 337657,13:52 起) | 跑着,GPU 23% | `runs/ftt_s5b_360.json` | 超 2.5h 查进程;死了就重跑 `PYTHONPATH=. python3 train/fovea_twotower/eval_s5.py --ckpt runs/ftt_w1b/ckpt.pt --out runs/ftt_s5b_360.json` |
| C1/S6 队列(runner 等哨兵,60s×900 轮询) | 等 ftt_s5b_360.json | `runs/ftt_s6_360.json` | runner 若死:手跑 train_cmd.py → eval_cmd.py(参数见 runner 脚本 /home/ame/.claude/jobs/c6d3e768/tmp/ 下) |
| Nemotron 权重 curl 续传(后台任务 bsbyx0qy5) | 3.2/7.9GB,~35min | HF cache snapshot 出现 `model.safetensors` 软链 | curl 带 -C - 自动续传;任务被截断就**原命令重发**(不会从零);期望 sha256=55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42,大小 7947142640;校验+装缓存逻辑已在命令里 |

GPU 纪律:3090 24GB,队列训练时占 ~14GB;**别在 C1 训练期间加载 4B 模型**,
冒烟放队列换挡间隙,或先 CPU 加载验代码路径。

## 3. 待判事项与预登记判据(先判后看,别现编)

- **S5b 对照(ftt_s5b_360.json 出分后)**:这是**探索性对比**,无预登记过/败门——
  正确姿势 = W1b 的 (STATE−FRAME) ΔR² 对比 step2 §8 的 W1 数字,回答
  "aux_msg 是否顺带改变场景记忆视界",写入 step2 §7.1 续段。**不得**事后发明判据。
- **S6(ftt_s6_360.json)**:判据在 step3 doc §2——S6a TRUE−FREE dx方向 AUC Δ CI>0;
  **S6b obedience CI 下界>0.5(核心)**;S6c fire_steer CI>0 且 turn→fire 串扰低。
  败处置(预登记):S6b 败查 cmd 注入路径(钉头 token 是否被注意力读到)再换注入方式。
- **S7a/S7b(Step4)**:判据在 step4 doc §2。S7a = S4a 同款探针打预训练塔
  (STATE = 21 层 Mamba2 ssm_state 池化),过 = 方位 acc≥FRAME+5pt 且 age R² CI>0。
  **注意对照口径**:W4 若纯 MSE(免费搭车)训练即过 S7a → "预训练修复免费搭车",
  是最强主张;若不过 → W4b 加 aux_msg 复跑(与 W1b 同目标,对照"同目标下预训练增益")。
  败处置:W4b → 换 9B-v2-Base → 显式目标路线,**不停摆**。

## 4. 可能出现的问题(按概率排序)+ 应对

1. **Nemotron 加载失败(transformers 4.57.3 vs 卡片测试的 4.48.3)**:
   trust_remote_code 自带建模代码,大概率兼容;若报 Cache/API 类错误
   (`DynamicCache`、`get_seq_length`、`_prepare_cache` 签名变动是常见雷),
   **优先本地补丁 HF cache 里的 modeling_nemotron_h.py**(改动小、不动环境);
   实在不行按授权建独立 venv 装 transformers==4.48.3(torch 重下 ~2.5G,告知用户;
   **别降级主 conda 环境**——夜间队列依赖 fla 0.5.1 + torch 2.9.1+cu128)。
2. **inputs_embeds 支持**:W4 适配器要绕过 tokenizer 直接喂投影后的嵌入。
   接手后先 grep modeling_nemotron_h.py 确认 `NemotronHModel.forward` 接受
   `inputs_embeds`;若不接受,直接调 backbone 层循环(参考其 forward 源码)。
3. **mamba-ssm 上游 bug**:`selective_state_update` 在 `D=None` 时炸
   (运算符优先级,`*(...) if ... else 0` 解析成 `*0`)。Nemotron 传非 None 的 D,
   不触发;**自己写调用时必须传 D**。内核已功能验证(4 算子在 3090 实跑过,
   版本 mamba-ssm 2.3.2.post1 / causal-conv1d 1.6.2.post1,torch/fla 未受损)。
4. **LoRA target 模块名**:peft 配置前先
   `[n for n,_ in model.named_modules()]` 确认真实名字(Mamba2 是 in_proj/out_proj,
   attn 是 q_proj/k_proj/v_proj/o_proj,但以实际为准),别抄设计稿猜名。
5. **显存**:4B bf16 权重 ~8GB + LoRA 激活;bs1×accum4、grad checkpointing、
   seq 64 帧×83 token=5312。若 OOM:先砍 seq 到 32 帧,再考虑 4-bit 基座(qlora)。
6. **eval_s5 慢**(W1 版 20min,W1b 版已 1h+):大概率是同机下载/编译抢 CPU,
   不是死锁;进程在、GPU 有占用就等。
7. **代理断流**:HF 下载曾两次卡死(75min/13min 无进展)。已换 curl
   `--speed-limit 10240 --speed-time 60 --retry` 自愈;新下载沿用此法,别用裸 hf CLI。

## 5. 缺口清单与建议优先级(可行性论证对账,2026-07-05 版)

| 优先 | 缺口 | 一句话 | 成本 |
|---|---|---|---|
| P0 | S4b/S4c 复判(W1b 塔重训 M1/M0/Mscr) | 记忆在状态里≠策略用得上;一档 S4b 显著反向,这环断了全断 | 一夜 |
| P0 | S6 指令服从 | 慢指挥快的最小实证 | 今晚自动出 |
| P1 | S7a/S7b 预训练增益 | "必须预训练"从信念变证据 | W4 代码+一夜 |
| P1 | S7c 语义端到端 | LLM 自己读历史发指令 | 依赖前两项 |
| P2 | 多步前瞻(AR 盲滚动) | S5c 只有一步深 | CausalAttn 加 KV cache,半天+一夜 |
| P2 | 实时性预算 | 4B 单步延迟 vs 帧率 | 模型能加载后 5 分钟 |
| P3 | 在线闭环 | 全部证据都离线;终审是真游戏跑赢 | 最大,放最后 |
| 债 | n=1 无跑间方差;单数据集 | 关键 PASS(S4a二档/S5c)补 2–3 种子 | 每种子一夜 |

诚实外推声明(写结论时必带):在 4B 上论证,只能主张 58M从零→4B预训练 的
增益**方向**,不能主张 27B 终点。

## 6. Git 状态

未提交:tower.py / dataset.py 改动 + step2/3/4 文档 + train_w1/train_w2/train_cmd/
eval_cmd/eval_s4/eval_s5/probe_b + 本文档。未推送:bf8f7cc、20ece0b。
计划:今晚 S6 出分判完后一并 commit(用户推)。分支 main;
另有 net/vpt、train/craftground 下他人/它任务的改动,**别混进 fovea commit**。

## 7. 环境速查

conda base(~/miniconda3),python 3.13.11,torch 2.9.1+cu128(驱动 550 可用,
勿信旧 cu124 记忆),transformers 4.57.3,peft 0.18.1,fla 0.5.1,
mamba-ssm 2.3.2.post1,causal-conv1d 1.6.2.post1(均过 GPU 功能冒烟)。
代理 http://127.0.0.1:2080(HF 必须走);HF_TOKEN 见记忆 hf-token。
数据 runs/data/g500_360p(17.4h);已训 ckpt:runs/ftt_w1(一档)、runs/ftt_w1b(二档)。
