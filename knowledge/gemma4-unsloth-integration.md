2026-07-21 | Gemma4-26B-A4B + Unsloth 换主干集成实测(本项目自跑) | 项目内部实验数据(非外部信源)·本机可复现 | 更新于2026-07-21;失效:换模型/换卡/unsloth 或 transformers 大版本变动后重验
---
**性质说明**:本条目是**项目自跑集成实测**,非外部论文。应项目主人要求,把换主干（Qwen3-VL-8B → `google/gemma-4-26B-A4B-it`，MoE 25.2B总/3.8B激活，Text+Image）过程中踩过的坑与结论记于此，便于复现。

## 实测环境
RTX PRO 6000 Blackwell（cc 12.0，96GB）。栈:`torch 2.10.0+cu128`、`transformers 5.5.0`（原生带 `gemma4` 模块）、`unsloth 2026.7.4`、`unsloth-zoo 2026.7.4`、`peft 0.19.1`、`trl 0.24.0`、`xformers 0.0.35`。

## 结论:Unsloth 微调 Gemma4-26B-A4B 可行 ✅
`FastVisionModel` 加载(16bit)显存 51.6GB,注入 LoRA(r=16→494.4M可训练/26.30B总,1.88%),一步图文 SFT 前向 loss 有限、反向 grad_norm 有限**无 NaN**,峰值 55.9GB。unsloth 自动识别 MoE(`num_experts=128`),在 `experts.gate_up_proj`/`experts.down_proj` 上挂 LoRA。

## 踩过的坑(复现必看)
1. **顶层 `datasets/` 包名冲突(AGENTS §65)**:`import unsloth` 内部 `from datasets import Dataset`,被项目顶层 `datasets/` 包遮蔽 → unsloth_zoo import 崩。用 `python -m` 从项目根跑必现(cwd 进 sys.path[0])。**本轮已把 `datasets/` 重命名为 `data_pipelines/` 根治**。
2. **HF 缓存目录大小写导致重复下载**:`snapshot_download('unsloth/gemma-4-26B-A4B-it')`(大写)存到大写目录;但 unsloth 默认 `use_exact_model_name=False` 会把名字规范化成**小写**去找 → 缺 shard → 重下 47GB 撑爆盘。修复:加 `use_exact_model_name=True` 并直接传本地快照绝对路径。
3. **torch 版本被 unsloth 降级**:装 unsloth 会把 torch 从 2.13(cu130)降到 **2.10.0(cu128)**;cu128 在 Blackwell 上实测正常。副作用:transformers 5.5.0 的 cpp 扩展要 torch≥2.11,会打印 `Skipping import of cpp extensions`,基本功能不受影响。
4. **Gemma4 chat 模板要求 content 是 list**:`Gemma4Processor.apply_chat_template` 里 **system/user 的 `content` 必须是 `[{"type":"text",...}]` 列表,裸字符串会 `TypeError: string indices must be integers`**。system role 本身支持。返回键是 `input_ids/attention_mask/mm_token_type_ids/pixel_values/image_position_ids`(与 Qwen3VL 不同)。
5. **26B-A4B 无官方 `-unsloth-bnb-4bit` 变体**(31B/E2B/E4B 有),只能下 bf16(~49GB)或运行时动态量化。
6. **数值坑已被 unsloth 修**:gemma4 MoE 在 fp16 反向会把 grad_norm 变 NaN,unsloth 用 `gemma4_float32.py` 定向 float32 修好(对齐本项目 §5 fp32 归约不变量)。

## 训练 batch 并行数实测(2026-07-21，训练默认规格 history_frames=4、252x448、5 帧动作目标、seq≈1178)
基线(加载+LoRA r=16)53.6GB;每 +1 micro-batch 约 +2.15GB 激活。带 warmup 消除 triton 首编译后的干净数据:

| micro-batch | 峰值显存 | 每样本耗时 | 相对 bs=1 |
|---|---|---|---|
| 1 | 56.2GB | 1.243s | 1.0× |
| 2 | 57.9GB | 0.828s | 1.5× |
| 4 | 62.3GB | 0.647s | 1.9× |
| **8** | **71.0GB** | **0.541s** | **2.3×** |
| 12 | 79.6GB | 0.536s | 2.3× |
| 16 | 88.3GB | 0.512s | 2.4× |

**敲定:训练 micro-batch = 8**(峰值 71GB,留 ~24GB 给真实窗口的序列长度波动;bs=8 后吞吐收益枯竭,bs=16 贴边易 OOM)。
**重要**:旧代码"图像数随窗口变→无法张量化 batch"的假设**是错的**——图像尺寸固定→patch 数固定(pixel_values `[B,2520,768]`)→完全可批量。已把 `world_model_training.py`(该 MineStudio 离线 SFT 入口已随 2026-07-24 重构移除,以下为当时实测)从 `batch_size=1 + grad_accum=8` 改成真实 micro-batch=8(`--micro-batch` 参数,grad_accum 默认降为 1);collate 返回窗口列表、`supervised_loss` 支持 batch 与 padding_side=left 的逐样本 label mask(动作 token 右对齐,`action_len = full有效长度 - prompt有效长度`)。探针:runs/gemma4_batch_probe.py。

## 真实数据端到端冒烟(2026-07-21，改造后验证)
真实 MineStudio 10xx 子集(1 image 分片 part-476 29GB + 全 action 分片,634863 窗口)+ micro-batch=8 走生产 `_sft_loss`→`supervised_loss`:前向 loss=0.53 有限、反向 530 个 LoRA 参数梯度全有限、grad_norm=4.69、峰值 71.1GB(与合成探针 71.0GB 吻合)。**换主干 + batch 改造端到端可行**。脚本 runs/gemma4_realdata_smoke.py。
坑:单个 image 分片 29GB,与 49GB model 挤 100GB 盘,下载 staging 需先清 `.cache/huggingface/download` 残留;download.py 的并行参数是 `--maximum-workers`(非 `--download-workers`)。

---
来源:本项目本机实测(2026-07-21)。加载/一步 SFT 见 net/gemma4_policy.py 与 train/minecraft/ 入口;候选模型评测背景见 [model-benchmarks-vlm-omni.md]。
