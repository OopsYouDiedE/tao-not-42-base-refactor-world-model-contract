# MineStudio → Lumine 预训练数据 + Unsloth 视觉 SFT

两件事：把 MineStudio 的 Minecraft 轨迹批量转成 Lumine 格式的动作预训练数据，
再用 Unsloth 在这份数据上微调 Gemma 4 或 Qwen3.6 视觉主干。

    minestudio_dataset/    批量下载 MineStudio LMDB + 转 Lumine 格式预训练数据
    train/                 Unsloth 视觉 SFT：Gemma 4 与 Qwen3.6 两个入口
    tests/                 动作编解码与时间布局的单元测试
    runs/bc_datasets/      数据集（原始 LMDB 与转换产物），Git ignored
    runs/trains/           checkpoint 与训练日志，Git ignored

## 安装

    python -m venv .venv && source .venv/bin/activate
    python -m pip install -e ".[dev]"          # 数据管线
    python -m pip install -e ".[train,dev]"    # 加训练侧（CUDA 栈）

数据管线不需要 CUDA，训练侧依赖单列在 `train` extra 里。

## 一、下载 MineStudio 数据

数据来自 `CraftJarvis/minestudio-data-{6xx,7xx,8xx,9xx,10xx}-v110`（OpenAI VPT
contractor data 转成 MineStudio 轨迹结构）。仓库内按模态解耦存放，
布局为 `<模态>/part-<编号>/{data.mdb,lock.mdb}`。

    # 先看有哪些分片，别一上来就全量拉
    python -m minestudio_dataset.huggingface_download --dataset 10xx \
        --modal image action --list-parts

    # 只下 1 个分片试水（单个 image 分片可达 29GB）
    python -m minestudio_dataset.huggingface_download \
        --dataset 10xx --modal image action meta_info \
        --maximum-parts 1 --maximum-workers 8 \
        --output-dir runs/bc_datasets

并行参数是 `--maximum-workers`。盘要留够：一个 image 分片 29GB 起。

## 二、划分训练 / 验证集

episode 名形如 `lovely-persimmon-angora-02e496ce4abb-20220421-092639`，
结构是 `<前缀>-<12 位 hex>-<日期>-<时间>`。**前缀是玩家标识**（10xx 全量 19 个），
中间的 hex 名义上是会话 ID，但 10xx 里 `f153ac423f61` 一个值就横跨全部 19 个前缀、
覆盖 442 条 episode——它是退化占位值，不能单独当分组键。

    python -m minestudio_dataset.episode_split \
        --dataset-dir runs/bc_datasets/minestudio-data-10xx-v110 \
        --holdout-level prefix --validation-ratio 0.1 \
        --output runs/bc_datasets/split-10xx.json

两种粒度，选哪个取决于你要衡量什么：

- `prefix`（默认）：整个玩家留出，同一人的运镜与按键习惯不会同时出现在两边，
  衡量**跨玩家泛化**。10xx 上验证集是 6 个前缀 / 198 episodes / 26.0 h。
- `episode`：按 episode 打散，同一玩家两边都有（10xx 实测验证集摸到 19 个前缀里的 16 个），
  数字更好看但只衡量同分布内插。

前缀级帧数分布极偏（10xx 前 4 个前缀占 66%），随机抽组会让验证集占比在 0.1%–22%
之间乱跳，所以组数 ≤20 时精确枚举全部子集取最接近目标占比的那个——10xx 上精确命中 90.00/10.00。

## 三、转成 Lumine 格式

    python -m minestudio_dataset.lumine_pretrain_builder \
        --dataset-dir runs/bc_datasets/minestudio-data-10xx-v110 \
        --output-dir runs/bc_datasets/lumine-10xx

产物：`samples_train.jsonl` + `samples_validation.jsonl` + `frames/`（观测帧 JPEG）
+ `split.json` + `dataset_info.json`。划分在构建时自动完成，参数与上一步一致。
image 模态还没下载时加 `--no-images`，只出动作文本。

### 动作格式

照搬 Lumine（arXiv 2511.08892）的 run-length 表示，帧率换成 Minecraft 的 20Hz：
感知端 4 帧一个窗口（200ms，对应 Lumine 的 5Hz），窗口内每帧一个电机 chunk（50ms）。
每个 chunk 只列出**该 chunk 按住的键**——键在相邻 chunk 连续出现即保持按下、不重按，
某 chunk 缺席即在该 chunk 松开。格式里没有时长数值，也没有按下/松开事件 token，
以此规避“按 3 秒还是 2 秒”这类脆弱的时长回归。

    <|action_start|>ΔX ΔY ΔZ ; W space ; W space ; W space ; W<|action_end|>

`ΔX ΔY` 是窗口内累计的鼠标像素增量（相机度数 ÷ 0.15 度/像素，钳到 ±999），
`ΔZ` 是滚轮档位——VPT 动作空间没有滚轮，快捷栏走数字键，所以恒为 0。

上面这串的语义：W 全程按住只按一次，space 在最后 50ms 松开。

Lumine 原文的 6×33ms 是《原神》的 30Hz 电机步，这里按 Minecraft tick 换算成 4×50ms，
没有照抄。`--frames-per-chunk 2` 可换成 100ms 电机步（此时一个 chunk 内任一帧按下即
记为按住，短按不丢）。

## 四、训练

    python -m train.gemma_vision_sft --model gemma-4-26B-A4B-it \
        --dataset-dir runs/bc_datasets/lumine-10xx --output-dir runs/trains/sft-gemma

    python -m train.qwen_vision_sft --model Qwen3.6-35B-A3B \
        --dataset-dir runs/bc_datasets/lumine-10xx --output-dir runs/trains/sft-qwen

两族共用同一套数据与训练流程（`train/unsloth_supervised_finetuning.py`），
入口只差候选模型与 chat template。`--micro-batch` 默认 8：96GB 卡上实测这是
吞吐/显存拐点，再往上收益枯竭且易 OOM。MoE 主干（gemma-4-26B-A4B、Qwen3.6-35B-A3B）
不建议 `--load-in-4bit`，走 bf16 LoRA。

## 验收

    python -m pytest
    python -m compileall -q minestudio_dataset train tests
