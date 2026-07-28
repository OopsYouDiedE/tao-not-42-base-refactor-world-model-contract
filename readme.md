# MineStudio → Lumine 动作预训练 + Unsloth 视觉 SFT

两件事：把 MineStudio 的 Minecraft 轨迹转成 Lumine 格式的动作序列，
再用 Unsloth 在这份数据上微调视觉主干。数据集全量下载到本地，训练时**流式加载**，
不落盘中间产物。

**本次使用的模型：`gemma-4-26B-A4B-it`**（MoE，26B 总参 / 4B 激活），
测试与训练都用它，也是两个入口的默认值。备选主干见「四、训练」。

    bc_datasets/minestudio/   批量下载 MineStudio LMDB 分片
    machine_environment/      环境检测 + 开工前体检（显存 / 存储 / 网络）
    train/                    流式数据加载 + Unsloth 视觉 SFT
    tools/                    动作编码查看器（Gradio），开发期用
    tests/                    纯 CPU 单元测试，不碰真实数据集与权重
    runs/bc_datasets/         下载的 LMDB 分片，Git ignored
    runs/trains/              checkpoint 与训练日志，Git ignored

`bc_datasets/` 是代码命名空间，每个子包对应一个数据来源；`runs/bc_datasets/` 是数据落地处。

## 安装

一条命令建虚拟环境并装齐全部依赖（训练 + 工具 + 测试）：

    python -m venv .venv && . .venv/Scripts/activate && python -m pip install -U pip && python -m pip install -e ".[train,tools,dev]" && python -m machine_environment.preflight

Linux / macOS 把 `. .venv/Scripts/activate` 换成 `source .venv/bin/activate`。
末尾的 `preflight` 会顺手做一次环境体检，装完立刻知道这台机器够不够用。

只跑数据管线的机器不需要 CUDA 栈：

    python -m pip install -e ".[dev]"           # 只要下载与数据读取
    python -m pip install -e ".[train,dev]"     # 加训练侧（CUDA 栈）
    python -m pip install -e ".[tools,dev]"     # 加动作查看器（Gradio）

## 先看机器

开工前必须先体检。三项建议指标：单卡显存 ≥ 24GiB、可用空间 ≥ 100GiB、能连上 HuggingFace。

    python -m machine_environment.preflight                 # 规格 + 体检结论
    python -m machine_environment.preflight --no-network-check
    python -m machine_environment.hardware_report --json    # 只要规格，机器可读

任一项不达标只**警告**，不阻止执行——显存小可以调低 `--micro-batch`，盘小可以少下几个
分片，网络不通只影响下载而不影响已下好的数据。检测不到的项按"未知"处理并同样警告，
未知不等于达标。下载与训练入口会自动跑一遍体检（`--no-preflight` 可跳过），
结论写 stderr，stdout 仍是纯 JSON。

显存取**单卡最大值**而不是多卡合计：LoRA 跑在单卡上，两块 12GB 不等于一块 24GB。

CUDA 有三个互不相同的版本号，报告里分开列——混为一谈是排错时最常见的误判来源：
驱动支持的**上限**（`nvidia-smi`）、已装**工具链**（`nvcc`）、torch **编译时链接**的版本。
后两者不一致通常无害，torch 自带运行时。

## 一、下载 MineStudio 数据

数据来自 `CraftJarvis/minestudio-data-{6xx,7xx,8xx,9xx,10xx}-v110`（OpenAI VPT
contractor data 转成 MineStudio 轨迹结构）。仓库内按模态解耦存放，
布局为 `<模态>/part-<编号>/{data.mdb,lock.mdb}`。

    # 先看有哪些分片，估一下总容量
    python -m bc_datasets.minestudio.huggingface_download --dataset 10xx \
        --modal image action --list-parts

    # 全量下载一个数据集（训练用这个）
    python -m bc_datasets.minestudio.huggingface_download \
        --dataset 10xx --modal image action meta_info \
        --maximum-workers 8 --output-dir runs/bc_datasets

    # 先下 1 个分片试水（单个 image 分片可达 29GB）
    python -m bc_datasets.minestudio.huggingface_download \
        --dataset 10xx --modal image action meta_info \
        --maximum-parts 1 --output-dir runs/bc_datasets

不加 `--maximum-parts` 就是全量。并行参数是 `--maximum-workers`。
盘要留够：一个 image 分片 29GB 起，全量远超体检的 100GiB 下限。

下载到本地的 LMDB 就是训练直接读的数据，**不需要再转一次落盘**。

## 二、划分训练 / 验证集

episode 名形如 `lovely-persimmon-angora-02e496ce4abb-20220421-092639`，
结构是 `<前缀>-<12 位 hex>-<日期>-<时间>`。**前缀是玩家标识**（10xx 全量 19 个），
中间的 hex 名义上是会话 ID，但 10xx 里 `f153ac423f61` 一个值就横跨全部 19 个前缀、
覆盖 442 条 episode——它是退化占位值，不能单独当分组键。

    python -m bc_datasets.minestudio.episode_split \
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

## 三、数据集怎么转成训练格式

盘上是按模态解耦的 LMDB 分片，训练要的是「图像 + 指令 → 动作串」的对话。
这个转换在训练时**逐样本流式完成**（`train/lumine_streaming_dataset.py`），
不生成任何中间文件。五步：

1. **建索引，不读数据。** `TrajectoryReader` 打开 `action` 与 `image` 两个模态，
   取 episode 名的交集（各模态分片边界不同，只能按名字对齐）。此时只读了每个分片的
   `__chunk_infos__` 元数据拿到帧数，一帧像素都没解码。

2. **切样本位。** 每条 episode 按 `stride_frames` 切出起始帧，构成 `(episode, 起始帧)`
   的扁平索引表。起始帧要留足历史回溯空间，且窗口不能越过 episode 末尾。
   这张表是纯整数元组，全量数据也只占几十 MB，常驻内存。

3. **取一个窗口。** 拿到 `(episode, start)` 后，从 `action` 读 4 帧（200ms），
   从 `image` 读当前帧与历史帧。LMDB 有块级 LRU 缓存，顺序扫描时相邻样本大量命中
   同一块，避免重复解码。

4. **编码动作。** `encode_lumine_action` 把窗口内的按键与相机增量转成 Lumine
   run-length 动作串，这就是监督目标。前序窗口同样编码一次，作为 prompt 里的动作历史。

5. **组装对话。** 拼成 messages，图像排在文本指令之前（Unsloth 视觉微调的硬约束），
   assistant 回复只含动作串，于是监督目标就是动作 token 本身。

帧以 `PIL.Image` 对象直接进对话，**从不写 JPEG**。落盘方案要先把全部帧写成文件再训练，
全量数据下这一步的耗时与占盘都远超训练本身，且 JPEG 有损重编码会白丢画质。

### 并行喂数据

解码是 CPU 密集的（H.264 解码 + resize），GPU 会等数据，所以用多 worker 预取。
worker 数默认自动推算，按核心数与可用内存两头取小：

    worker 数 = min(逻辑核心数 - 1, 可用内存 / 1GiB, 16)

留一个核心给主进程做 collate 与 GPU 提交；每个 worker 自己持有一份 LMDB 块缓存，
所以内存是硬约束——内存不够时多开 worker 只会触发换页，比串行更慢。
`--dataloader-workers N` 可以手动指定。

LMDB 环境不能跨 fork 共享，因此每个 worker 在首次取数时惰性打开自己的 reader。

### 划分与落盘（可选）

划分在流式加载时自动完成，口径与 `--holdout-level` / `--validation-ratio` 一致。
要单独查看划分结果，或需要一份可离线检查的产物，仍可跑落盘路径：

    python -m bc_datasets.minestudio.lumine_pretrain_builder \
        --dataset-dir runs/bc_datasets/minestudio-data-10xx-v110 \
        --output-dir runs/bc_datasets/lumine-10xx

产物：`samples_train.jsonl` + `samples_validation.jsonl` + `frames/` + `split.json`
+ `dataset_info.json`。训练时加 `--no-streaming` 才会读它。

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

### 逐帧核对编码

    python -m tools.action_inspector --dataset-dir runs/bc_datasets/minestudio-data-10xx-v110

选一条轨迹拖进度条，对照画面看该窗口的按键与相机增量，以及 1 帧/chunk 与 5 帧/chunk
两种粒度的编码串。只监听 127.0.0.1，界面无鉴权，不要用 `--share` 暴露到公网。

## 四、训练

本次用 `gemma-4-26B-A4B-it`，`--dataset-dir` 直接指向下载好的 LMDB 目录：

    python -m train.gemma_vision_sft \
        --dataset-dir runs/bc_datasets/minestudio-data-10xx-v110 \
        --output-dir runs/trains/sft-gemma

`--model` 不写就是这个默认值。备选主干：

| 入口 | 可选模型 | 说明 |
|---|---|---|
| `train.gemma_vision_sft` | **`gemma-4-26B-A4B-it`**（默认，本次使用） | MoE，26B 总 / 4B 激活；无官方 4bit 变体 |
| | `gemma-4-31B-it` | 稠密 |
| | `gemma-4-E4B-it` / `gemma-4-E2B-it` | 显存不足时的退路 |
| `train.qwen_vision_sft` | `Qwen3.6-35B-A3B`（默认） | MoE，35B 总 / 3B 激活 |
| | `Qwen3.6-27B` | 稠密 |

两族共用同一套数据与训练流程（`train/unsloth_supervised_finetuning.py`），
入口只差候选模型与 chat template。

常用参数：

    --subset validation        用验证集样本训练（默认 train）
    --dataloader-workers 8     手动指定并行加载数，默认自动推算
    --holdout-level episode    改划分粒度（默认 prefix）
    --micro-batch 4            显存不足时调低（默认 8）
    --no-streaming             改读 lumine_pretrain_builder 的落盘产物
    --no-preflight             跳过开工前体检

`--micro-batch` 默认 8：96GB 卡上实测这是吞吐/显存拐点，再往上收益枯竭且易 OOM。
显存低于 24GiB 时体检会警告并建议调低它。MoE 主干（gemma-4-26B-A4B、Qwen3.6-35B-A3B）
不建议 `--load-in-4bit`，走 bf16 LoRA。

stdout 是 JSON 训练统计（含样本数、worker 数与划分口径），体检结论走 stderr，
所以 `... 2>/dev/null | jq` 可以直接用。

## 验收

    python -m pytest
    python -m compileall -q bc_datasets train tests tools machine_environment

单元测试是纯 CPU 的，不碰真实数据集与模型权重。涉及真实 LMDB、CUDA 或模型加载的
验证属于冒烟，需单独说明在哪台机器上跑过。
