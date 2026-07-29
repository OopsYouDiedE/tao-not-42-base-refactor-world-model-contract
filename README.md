# TaoNot42 大语言模型游戏控制器

本项目以 Minecraft 轨迹训练视觉游戏控制模型。

## 代码结构

| 目录 | 职责 | 是否保存生成文件 |
|---|---|---|
| `datasets/` | 动作协议、数据划分和训练数据构建 | 否 |
| `datasets/minestudio_data/` | `CraftJarvis/minestudio-data` 数据下载与 LMDB 加载 | 否 |
| `datasets/minestudio_finetune/` | 生成并双审三类轨迹题，打包 HDF5 后接入视觉 LoRA 训练 | 否 |
| `train/` | 把数据转换为模型输入并运行视觉 SFT | 否 |
| `tools/` | 人工检查动作和图片的 Gradio 工具 | 否 |
| `tests/` | 与源码同名的自动化测试 | 否 |
| `docs/` | 稳定的数据契约文档 | 否 |
| `runs/` | 数据集、验证结果、图片、Benchmark、缓存、日志和模型权重 | 是，Git 忽略整个目录 |

### MineStudio

`datasets/minestudio_data/` 只包含两个文件：

| 文件 | 职责 |
|---|---|
| `download.py` | 从 Hugging Face 下载 MineStudio 各模态 |
| `load.py` | 读取 LMDB 分片，并按 episode 和帧窗口返回图片、动作及元数据 |

### 数据集代码

| 文件 | 职责 |
|---|---|
| `datasets/action_codec.py` | 动作 token 编码与解码 |
| `datasets/episode_split.py` | 按玩家或 episode 划分训练集和验证集 |
| `datasets/pretraining_dataset.py` | 从 `minestudio-data` 构建落盘训练集 |
| `datasets/variable_action_contract.py` | 变长动作段与动作—图片逐帧对齐校验 |

`datasets/minestudio_finetune/README.md` 详细说明三类轨迹题的生成、人工与 AI 双审、HDF5 打包和训练加载，
训练准入规则和模型做题测试方法。

### 训练代码

| 文件 | 职责 |
|---|---|
| `train/lumine_conversation_dataset.py` | 把落盘 JSONL 转为视觉模型对话格式 |
| `train/lumine_streaming_dataset.py` | 训练时直接流式读取 LMDB，不生成中间数据集 |
| `train/unsloth_vision_sft.py` | Gemma 与 Qwen 共用的 LoRA/SFT 训练流程 |
| `train/command_line.py` | 两个模型入口共用的命令行参数 |
| `train/gemma_vision_sft.py` | Gemma 训练入口 |
| `train/qwen_vision_sft.py` | Qwen 训练入口 |

### 工具与测试

`tools/action_inspector.py` 是人工检查工具。它读取 `runs/` 中的数据集，显示图片和对应动作，
不参与训练。

`tests/` 采用 Python 社区通用命名。目录已铺平，不再设置额外的 `unit/` 层；每个
`test_*.py` 文件对应一个正式源码模块。

## 安装

以下命令用于 Linux 训练环境。建议使用 Docker 或独立云服务器。

```bash
# 更新系统环境和安装依赖
sudo apt update
sudo apt install -y curl git ffmpeg libgl1 libglib2.0-0 xvfb 


# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"

# 下载项目并安装 Python 依赖
git clone https://github.com/OopsYouDiedE/tao-not-42-base-refactor-world-model-contract.git
cd tao-not-42-base-refactor-world-model-contract
uv venv --python 3.13
uv pip install unsloth lmdb av pillow opencv-python-headless pytest
```
