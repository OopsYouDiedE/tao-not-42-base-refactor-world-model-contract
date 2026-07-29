# TaoNot42大语言模型游戏控制器

本项目被设计以Minecraft作为强化学习环境，兼容键鼠和手柄两种操作方式，目标通用型游戏模型。

## 一键安装
本命令行仅用于linux环境，用于训练用配置。注意，这里建议使用Docker或者是云服务器，而不是使用和其他任务混用的服务器。

```bash
#更新系统环境和安装需要的包
sudo apt update
sudo apt install -y curl git ffmpeg libgl1 libglib2.0-0 xvfb 


# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"

#下载项目，cd，安装依赖。
git clone https://github.com/OopsYouDiedE/tao-not-42-base-refactor-world-model-contract.git
cd tao-not-42-base-refactor-world-model-contract
sh tests/system/environment_test.sh
uv venv --python 3.13
uv pip install unsloth lmdb av pillow opencv-python-headless pytest
```
