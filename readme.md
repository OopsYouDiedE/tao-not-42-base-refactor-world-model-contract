# TaoNot42大语言模型游戏控制器

本项目被设计以Minecraft作为强化学习环境，被设计为兼容键鼠和手柄两种操作方式，用来设计通用型游戏模型。

## 一键安装
本命令行仅用于linux环境，用于训练用配置。
```bash
sudo apt update
sudo apt install -y curl git ffmpeg libgl1 libglib2.0-0

curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
git clone https://github.com/OopsYouDiedE/tao-not-42-base-refactor-world-model-contract.git
cd tao-not-42-base-refactor-world-model-contract

uv venv --python 3.13
uv add unsloth lmdb av pillow opencv-python-headless pytest
sh sys_env_test.sh
```
