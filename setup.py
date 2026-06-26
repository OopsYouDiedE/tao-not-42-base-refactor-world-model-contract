#!/usr/bin/env python3
"""项目安装配置 — 支持模块化可选依赖。

使用方法:
    pip install -e .              # 只装核心依赖
    pip install -e .[crafter]     # 装 Crafter 训练依赖
    pip install -e .[dreamer]     # 装 DreamerV3 依赖（包含 crafter）
    pip install -e .[ppo-ad]      # 装 PPO+AD 依赖（包含 crafter）
    pip install -e .[minecraft]   # 装 Minecraft 相关依赖
    pip install -e .[dev]         # 装开发工具（pytest, black）
    pip install -e .[all]         # 装全部（所有可选）
"""

from setuptools import setup, find_packages

setup(
    name="tao-not-42",
    version="0.1.0",
    description="World model and RL training framework",
    author="OopsYouDiedE",
    packages=find_packages(exclude=["tests", "runs", "*.egg-info"]),
    python_requires=">=3.9",

    # ── 核心依赖（所有用户都需要）──────────────────────────────────────────
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",          # cv2: VPT 数据集解码 + 可视化
        "matplotlib",             # 训练面板 minecraft_viz / train_probe
        "transformers",           # 视觉骨干 DINOv2/v3 + 文本编码 MiniLM
        "wandb",                  # 远程实验记录 (--wandb)
        "requests",               # 数据下载
        "pyyaml",                 # YAML 配置读取 (utils/config_io)
    ],

    # ── 可选依赖分组 ──────────────────────────────────────────────────────
    extras_require={
        # Crafter 环境（PPO+AD 和 DreamerV3 都需要）
        "crafter": [
            "crafter",
            "ray[tune]>=2.0",        # 分布式训练 / 子进程并行（可选，train_ppo_ad --vec subproc）
        ],

        # PPO + Achievement Distillation
        "ppo-ad": [
            "crafter",
            "scikit-optimize",       # 最优传输（POT）匹配
            "pot",                   # Python Optimal Transport
        ],

        # DreamerV3 世界模型
        "dreamer": [
            "crafter",
            "tensorflow>=2.13",      # 仅用于比对 / 基线（可选）
        ],

        # Minecraft 相关（VPT dataset）
        "minecraft": [
            "minerl",                # Minecraft 环境接口
            "pillow",                # 图像处理
        ],

        # RL 基础工具
        "rl": [
            "gymnasium>=0.29",       # 环保 gym 替代品
            "envpool",               # 高速向量环境
        ],

        # 开发工具
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],

        # 全部（包括所有可选）
        "all": [
            # crafter 分组
            "crafter",
            "ray[tune]>=2.0",
            # ppo-ad 分组
            "scikit-optimize",
            "pot",
            # dreamer 分组
            "tensorflow>=2.13",
            # minecraft 分组
            "minerl",
            "pillow",
            # rl 分组
            "gymnasium>=0.29",
            "envpool",
            # dev 分组
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
    },

    entry_points={
        "console_scripts": [
            # 如果需要，可添加 CLI 入口点
        ],
    },
)
