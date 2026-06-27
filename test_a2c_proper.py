#!/usr/bin/env python3
"""使用项目现有的 CraftgroundVecEnv 来训练 A2C 模型"""

import sys
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import NatureCNN
import gymnasium as gym
import numpy as np

# 导入项目的环境
from train.craftground.env import CraftgroundVecEnv

print("=" * 70)
print("Craftground A2C 训练测试")
print("=" * 70)

try:
    print("\n[1/3] 初始化 Craftground 向量化环境...")
    vec_env = CraftgroundVecEnv(
        nproc=1,           # 单个环境
        device="cuda",
        max_episode_steps=100,  # 短 episode 用于测试
        base_port=8023
    )
    print("  ✓ 环境已创建")

    print("\n[2/3] 重置环境...")
    obs = vec_env.reset()
    print(f"  ✓ 初始观测形状: {obs.shape}")
    print(f"  ✓ 观测数据类型: {obs.dtype}")

    print("\n[3/3] 运行 5 步采样...")
    for step in range(5):
        action = torch.randint(0, 27, (1,))  # Minecraft 有 27 个动作
        obs, reward, done, info = vec_env.step(action)
        print(f"  Step {step+1}: reward={reward[0,0]:.2f}, done={done[0,0]}")

    print("\n" + "=" * 70)
    print("✓ SUCCESS! 环境工作正常")
    print("=" * 70)
    print("\n环境已验证，可以用于训练")
    vec_env.close()

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
