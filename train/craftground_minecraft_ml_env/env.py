"""Minecraft Craftground 环境的标准化包装。

提供：
  1. 标准 gym 接口（reset, step）
  2. 成就追踪（用于 Achievement Distillation）
  3. 观测标准化（uint8 → float32 [0,1]）

对外接口：
    MinecraftCraftgroundEnv — 单环境（兼容 gym）
    MinecraftCraftgroundVecEnv — 向量环境（并行多 env）
"""

from typing import Dict, Tuple

import numpy as np
import torch


class MinecraftCraftgroundEnv:
    """单个 Minecraft Craftground 环境。

    兼容 gym/gymnasium 接口：
      reset() → obs
      step(action) → (obs, reward, done, info)

    额外功能：
      - 自动追踪成就
      - 观测格式转换（uint8 [0,255] → float32 [0,1]）
    """

    def __init__(self, seed: int = 0):
        """初始化 Minecraft 环境。

        Args:
            seed: 随机种子
        """
        self.seed = seed
        # TODO: 初始化 craftground.Env()
        # from craftground import Env
        # self.env = Env(seed=seed)
        self.episode_step = 0
        self.max_episode_steps = 1000

    def reset(self) -> np.ndarray:
        """重置环境。

        Returns:
            obs: (H=64, W=64, C=3) uint8 RGB 图像
        """
        # TODO: return self.env.reset()
        raise NotImplementedError("待 craftground 集成")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步。

        Args:
            action: int [0, 26]

        Returns:
            obs: (H, W, C) uint8
            reward: float（成就触发时 > 0）
            done: bool（episode 结束）
            info: dict with keys:
                - 'achievements': np.array of int (n_achievements,)
                - 'successes': np.array of bool (n_achievements,)
        """
        # TODO: obs, rew, done, info = self.env.step(action)
        raise NotImplementedError("待 craftground 集成")

    def close(self):
        """关闭环境。"""
        # TODO: self.env.close()
        pass


class MinecraftCraftgroundVecEnv:
    """向量化 Minecraft 环境（多 env 并行）。

    用于 PPO+AD 训练。

    Args:
        nproc: 并行环境数
        device: 张量目标设备
    """

    def __init__(self, nproc: int = 1, device: str = "cuda"):
        self.nproc = nproc
        self.device = device
        # TODO: 创建 nproc 个 MinecraftCraftgroundEnv 实例

    def reset(self) -> torch.Tensor:
        """重置所有环境。

        Returns:
            obs: (nproc, C, H, W) float32 [0,1] on device
        """
        # TODO: 实现
        raise NotImplementedError("待 craftground 集成")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """并行步进。

        Args:
            actions: (nproc,) long

        Returns:
            obs: (nproc, C, H, W) float32
            rewards: (nproc, 1) float32
            dones: (nproc, 1) float32
            infos: dict with:
                - 'achievements': (nproc, n_achievements) int32
                - 'successes': (nproc, n_achievements) bool
        """
        # TODO: 实现
        raise NotImplementedError("待 craftground 集成")

    def close(self):
        """关闭所有环境。"""
        pass
