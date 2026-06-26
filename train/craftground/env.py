"""Craftground 向量化环境（PPO+AD 训练用）。

Craftground 是基于 Minecraft Java 版的 RL 环境。
本模块提供：
  1. 单环境到向量环境的转换（多 env 并行）
  2. 成就追踪（Achievement Distillation 需要）
  3. 观测标准化

对外接口：
    CraftgroundVecEnv — 向量化环境，返回观测 + 成就向量
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch


class CraftgroundVecEnv:
    """Craftground 向量化环境（子进程并行，兼容 AD 成就追踪）。

    Craftground 暴露 Minecraft Java 版的 RGB 观测（图像）和离散动作空间。
    本类提供：
    - 多环境并行（通过 multiprocessing）
    - 成就追踪（从 info["achievements"] 提取）
    - 观测格式转换（uint8 → float32）

    Args:
        nproc: 并行环境数
        device: 张量目标设备（'cuda' 或 'cpu'）
        max_episode_steps: 单个 episode 最大步数（超时自动 done）
    """

    def __init__(
        self,
        nproc: int = 1,
        device: str = "cuda",
        max_episode_steps: int = 1000,
    ):
        # 注：实际实现需要根据 craftground 的真实 API 调整
        # 这里先是框架代码，待 craftground 安装后填充具体细节
        self.nproc = nproc
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.step_count = [0] * nproc

        # TODO: 初始化 craftground 环境
        # from craftground.env import CraftgroundEnv
        # self.envs = [CraftgroundEnv(...) for _ in range(nproc)]

    def reset(self) -> torch.Tensor:
        """重置所有环境，返回初始观测。

        Returns:
            obs: (nproc, C, H, W) float32 [0,1]（观测）
        """
        # TODO: 实现重置逻辑
        raise NotImplementedError("待 craftground 完全集成")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """步进所有环境。

        Args:
            actions: (nproc,) long — 动作 ID

        Returns:
            obs: (nproc, C, H, W) float32 [0,1]
            rewards: (nproc, 1) float32
            dones: (nproc, 1) float32
            infos: dict with keys:
                - 'achievements': (nproc, n_achievements) int32
                - 'successes': (nproc, n_achievements) int32（achievements > 0）
                - 'episode_lengths': (nproc,) int32（仅 done 时有效）
                - 'episode_rewards': (nproc,) float32（仅 done 时有效）
        """
        # TODO: 实现 step 逻辑
        raise NotImplementedError("待 craftground 完全集成")

    def close(self):
        """关闭所有环境。"""
        # TODO: 实现关闭逻辑
        pass

    @staticmethod
    def _to_obs(raw: np.ndarray) -> torch.Tensor:
        """[N,H,W,C] uint8 → [N,C,H,W] float32 [0,1]。"""
        if raw.ndim == 3:  # 单张图像
            raw = raw[np.newaxis, ...]
        # Craftground 观测格式：(H, W, 3) RGB uint8
        return torch.from_numpy(raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)


# 占位符：成就列表（待从 craftground 的实际成就系统提取）
ACHIEVEMENTS = [
    # Minecraft 标准成就（示例）
    # "minecraft.story.root",
    # "minecraft.story.mine_wood",
    # ... (实际列表待确认)
]
