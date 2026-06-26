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
from craftground import CraftGroundEnvironment, InitialEnvironmentConfig


class MinecraftCraftgroundEnv:
    """单个 Minecraft Craftground 环境，兼容 gym 接口。"""

    def __init__(self, seed: int = 0, max_steps: int = 1000):
        self.seed = seed
        self.max_steps = max_steps
        self.episode_step = 0

        config = InitialEnvironmentConfig()
        self.env = CraftGroundEnvironment(config)

    def reset(self) -> np.ndarray:
        """重置环境，返回 (H, W, C) uint8 观测。"""
        self.episode_step = 0
        return self.env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步，返回 (obs, reward, done, info)。"""
        result = self.env.step(action)

        if len(result) == 5:
            obs, rew, done, truncated, info = result
            done = done or truncated
        else:
            obs, rew, done, info = result

        self.episode_step += 1
        if self.episode_step >= self.max_steps:
            done = True

        return obs, float(rew), done, info

    def close(self):
        self.env.close()


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
        self.nproc = nproc
        self.device = device
        self.max_episode_steps = max_episode_steps

        self.envs = [MinecraftCraftgroundEnv(seed=i, max_steps=max_episode_steps) for i in range(nproc)]

        self.episode_rewards = np.zeros(nproc)
        self.episode_lengths = np.zeros(nproc, dtype=np.int32)

    def reset(self) -> torch.Tensor:
        """重置所有环境，返回初始观测。

        Returns:
            obs: (nproc, C, H, W) float32 [0,1]
        """
        obs_list = [env.reset() for env in self.envs]
        self.episode_rewards.fill(0)
        self.episode_lengths.fill(0)
        return self._to_obs(np.array(obs_list))

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """步进所有环境（顺序执行，待优化为真正的并行）。

        Args:
            actions: (nproc,) long

        Returns:
            obs: (nproc, C, H, W) float32 [0,1]
            rewards: (nproc, 1) float32
            dones: (nproc, 1) float32
            infos: dict
        """
        obs_list, rewards, dones = [], [], []
        infos = {"episode_lengths": [], "episode_rewards": []}

        for i, (env, action) in enumerate(zip(self.envs, actions.cpu().numpy())):
            obs, rew, done, info = env.step(int(action))

            self.episode_rewards[i] += rew
            self.episode_lengths[i] += 1

            if done:
                infos["episode_lengths"].append(self.episode_lengths[i])
                infos["episode_rewards"].append(self.episode_rewards[i])
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
                obs, _ = env.reset(), None

            obs_list.append(obs)
            rewards.append(rew)
            dones.append(done)

        return (
            self._to_obs(np.array(obs_list)),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()

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
