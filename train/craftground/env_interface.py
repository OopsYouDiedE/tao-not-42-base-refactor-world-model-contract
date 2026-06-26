"""Craftground 环境接口 - 支持地形检测和种子控制。

功能：
  1. 固定种子启动（用于验证）
  2. 地形检测（避免不利地形）
  3. 成就信息聚合
  4. 清晰的观测/奖励/成就接口
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from train.craftground.env import MinecraftCraftgroundEnv, CraftgroundVecEnv
from train.craftground_minecraft_ml_env.achievements import ALL_ACHIEVEMENTS


class CraftgroundEnvWithTerrainCheck(MinecraftCraftgroundEnv):
    """添加地形检测的单环境。

    初始化时自动检测地形，如果不利就重新启动。
    """

    def __init__(self, seed: int = 0, max_steps: int = 1000):
        super().__init__(seed=seed, max_steps=max_steps)
        self.favorable_biomes = {"forest", "plains", "taiga", "birch_forest"}
        self.unfavorable_biomes = {"ocean", "deep_ocean", "stone_shore"}

    def reset(self) -> np.ndarray:
        """重置环境，如果地形不利则重新启动。"""
        obs = super().reset()

        # 检查地形（通过观测的视觉特征推断）
        retries = 0
        max_retries = 5

        while not self._is_favorable_terrain(obs) and retries < max_retries:
            print(f"  ⚠️  地形不利（重试 {retries + 1}/{max_retries}），重新启动...")
            obs = super().reset()
            retries += 1

        if retries > 0:
            print(f"  ✅ 找到合适地形（第 {retries + 1} 次尝试）")

        return obs

    def _is_favorable_terrain(self, obs: np.ndarray) -> bool:
        """通过观测推断地形是否合适。

        简单启发式：
          - 绿色像素多 → 草原/森林（好）
          - 蓝色像素多 → 海洋（坏）
          - 灰色像素多 → 石头区（坏）

        Args:
            obs: (H, W, 3) uint8 RGB 观测

        Returns:
            是否地形合适
        """
        if obs.dtype == np.uint8:
            obs_norm = obs.astype(np.float32) / 255.0
        else:
            obs_norm = obs

        # 计算颜色直方图
        r = obs_norm[..., 0].mean()
        g = obs_norm[..., 1].mean()
        b = obs_norm[..., 2].mean()

        # 启发式判断
        # 好地形：绿色成分高（草、树）
        # 坏地形：蓝色成分高（水）或灰色成分高（石头）
        green_score = g - r - b  # 绿色优势
        blue_score = b - g  # 蓝色优势

        is_good = (green_score > 0.05) and (blue_score < 0.2)

        return is_good


class CraftgroundVecEnvWithInterface(CraftgroundVecEnv):
    """向量化环境 + 清晰的接口。

    关键接口：
      - reset() → obs
      - step(actions) → obs, rewards, dones, achievements
      - get_achievement_vector() → (B, n_achievements) 成就向量
    """

    def __init__(
        self,
        nproc: int = 16,
        device: str = "cuda",
        max_episode_steps: int = 1000,
        use_terrain_check: bool = True,
        seed: Optional[int] = None,
    ):
        """初始化向量化环境。

        Args:
            nproc: 并行环境数
            device: 设备
            max_episode_steps: 每个 episode 的最大步数
            use_terrain_check: 是否启用地形检测
            seed: 随机种子（None = 不固定）
        """
        self.nproc = nproc
        self.device = device
        self.use_terrain_check = use_terrain_check
        self.seed = seed

        # 创建环境
        env_class = CraftgroundEnvWithTerrainCheck if use_terrain_check else MinecraftCraftgroundEnv

        self.envs = [
            env_class(seed=seed + i if seed is not None else i, max_steps=max_episode_steps)
            for i in range(nproc)
        ]

        # 成就追踪
        self.n_achievements = len(ALL_ACHIEVEMENTS)
        self.achievement_names = ALL_ACHIEVEMENTS
        self.current_achievements = np.zeros((nproc, self.n_achievements), dtype=np.int32)
        self.episode_achievements = [set() for _ in range(nproc)]

    def reset(self) -> torch.Tensor:
        """重置所有环境。

        Returns:
            obs: (B, C, H, W) float32 [0,1]
        """
        obs_list = [env.reset() for env in self.envs]
        self.current_achievements.fill(0)
        for i in range(self.nproc):
            self.episode_achievements[i].clear()

        return self._to_obs(np.array(obs_list))

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """步进所有环境。

        Args:
            actions: (B,) long，动作 ID

        Returns:
            obs: (B, C, H, W) float32 [0,1]
            rewards: (B, 1) float32
            dones: (B, 1) float32
            infos: dict with:
                - 'achievements': (B, n_achievements) int32（当前累计成就）
                - 'achievement_rewards': (B,) float32（本步成就奖励）
                - 'episode_done': (B,) bool（episode 完成）
        """
        obs_list, reward_list, done_list = [], [], []
        ach_reward_list = []
        achievement_vectors = []

        for i, (env, action) in enumerate(zip(self.envs, actions.cpu().numpy())):
            obs, rew, done, info = env.step(int(action))

            obs_list.append(obs)
            reward_list.append(rew)
            done_list.append(done)

            # 成就处理
            ach_reward = self._process_achievements(i, info)
            ach_reward_list.append(ach_reward)

            # 如果 done，重置成就追踪
            if done:
                self.episode_achievements[i].clear()
                obs, _ = env.reset(), None
                obs_list[-1] = obs

            achievement_vectors.append(self.current_achievements[i].copy())

        infos = {
            "achievements": torch.tensor(
                np.array(achievement_vectors), dtype=torch.int32, device=self.device
            ),
            "achievement_rewards": torch.tensor(
                ach_reward_list, dtype=torch.float32, device=self.device
            ).unsqueeze(1),
            "episode_done": np.array(done_list),
        }

        return (
            self._to_obs(np.array(obs_list)),
            torch.tensor(reward_list, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(done_list, dtype=torch.float32, device=self.device).unsqueeze(1),
            infos,
        )

    def _process_achievements(self, env_idx: int, info: Dict) -> float:
        """处理成就信息，返回成就奖励。

        Args:
            env_idx: 环境索引
            info: env.step() 返回的 info 字典

        Returns:
            本步成就奖励（新解锁的成就数）
        """
        ach_reward = 0.0

        # 从 info 提取成就信息
        if isinstance(info, dict) and "achievements" in info:
            new_achievements = info.get("successes", [])
            if isinstance(new_achievements, (list, np.ndarray)):
                for ach_id in new_achievements:
                    if ach_id not in self.episode_achievements[env_idx]:
                        self.episode_achievements[env_idx].add(ach_id)
                        self.current_achievements[env_idx, ach_id] = 1
                        ach_reward += 1.0  # 每个新成就 +1 奖励

        return ach_reward

    def get_achievement_vector(self) -> torch.Tensor:
        """获取当前成就向量。

        Returns:
            (B, n_achievements) int32，0/1 表示成就是否解锁
        """
        return torch.tensor(self.current_achievements, dtype=torch.int32, device=self.device)

    def close(self):
        """关闭所有环境。"""
        for env in self.envs:
            env.close()

    @staticmethod
    def _to_obs(raw: np.ndarray, target_h: int = 384, target_w: int = 640) -> torch.Tensor:
        """[N,H,W,C] uint8 → [N,C,H,W] float32 [0,1]。

        并自动填充到目标分辨率（兼容 YOLO 的 32 倍数要求）。
        使用上下对称零填充以保持图像在中心。

        Args:
            raw: 原始观测 (B, H, W, 3)
            target_h: 目标高度（默认 384，= ceil(360 / 32) * 32）
            target_w: 目标宽度（默认 640）

        Returns:
            观测张量 (B, 3, H, W)，填充到目标大小
        """
        if raw.ndim == 3:
            raw = raw[np.newaxis, ...]

        # 转换为 torch 张量并转移到 [0, 1]
        obs = torch.from_numpy(raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)

        # 填充到目标分辨率（如果需要）
        if obs.shape[2] != target_h or obs.shape[3] != target_w:
            pad_h = max(0, target_h - obs.shape[2])
            pad_w = max(0, target_w - obs.shape[3])

            if pad_h > 0 or pad_w > 0:
                # 上下对称填充，左右对称填充
                pad_h_top = pad_h // 2
                pad_h_bottom = pad_h - pad_h_top
                pad_w_left = pad_w // 2
                pad_w_right = pad_w - pad_w_left

                # pad: (left, right, top, bottom)
                obs = F.pad(obs, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
                           mode="constant", value=0.0)

        return obs
