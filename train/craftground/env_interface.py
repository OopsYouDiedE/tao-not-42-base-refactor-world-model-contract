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
from collections import deque
from typing import Tuple, Dict, Optional
from craftground.screen_encoding_modes import ScreenEncodingMode
from train.craftground.env import MinecraftCraftgroundEnv, CraftgroundVecEnv
from train.craftground_minecraft_ml_env.achievements import ALL_ACHIEVEMENTS


class CraftgroundEnvWithTerrainCheck(MinecraftCraftgroundEnv):
    """添加地形检测的单环境。

    初始化时自动检测地形，如果不利就重新启动。
    """

    def __init__(self, seed: int = 0, max_steps: int = 1000,
                 screen_encoding_mode: ScreenEncodingMode = ScreenEncodingMode.RAW):
        super().__init__(seed=seed, max_steps=max_steps,
                         screen_encoding_mode=screen_encoding_mode)
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
        # obs 可能是 numpy (RAW) 或 CUDA torch.Tensor (ZEROCOPY)，统一取出
        # 三通道均值的 python float，避免对 tensor 直接做布尔运算。
        if isinstance(obs, torch.Tensor):
            x = obs.float()
            if x.max() > 1.0:
                x = x / 255.0
            r = float(x[..., 0].mean())
            g = float(x[..., 1].mean())
            b = float(x[..., 2].mean())
        else:
            obs_norm = obs.astype(np.float32) / 255.0 if obs.dtype == np.uint8 else obs
            r = float(obs_norm[..., 0].mean())
            g = float(obs_norm[..., 1].mean())
            b = float(obs_norm[..., 2].mean())

        # 启发式判断（宽松：只排除明显的海洋/水面出生点）。
        # 注意：旧判据 g-r-b>0.05 对任何真实画面几乎恒为 False（绿通道均值
        # 不可能高过 红+蓝 之和），导致每次 reset 都白跑满 5 次重启再强制接受，
        # 既没真正筛地形又制造海量 Minecraft 重启 + RAM 抖动。现改为：
        # 蓝色明显主导（b 比 g 高出阈值）才判为水面/海洋 → 重试；其余陆地直接接受。
        blue_dominance = b - g  # 蓝色相对绿色的优势（水面/海洋偏高）
        is_ocean = blue_dominance > 0.10

        return not is_ocean


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
        screen_encoding_mode: ScreenEncodingMode = ScreenEncodingMode.RAW,
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
            env_class(seed=seed + i if seed is not None else i, max_steps=max_episode_steps,
                      screen_encoding_mode=screen_encoding_mode)
            for i in range(nproc)
        ]

        # 成就追踪
        self.n_achievements = len(ALL_ACHIEVEMENTS)
        self.achievement_names = ALL_ACHIEVEMENTS
        self.current_achievements = np.zeros((nproc, self.n_achievements), dtype=np.int32)
        self.episode_achievements = [set() for _ in range(nproc)]
        # 每个**完成**的 episode 解锁了哪些成就（索引集合），滑窗保留最近若干个，
        # 用于算 per-episode 成功率（替代"曾解锁"的弱指标）。
        self.completed_episode_ach = deque(maxlen=300)

    def reset(self) -> torch.Tensor:
        """重置所有环境。

        Returns:
            obs: (B, C, H, W) float32 [0,1]
        """
        obs_list = [env.reset() for env in self.envs]
        self.current_achievements.fill(0)
        for i in range(self.nproc):
            self.episode_achievements[i].clear()

        return self._to_obs(obs_list)

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

            # 如果 done，先把本 episode 解锁的成就快照进滑窗，再重置追踪
            if done:
                self.completed_episode_ach.append(set(self.episode_achievements[i]))
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
            self._to_obs(obs_list),
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

    def get_episode_success_rates(self) -> Tuple[Dict[int, float], int]:
        """基于最近完成的 episode，返回每个成就的 per-episode 成功率。

        这是比"曾解锁(current_achievements)"强得多的指标：它衡量智能体
        **能否稳定复现**某成就，而非历史上撞到过一次。

        Returns:
            (rates, n): rates[成就索引] = 最近 n 个完成 episode 中解锁该成就的比例；
                        n = 统计用的已完成 episode 数（0 表示还没有 episode 结束）。
        """
        n = len(self.completed_episode_ach)
        if n == 0:
            return {}, 0
        rates: Dict[int, float] = {}
        for idx in range(self.n_achievements):
            rates[idx] = sum(1 for s in self.completed_episode_ach if idx in s) / n
        return rates, n

    def close(self):
        """关闭所有环境。"""
        for env in self.envs:
            env.close()

    def _to_obs(self, raw, target_h: int = 384, target_w: int = 640) -> torch.Tensor:
        """每环境 (H,W,3) 帧 → [N,C,H,W] float32 [0,1]，移到 self.device。

        raw 可为：
          - list[np.ndarray]   (RAW 编码，CPU)
          - list[torch.Tensor] (ZEROCOPY 编码，已在 GPU，零拷贝)
          - np.ndarray (N,H,W,3) 或 (H,W,3)  （向后兼容）
        """
        if isinstance(raw, list):
            if len(raw) > 0 and isinstance(raw[0], torch.Tensor):
                # ZEROCOPY：各帧已是 GPU 上的 (H,W,3) tensor，stack 不经 CPU
                stacked = torch.stack(raw, dim=0)              # (N,H,W,3)
                obs = stacked.permute(0, 3, 1, 2).float()      # (N,3,H,W)
                if obs.max() > 1.0:
                    obs = obs / 255.0
            else:
                arr = np.array(raw)                            # (N,H,W,3)
                obs = torch.from_numpy(arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        else:
            arr = raw
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            obs = torch.from_numpy(arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)

        obs = obs.to(self.device)

        if obs.shape[2] != target_h or obs.shape[3] != target_w:
            pad_h = max(0, target_h - obs.shape[2])
            pad_w = max(0, target_w - obs.shape[3])
            if pad_h > 0 or pad_w > 0:
                pad_h_top = pad_h // 2
                pad_h_bottom = pad_h - pad_h_top
                pad_w_left = pad_w // 2
                pad_w_right = pad_w - pad_w_left
                obs = F.pad(obs, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
                           mode="constant", value=0.0)

        return obs.to(self.device)
