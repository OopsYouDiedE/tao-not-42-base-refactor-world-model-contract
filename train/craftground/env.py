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
from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
from craftground.screen_encoding_modes import ScreenEncodingMode

from train.craftground.reward import RewardShaper

# ── V2_MINERL_HUMAN 动作空间 ────────────────────────────────────────────────
# V2 动作是一个 dict：布尔按键 (attack/forward/jump/...) + hotbar.1-9 +
# camera_pitch / camera_yaw（度，正 pitch=向下，正 yaw=向右）。
# 我们把策略输出的 Discrete(27) 映射为 27 个完整的 V2 动作 dict。
_CAM_SMALL = 15.0  # 小幅转头（度）
_CAM_BIG = 30.0    # 大幅转头（度）


def _v2(forward=False, back=False, left=False, right=False, jump=False,
        sneak=False, sprint=False, attack=False, use=False,
        cam_pitch=0.0, cam_yaw=0.0):
    """基于 no_op 构造一个完整的 V2 动作 dict（保证所有键齐全）。"""
    a = no_op_v2()
    a["forward"], a["back"] = forward, back
    a["left"], a["right"] = left, right
    a["jump"], a["sneak"], a["sprint"] = jump, sneak, sprint
    a["attack"], a["use"] = attack, use
    a["camera_pitch"], a["camera_yaw"] = float(cam_pitch), float(cam_yaw)
    return a


# 27 个离散动作 → V2 dict（语义与旧 V1 表对齐）
DISCRETE_TO_V2 = [
    _v2(),                                            # 0: no-op
    _v2(forward=True),                                # 1: forward
    _v2(back=True),                                   # 2: back
    _v2(left=True),                                   # 3: strafe left
    _v2(right=True),                                  # 4: strafe right
    _v2(jump=True),                                   # 5: jump
    _v2(forward=True, jump=True),                     # 6: forward+jump
    _v2(attack=True),                                 # 7: attack
    _v2(forward=True, attack=True),                   # 8: forward+attack
    _v2(use=True),                                    # 9: use
    _v2(forward=True, use=True),                      # 10: forward+use
    _v2(sneak=True),                                  # 11: sneak
    _v2(forward=True, sprint=True),                   # 12: forward+sprint
    _v2(cam_pitch=_CAM_SMALL),                        # 13: look down
    _v2(cam_pitch=-_CAM_SMALL),                       # 14: look up
    _v2(cam_yaw=_CAM_SMALL),                          # 15: look right
    _v2(cam_yaw=-_CAM_SMALL),                         # 16: look left
    _v2(cam_pitch=_CAM_BIG),                          # 17: look down (big)
    _v2(cam_pitch=-_CAM_BIG),                         # 18: look up (big)
    _v2(cam_yaw=_CAM_BIG),                            # 19: look right (big)
    _v2(cam_yaw=-_CAM_BIG),                           # 20: look left (big)
    _v2(forward=True, cam_pitch=_CAM_SMALL),          # 21: forward+look down
    _v2(forward=True, cam_pitch=-_CAM_SMALL),         # 22: forward+look up
    _v2(forward=True, cam_yaw=_CAM_SMALL),            # 23: forward+look right
    _v2(forward=True, cam_yaw=-_CAM_SMALL),           # 24: forward+look left
    _v2(attack=True, cam_pitch=_CAM_SMALL),           # 25: attack+look down
    _v2(attack=True, cam_pitch=-_CAM_SMALL),          # 26: attack+look up
]


class MinecraftCraftgroundEnv:
    """单个 Minecraft Craftground 环境，兼容 gym 接口。"""

    def __init__(self, seed: int = 0, max_steps: int = 1000, port: int = 8000):
        self.seed = seed
        self.max_steps = max_steps
        self.episode_step = 0
        self.reward_shaper = RewardShaper()
        self.screen_encoding_mode = ScreenEncodingMode.RAW

        # 检查 DISPLAY 环境变量
        if 'DISPLAY' not in os.environ:
            raise RuntimeError(
                "No DISPLAY environment variable found! \\n"
                "To run with GPU headless rendering or Xvfb, please use the gpu_run.sh wrapper:\\n"
                "    ./scripts/gpu_run.sh python your_script.py"
            )

        config = InitialEnvironmentConfig(screen_encoding_mode=self.screen_encoding_mode)
        self.env = CraftGroundEnvironment(
            config,
            action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
            port=port,
            find_free_port=True,
            verbose=False,
        )

    def reset(self) -> np.ndarray:
        """重置环境，返回 (H, W, C) uint8 numpy 观测。"""
        self.episode_step = 0
        self.reward_shaper.reset()
        obs, _ = self.env.reset()
        return obs["rgb"]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步，返回 (obs, reward, done, info)。

        Craftground 基类 reward 恒为 0、不提供成就，因此这里从 obs["full"]
        （protobuf）自行构造稠密内在奖励，并在 info["successes"] 中给出本步
        新解锁的成就索引（供上层 env_interface 统计与发放成就奖励）。
        """
        result = self.env.step(DISCRETE_TO_V2[int(action)])

        # craftground.step 返回 (final_obs, reward, done, truncated, final_obs)
        if len(result) == 5:
            obs, _rew, done, truncated, _info = result
            done = done or truncated
        else:
            obs, _rew, done, _info = result

        self.episode_step += 1

        # 从完整观测构造奖励与成就（在取 rgb 之前）
        full_obs = obs.get("full", None)
        intrinsic, new_ach_indices, force_done = (0.0, [], False)
        if full_obs is not None:
            intrinsic, new_ach_indices, force_done = self.reward_shaper.compute(
                full_obs, self.episode_step
            )

        rgb = obs["rgb"]
        info = {"achievements": True, "successes": new_ach_indices}

        # 死档(idle/无镐深入)触发强制重开
        if force_done:
            done = True

        if self.episode_step >= self.max_steps:
            done = True

        return rgb, float(intrinsic), done, info

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
        base_port: int = 8000,
    ):
        self.nproc = nproc
        self.device = device
        self.max_episode_steps = max_episode_steps

        self.envs = [
            MinecraftCraftgroundEnv(
                seed=i,
                max_steps=max_episode_steps,
                port=base_port + i * 10,
            )
            for i in range(nproc)
        ]

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
        """[N,H,W,C] uint8 numpy → [N,C,H,W] float32 [0,1] on CUDA。"""
        if raw.ndim == 3:  # 单张图像
            raw = raw[np.newaxis, ...]
        # Craftground 观测格式：(H, W, 3) RGB uint8
        return torch.from_numpy(raw.transpose(0, 3, 1, 2)).cuda().float() / 255.0


# 占位符：成就列表（待从 craftground 的实际成就系统提取）
ACHIEVEMENTS = [
    # Minecraft 标准成就（示例）
    # "minecraft.story.root",
    # "minecraft.story.mine_wood",
    # ... (实际列表待确认)
]
