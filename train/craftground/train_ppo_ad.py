#!/usr/bin/env python3
"""Craftground PPO + Achievement Distillation 训练 (train/craftground/train_ppo_ad.py)。

结合：
  1. YOLO-World-Dreamer 的世界模型 + 成就头 ψ(s)
  2. PPO 策略优化
  3. Achievement Distillation (AD) - 成就驱动探索

使用方法：
    python -m train.craftground.train_ppo_ad --n-envs 16 --total-timesteps 1000000
"""

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train.craftground.env import CraftgroundVecEnv
from train.craftground_minecraft_ml_env.achievements import ALL_ACHIEVEMENTS
from net.yoloworld.world_model import WorldModel
from net.yoloworld.craftground_config import CraftgroundConfig


class PPOADActor(nn.Module):
    """PPO Actor - 策略网络（基于世界模型隐状态）。

    Args:
        state_dim: 隐状态维度（来自 RSSM）
        num_actions: 动作数
        hidden_dim: 隐层维度
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Returns:
            action_logits: (B, A)
            value: (B, 1)
        """
        feat = self.net(state)
        return self.action_head(feat), self.value_head(feat)


class PPOADCritic(nn.Module):
    """PPO Critic - 值网络。"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="Craftground PPO+AD 训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16, help="并行环境数")
    p.add_argument("--n-steps", type=int, default=512, help="每次 rollout 步数")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--run-dir", default="runs/craftground_ppo_ad")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    """主训练循环。"""
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("🎮 Craftground PPO + Achievement Distillation 训练")
    print("=" * 70)
    print(f"📍 设备: {device}")
    print(f"🔧 配置:")
    print(f"   - 环境数: {args.n_envs}")
    print(f"   - 总步数: {args.total_timesteps}")
    print(f"   - 学习率: {args.lr}")
    print(f"   - PPO Epochs: {args.ppo_epochs}")

    # ─── 初始化环境 ────────────────────────────────────────────
    print("\n📦 初始化 Craftground 向量环境...")
    env = CraftgroundVecEnv(
        nproc=args.n_envs,
        device=args.device,
        max_episode_steps=1000,
    )
    obs = env.reset()
    print(f"✅ 环境初始化成功: obs.shape={obs.shape}")

    # ─── 初始化配置 ────────────────────────────────────────────
    cfg = CraftgroundConfig()
    cfg.n_achievements = len(ALL_ACHIEVEMENTS)
    cfg.num_actions = 27
    print(f"✅ 配置初始化: {cfg.n_achievements} 个成就, {cfg.num_actions} 个动作")

    # ─── 初始化世界模型 ────────────────────────────────────────
    print("\n🧠 初始化 YOLO-World 世界模型...")
    world_model = WorldModel(cfg).to(device)
    world_model_optim = optim.Adam(world_model.parameters(), lr=args.lr)
    print(f"✅ 世界模型初始化: {sum(p.numel() for p in world_model.parameters()) / 1e6:.1f}M 参数")

    # ─── 初始化 PPO Actor/Critic ──────────────────────────────
    print("\n🎯 初始化 PPO Actor/Critic...")
    state_dim = cfg.dyn_deter + cfg.dyn_stoch * cfg.dyn_discrete
    actor = PPOADActor(state_dim, cfg.num_actions).to(device)
    critic = PPOADCritic(state_dim).to(device)
    actor_optim = optim.Adam(actor.parameters(), lr=args.lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr)
    print(f"✅ Actor/Critic 初始化")

    # ─── 创建输出目录 ──────────────────────────────────────────
    os.makedirs(args.run_dir, exist_ok=True)

    # ─── 训练循环 ──────────────────────────────────────────────
    print("\n🚀 开始训练...")
    print("   [成就驱动探索 - Achievement Distillation]")
    print()

    start_time = time.time()
    total_steps = 0
    update_count = 0
    achievements_unlocked = set()

    try:
        while total_steps < args.total_timesteps:
            # Rollout 数据收集
            obs_list = []
            action_list = []
            reward_list = []
            done_list = []
            logprob_list = []
            value_list = []
            achievement_list = []

            for step in range(args.n_steps):
                # 世界模型推断：获取隐状态
                with torch.no_grad():
                    # TODO: 从观测编码到隐状态（需要实现 encoder）
                    # state = world_model.encoder(obs)
                    state = torch.randn(args.n_envs, state_dim, device=device)

                    # Actor 决策
                    action_logits, values = actor(state)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    actions = torch.multinomial(action_probs, 1).squeeze(1)
                    logprobs = torch.log_softmax(action_logits, dim=-1)
                    logprobs = logprobs.gather(1, actions.unsqueeze(1)).squeeze(1)

                # 环境步进
                obs, rewards, dones, infos = env.step(actions)

                # 记录数据
                obs_list.append(obs)
                action_list.append(actions)
                reward_list.append(rewards.squeeze(1))
                done_list.append(dones.squeeze(1))
                logprob_list.append(logprobs)
                value_list.append(values.squeeze(1))

                # 成就追踪
                for info in infos:
                    if isinstance(info, dict) and "achievements" in info:
                        # TODO: 聚合成就信息
                        pass

                total_steps += args.n_envs

            # PPO 更新
            if update_count % args.ppo_epochs == 0:
                # 计算 advantages (GAE)
                returns = compute_gae_returns(
                    reward_list,
                    value_list,
                    done_list,
                    gamma=cfg.discount,
                    gae_lambda=cfg.ppo_gae_lambda,
                )

                # PPO 策略更新（clip）
                for epoch in range(args.ppo_epochs):
                    # TODO: 实现完整的 PPO 梯度步
                    pass

                # 值函数更新
                # TODO: MSE loss on returns

            # 日志输出
            if update_count % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{update_count:5d} updates | {total_steps:7d} steps | {elapsed/3600:6.2f}h]"
                )
                print(f"  📊 已解锁成就: {len(achievements_unlocked)}/{cfg.n_achievements}")

            update_count += 1

    except KeyboardInterrupt:
        print("\n⏹️  训练中断（用户按 Ctrl+C）")
    finally:
        env.close()

    # ─── 总结 ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ 训练完成！耗时 {elapsed/3600:.2f} 小时")
    print(f"📊 总步数: {total_steps:,}")
    print(f"🏆 解锁成就: {len(achievements_unlocked)}/{cfg.n_achievements}")
    print("=" * 70)


def compute_gae_returns(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """计算 GAE advantages 和 returns。

    Args:
        rewards: list of (B,) tensors
        values: list of (B,) tensors
        dones: list of (B,) tensors
        gamma: discount factor
        gae_lambda: λ for GAE

    Returns:
        returns: list of (B,) tensors
    """
    # TODO: 实现 GAE 计算
    return rewards


if __name__ == "__main__":
    main()
