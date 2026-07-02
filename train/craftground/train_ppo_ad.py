#!/usr/bin/env python3
"""Craftground PPO + Achievement Distillation (Model-Free)。

纯 model-free 强化学习：
  1. YOLO26s 编码器 → 特征提取
  2. PPO Actor/Critic → 策略优化
  3. 成就头 ψ(s) → 成就预测
  4. Achievement Distillation → 成就驱动奖励塑形

使用方法：
    python -m train.craftground.train_ppo_ad --n-envs 16 --total-timesteps 1000000
"""

import argparse
import copy
import os
import queue
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from train.craftground.env_interface import CraftgroundVecEnvWithInterface
from train.craftground.achievements import ALL_ACHIEVEMENTS
from net.encoders.yolo_backbone_encoder import YoloBackboneEncoder


class PPOActorCritic(nn.Module):
    """PPO Actor-Critic 网络（从 YOLO 特征输入）。"""

    def __init__(self, feature_dim: int = 512, num_actions: int = 27, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions

        # 共享特征处理
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head：动作概率
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
        )

        # Critic head：值函数
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, feature_dim) from YOLO encoder

        Returns:
            action_logits: (B, num_actions)
            values: (B, 1)
        """
        shared_feat = self.shared(features)
        action_logits = self.action_head(shared_feat)
        values = self.value_head(shared_feat)
        return action_logits, values


class AchievementHead(nn.Module):
    """成就预测头 ψ(s)：从特征预测成就完成度。"""

    def __init__(self, feature_dim: int = 512, num_achievements: int = 100, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_achievements),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim)

        Returns:
            achievement_logits: (B, num_achievements) for BCE loss
        """
        return self.net(features)


def update_encoder_freeze_schedule(
    encoder: nn.Module,
    current_steps: int,
    freeze_start: int = 50_000,
    freeze_end: int = 150_000,
) -> float:
    """更新编码器的冻结状态。

    Args:
        encoder: YOLO 编码器
        current_steps: 当前环境步数
        freeze_start: 冻结结束点（之前完全冻结）
        freeze_end: 解冻结束点（之后完全解冻）

    Returns:
        freeze_ratio: 当前冻结比例 [0, 1]（用于日志）
    """
    if current_steps < freeze_start:
        # 完全冻结
        freeze_ratio = 1.0
        for param in encoder.backbone.parameters():
            param.requires_grad = False
    elif current_steps < freeze_end:
        # 线性过渡：从冻结到解冻
        progress = (current_steps - freeze_start) / (freeze_end - freeze_start)
        freeze_ratio = max(0.0, 1.0 - progress)

        # 逐步启用梯度
        for param in encoder.backbone.parameters():
            param.requires_grad = True

        # 用梯度缩放控制解冻程度
        # freeze_ratio=1→梯度缩放=0（冻结），freeze_ratio=0→梯度缩放=1（完全解冻）
        for param in encoder.backbone.parameters():
            param._grad_scale = 1.0 - freeze_ratio
    else:
        # 完全解冻
        freeze_ratio = 0.0
        for param in encoder.backbone.parameters():
            param.requires_grad = True
            param._grad_scale = 1.0

    return freeze_ratio


def compute_gae_advantages(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    dones: List[torch.Tensor],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 GAE advantages 和 returns。

    Args:
        rewards: list of (B,) tensors
        values: list of (B,) tensors
        dones: list of (B,) tensors
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (T*B,) tensor
        returns: (T*B,) tensor
    """
    T = len(rewards)
    B = rewards[0].shape[0]

    advantages = torch.zeros(T, B, device=rewards[0].device)
    gae = torch.zeros(B, device=rewards[0].device)

    # 倒序计算 GAE（values 含 bootstrap：长度 T+1）
    for t in reversed(range(T)):
        next_value = values[t + 1]

        # TD residual
        td_residual = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE 递推
        gae = td_residual + gamma * gae_lambda * (1 - dones[t]) * gae

        advantages[t] = gae

    returns = advantages + torch.stack(values[:T])

    # 展平
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)

    return advantages, returns


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="Craftground PPO+AD (Model-Free) 训练")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epochs", type=int, default=2,
                   help="每个 rollout 的 PPO 复用轮数；从 4 降到 2 砍半更新成本(占墙钟约六成)，"
                        "代价是样本复用变少；与流水线异步的滞后不要叠加过高")
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--ppo-batch-size", type=int, default=256)
    p.add_argument("--ad-scale", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--encoder-freeze-start", type=int, default=0,
                   help="编码器冻结终点（环境步数）；0=无初始全冻结期，从第0步起线性解冻")
    p.add_argument("--encoder-freeze-end", type=int, default=100_000,
                   help="编码器解冻终点（环境步数）；0-100k 线性解冻")
    p.add_argument("--run-dir", default="runs/craftground_ppo_ad_v1")
    p.add_argument("--save-interval", type=int, default=100,
                   help="每多少次 PPO 更新存一个 checkpoint（0=只在结束/中断时存 final.pt）")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-episode-steps", type=int, default=72000,
                   help="单 episode 最大步数。20Hz 下 72000≈60分钟≈3个昼夜循环(单循环=20分钟)。"
                        "上限较长，早期烂局靠死档机制提前重置，不会真跑满。")
    p.add_argument("--use-terrain-check", action="store_true", default=True)
    p.add_argument("--screen-encoding", choices=["raw", "zerocopy"], default="raw",
                   help="raw=CPU像素回传; zerocopy=GPU渲染零拷贝→torch tensor（需 DISPLAY 指向 GPU）")
    p.add_argument("--eval-random", action="store_true", default=False,
                   help="随机策略 baseline：均匀随机动作、不学习，只统计 per-episode 成功率作对照")
    p.add_argument("--async-collect", action="store_true", default=False,
                   help="有界流水线异步：后台线程采集下个rollout与PPO更新重叠(滞后≤1)，藏掉更新空转。"
                        "默认关闭；需同种子 A/B 验证不损训练后再用")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="混合精度(fp16 autocast + GradScaler)：编码器前向/反向用 fp16，"
                        "激活显存减半 + Tensor Core 提速。--no-amp 关闭。仅 cuda 生效")
    return p.parse_args()


def collect_rollout(encoder, actor_critic, env, obs, n_steps, n_envs,
                    ad_scale, device, eval_random, num_actions):
    """用给定(采集)策略跑一个 rollout，全程 no_grad。

    异步时 encoder/actor_critic 是 behavior 副本（滞后 ~1 rollout）；同步时就是主网络。

    Returns:
        data: dict，含 obs/action/reward/done/logprob/value/ach_r/ach 各为长度 n_steps 的列表
        last_obs: rollout 结束后的观测（供 bootstrap / 下一个 rollout）
        newly_unlocked: 本 rollout 内出现过的成就索引集合
        env_steps: 本 rollout 推进的环境步数（n_steps * n_envs）
    """
    obs_list, action_list, reward_list, done_list = [], [], [], []
    logprob_list, value_list, ach_r_list, ach_list = [], [], [], []
    newly_unlocked = set()
    for _ in range(n_steps):
        with torch.no_grad():
            features = encoder(obs)
            action_logits, values = actor_critic(features)
            if eval_random:
                actions = torch.randint(0, num_actions, (n_envs,), device=device)
            else:
                action_probs = torch.softmax(action_logits, dim=-1)
                actions = torch.multinomial(action_probs, 1).squeeze(1)
            logprobs = torch.log_softmax(action_logits, dim=-1).gather(
                1, actions.unsqueeze(1)).squeeze(1)

        obs, rewards, dones, infos = env.step(actions)
        ach_rewards = infos["achievement_rewards"].squeeze(1)
        total_rewards = rewards.squeeze(1) + ad_scale * ach_rewards

        # 观测以 uint8 存（[0,1]float → [0,255]uint8），rollout 缓冲省 4×；
        # 编码器入口会自动把 uint8 转回 float/255（见 YoloBackboneEncoder.forward）。
        obs_list.append((obs * 255.0).clamp_(0, 255).to(torch.uint8))
        action_list.append(actions)
        reward_list.append(total_rewards)
        done_list.append(dones.squeeze(1))
        logprob_list.append(logprobs)
        value_list.append(values.squeeze(1))
        ach_r_list.append(ach_rewards)
        ach_list.append(infos["achievements"].detach())

        ach_vec = infos["achievements"]
        for i in range(n_envs):
            for j in range(ach_vec.shape[1]):
                if ach_vec[i, j].item() == 1:
                    newly_unlocked.add(j)

    data = {
        "obs": obs_list, "action": action_list, "reward": reward_list,
        "done": done_list, "logprob": logprob_list, "value": value_list,
        "ach_r": ach_r_list, "ach": ach_list,
    }
    return data, obs, newly_unlocked, n_steps * n_envs


def ppo_update(data, last_obs, encoder, actor_critic, achievement_head,
               optimizer, scaler, use_amp, args):
    """对一个 rollout 跑 PPO+AD 更新（就地修改 encoder/actor_critic/achievement_head）。

    use_amp=True 时编码器前向/反向走 fp16 autocast，激活显存减半 + Tensor Core 提速；
    GradScaler 防 fp16 梯度下溢，且在 unscale_ 之后再应用冻结计划的 `_grad_scale`，最后 clip → step。
    """
    reward_list, value_list, done_list = data["reward"], data["value"], data["done"]
    obs_list, action_list, logprob_list, ach_list = (
        data["obs"], data["action"], data["logprob"], data["ach"])

    with torch.no_grad():
        _, last_values = actor_critic(encoder(last_obs))
        last_values = last_values.squeeze(1)

    advantages, returns = compute_gae_advantages(
        reward_list, value_list + [last_values], done_list,
        gamma=args.gamma, gae_lambda=args.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = TensorDataset(
        torch.cat(obs_list), torch.cat(action_list).unsqueeze(1),
        torch.cat(logprob_list), returns, advantages, torch.cat(ach_list))
    dataloader = DataLoader(dataset, batch_size=args.ppo_batch_size, shuffle=True)

    all_params = (list(encoder.parameters()) + list(actor_critic.parameters())
                  + list(achievement_head.parameters()))

    for _ in range(args.ppo_epochs):
        for (b_obs, b_act, b_lp, b_ret, b_adv, b_ach) in dataloader:
            with torch.autocast(device_type="cuda", enabled=use_amp):
                b_feat = encoder(b_obs)  # 编码器入口把 uint8 → float/255
                action_logits, values = actor_critic(b_feat)
                logprobs_new = torch.log_softmax(action_logits, dim=-1).gather(
                    1, b_act).squeeze(1)
                ratio = torch.exp(logprobs_new - b_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values.squeeze(1) - b_ret) ** 2).mean()
                probs = torch.softmax(action_logits, dim=-1)
                entropy = -(probs * torch.log_softmax(action_logits, dim=-1)).sum(dim=-1)
                entropy_loss = -entropy.mean()
                ach_loss = nn.BCEWithLogitsLoss()(achievement_head(b_feat), b_ach.float())
                total_loss = (policy_loss + args.value_coeff * value_loss
                              + args.entropy_coeff * entropy_loss + ach_loss)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            # 必须先 unscale_，才能在"真实尺度"的梯度上应用冻结计划缩放 + clip
            scaler.unscale_(optimizer)
            for param in encoder.backbone.parameters():
                if hasattr(param, "_grad_scale") and param.grad is not None:
                    param.grad *= param._grad_scale
            nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()


def main():
    """主训练循环。"""
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.eval_random:
        # 随机 baseline：禁用一切学习（ppo_epochs=0 → 更新循环空转），只跑+统计成功率
        args.ppo_epochs = 0
        print("🎲 [eval-random] 随机策略 baseline 模式：不学习，仅统计 per-episode 成功率")

    print("\n" + "=" * 80)
    print("🎮 Craftground PPO + Achievement Distillation (Model-Free)")
    print("=" * 80)
    print(f"📍 设备: {device}")
    print(f"🔧 配置:")
    print(f"   - 环境数: {args.n_envs}")
    print(f"   - Rollout 步数: {args.n_steps}")
    print(f"   - 总步数: {args.total_timesteps:,}")
    print(f"   - 学习率: {args.lr}")
    print(f"   - PPO Epochs: {args.ppo_epochs}")
    print(f"   - AD Scale: {args.ad_scale}")

    # ─── 初始化环境 ────────────────────────────────────────────────
    print("\n📦 初始化 Craftground 向量环境...")
    from craftground.screen_encoding_modes import ScreenEncodingMode
    enc_mode = (ScreenEncodingMode.ZEROCOPY
                if args.screen_encoding == "zerocopy"
                else ScreenEncodingMode.RAW)
    print(f"   - 渲染编码: {args.screen_encoding}"
          + ("（GPU 零拷贝，需 DISPLAY 指向 GPU 屏）" if enc_mode == ScreenEncodingMode.ZEROCOPY else "（RAW，CPU 像素回传）"))
    env = CraftgroundVecEnvWithInterface(
        nproc=args.n_envs,
        device=args.device,
        max_episode_steps=args.max_episode_steps,
        use_terrain_check=args.use_terrain_check,
        seed=args.seed,
        screen_encoding_mode=enc_mode,
    )
    obs = env.reset()
    print(f"✅ 环境初始化: obs.shape={obs.shape}")

    num_actions = 27
    num_achievements = len(ALL_ACHIEVEMENTS)
    feature_dim = 512

    # ─── 初始化 YOLO 编码器 ────────────────────────────────────────
    print("\n🔍 初始化 YOLO26s 编码器...")
    encoder = YoloBackboneEncoder(
        output_dim=feature_dim,
        pretrained=True,
        freeze_backbone=False,  # 允许梯度流，但用 freeze_schedule 控制
        device=args.device,
    )
    yolo_params = sum(p.numel() for p in encoder.parameters())

    # 初始化冻结：前 50k 步冻结 backbone
    for param in encoder.backbone.parameters():
        param.requires_grad = False
        param._grad_scale = 0.0

    print(f"✅ YOLO26s Backbone 初始化: {yolo_params / 1e6:.2f}M 参数")
    print(f"   冻结计划: 0-{args.encoder_freeze_start//1000}k 冻结, "
          f"{args.encoder_freeze_start//1000}k-{args.encoder_freeze_end//1000}k 线性解冻")

    # ─── 初始化 PPO Actor/Critic ──────────────────────────────────
    print("\n🎯 初始化 PPO Actor/Critic...")
    actor_critic = PPOActorCritic(
        feature_dim=feature_dim,
        num_actions=num_actions,
        hidden_dim=256,
    ).to(device)
    ac_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"✅ Actor/Critic 初始化: {ac_params / 1e6:.2f}M 参数")

    # ─── 初始化成就头 ────────────────────────────────────────────
    print("\n🏆 初始化成就预测头...")
    achievement_head = AchievementHead(
        feature_dim=feature_dim,
        num_achievements=num_achievements,
        hidden_dim=256,
    ).to(device)
    ach_params = sum(p.numel() for p in achievement_head.parameters())
    print(f"✅ 成就头初始化: {ach_params / 1e6:.2f}M 参数")

    # ─── 初始化优化器 ──────────────────────────────────────────────
    print("\n⚙️  初始化优化器...")
    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(actor_critic.parameters())
        + list(achievement_head.parameters()),
        lr=args.lr,
    )
    print(f"✅ 优化器初始化（包含编码器）")

    # ─── 混合精度 ─────────────────────────────────────────────────
    use_amp = args.amp and device.type == "cuda"  # AMP 仅 cuda 生效
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"⚙️  混合精度(AMP): {'启用 fp16' if use_amp else '关闭(fp32)'}")

    # ─── 创建输出目录 ──────────────────────────────────────────────
    os.makedirs(args.run_dir, exist_ok=True)

    # ─── 训练统计 ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 模型参数统计:")
    print(f"   YOLO26s Backbone:  {yolo_params / 1e6:7.2f}M")
    print(f"   Actor/Critic:      {ac_params / 1e6:7.2f}M")
    print(f"   Achievement Head:  {ach_params / 1e6:7.2f}M")
    print(f"   {'─' * 76}")
    total_params = yolo_params + ac_params + ach_params
    print(f"   总计:              {total_params / 1e6:7.2f}M")
    print("=" * 80)

    # ─── 训练循环 ──────────────────────────────────────────────────
    print("\n🚀 开始训练...\n")

    start_time = time.time()
    total_steps = 0
    update_count = 0
    achievements_unlocked = set()
    freeze_ratio_current = 1.0  # 追踪冻结状态

    def save_ckpt(tag: str):
        """保存编码器/AC/成就头/优化器到 run_dir/<tag>.pt。"""
        path = os.path.join(args.run_dir, f"{tag}.pt")
        torch.save({
            "encoder": encoder.state_dict(),
            "actor_critic": actor_critic.state_dict(),
            "achievement_head": achievement_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "total_steps": total_steps,
            "update_count": update_count,
            "achievements_unlocked": sorted(achievements_unlocked),
            "args": vars(args),
        }, path)
        print(f"💾 已保存 {path}（{total_steps:,} 步，成就 {len(achievements_unlocked)}/{num_achievements}）")

    # ─── rollout 来源：同步=主线程直接采集；异步=后台线程经队列(双缓冲) ──
    async_mode = args.async_collect and not args.eval_random
    collector_thread = None
    stop_evt = threading.Event()
    if async_mode:
        # behavior 策略：新建实例(deepcopy 会破坏 forward hook 闭包)，load_state_dict 同步权重
        behavior_encoder = YoloBackboneEncoder(
            output_dim=feature_dim, pretrained=False, freeze_backbone=False, device=args.device)
        behavior_ac = PPOActorCritic(feature_dim, num_actions, 256).to(device)
        behavior_encoder.load_state_dict(encoder.state_dict())
        behavior_ac.load_state_dict(actor_critic.state_dict())
        behavior_encoder.eval()
        behavior_ac.eval()

        rollout_q = queue.Queue(maxsize=1)   # maxsize=1 → 采集最多领先 1 个 rollout（滞后上界）
        weight_lock = threading.Lock()
        shared = {"enc_sd": None, "ac_sd": None, "ready": False}

        def collector_loop():
            c_obs = env.reset()
            while not stop_evt.is_set():
                if shared["ready"]:
                    with weight_lock:
                        behavior_encoder.load_state_dict(shared["enc_sd"])
                        behavior_ac.load_state_dict(shared["ac_sd"])
                        shared["ready"] = False
                data, c_obs, unlocked, steps = collect_rollout(
                    behavior_encoder, behavior_ac, env, c_obs,
                    args.n_steps, args.n_envs, args.ad_scale, device, False, num_actions)
                placed = False
                while not placed and not stop_evt.is_set():
                    try:
                        rollout_q.put((data, c_obs, unlocked, steps), timeout=1.0)
                        placed = True
                    except queue.Full:
                        pass

        collector_thread = threading.Thread(target=collector_loop, daemon=True)
        collector_thread.start()
        print("⚡ 异步采集已启用（双缓冲，策略滞后 ≤ 1 rollout）")

    rollout_state = {"obs": obs}

    try:
        while total_steps < args.total_timesteps:
            # 取一个 rollout
            if async_mode:
                data, last_obs, unlocked, steps = rollout_q.get()
            else:
                data, last_obs, unlocked, steps = collect_rollout(
                    encoder, actor_critic, env, rollout_state["obs"],
                    args.n_steps, args.n_envs, args.ad_scale, device,
                    args.eval_random, num_actions)
                rollout_state["obs"] = last_obs

            total_steps += steps
            achievements_unlocked |= unlocked
            reward_list = data["reward"]
            achievement_reward_list = data["ach_r"]

            # ─── 更新编码器冻结计划 ──────────────────────────────
            freeze_ratio_current = update_encoder_freeze_schedule(
                encoder,
                total_steps,
                freeze_start=args.encoder_freeze_start,
                freeze_end=args.encoder_freeze_end,
            )

            # ─── PPO+AD 更新（eval_random 模式跳过，纯收集统计） ──────────
            if not args.eval_random:
                ppo_update(data, last_obs, encoder, actor_critic,
                           achievement_head, optimizer, scaler, use_amp, args)
                if async_mode:
                    # 把更新后的权重快照交给采集线程（clone 避免读到正在写的张量；滞后 ≤ 1 rollout）
                    with weight_lock:
                        shared["enc_sd"] = {k: v.detach().clone()
                                            for k, v in encoder.state_dict().items()}
                        shared["ac_sd"] = {k: v.detach().clone()
                                           for k, v in actor_critic.state_dict().items()}
                        shared["ready"] = True

            # ─── 日志输出 ────────────────────────────────────────
            if update_count % args.log_interval == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean([r.mean().item() for r in reward_list])
                avg_ach_reward = np.mean([a.mean().item() for a in achievement_reward_list])

                freeze_status = "🔒冻结" if freeze_ratio_current > 0.99 else f"🔓{1-freeze_ratio_current:.0%}"

                print(
                    f"[{update_count:5d} | {total_steps:,} steps | {elapsed/60:6.1f}m] "
                    f"R={avg_reward:6.2f} | R_ach={avg_ach_reward:6.2f} | "
                    f"成就={len(achievements_unlocked):2d}/{num_achievements} | "
                    f"编码器{freeze_status}"
                )
                # per-episode 成功率（强指标：能否稳定复现，而非曾解锁一次）
                rates, n_ep = env.get_episode_success_rates()
                if n_ep > 0:
                    shown = [(ALL_ACHIEVEMENTS[idx].split(".")[-1], rt)
                             for idx, rt in sorted(rates.items())
                             if rt > 0 and ALL_ACHIEVEMENTS[idx] != "minecraft.story.root"]
                    if shown:
                        rate_str = ", ".join(f"{name}={rt:.0%}" for name, rt in shown)
                        print(f"        └─ 近{n_ep}局成功率: {rate_str}")
                    else:
                        print(f"        └─ 近{n_ep}局: 除root外无成就达成")

            update_count += 1
            if args.save_interval > 0 and update_count % args.save_interval == 0:
                save_ckpt(f"ckpt_{total_steps:08d}")

    except KeyboardInterrupt:
        print("\n⏹️  训练中断")
    finally:
        # 先停采集线程（异步），否则它还在 env.step，close 会崩
        stop_evt.set()
        if collector_thread is not None:
            try:
                rollout_q.get_nowait()  # 腾出队列，解开采集线程可能的 put 阻塞
            except Exception:
                pass
            collector_thread.join(timeout=30)
        # finally 在正常完成/中断/异常(含 OOM)时都会执行 → 权重不丢
        try:
            save_ckpt("final")
        except Exception as e:
            print(f"⚠️  最终保存失败: {e}")
        env.close()

    # ─── 总结 ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ 训练完成！耗时 {elapsed/3600:.2f} 小时")
    print(f"📊 总步数: {total_steps:,}")
    print(f"🏆 解锁成就: {len(achievements_unlocked)}/{num_achievements}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
