"""Crafter PPO + Achievement Distillation 训练主程序 (train/crafter/train_ppo_ad.py)。

使用方法(从仓库根目录执行):
    python -m train.crafter.train_ppo_ad
    python -m train.crafter.train_ppo_ad --n-envs 4 --total-timesteps 2000000

对外接口:
    main() — 解析 CLI 参数,初始化所有组件,运行训练循环。
"""
import argparse
import os
import time

import numpy as np
import torch

from net.ppo_ad.actor_critic import ActorCritic
from net.ppo_ad.config import PPOADConfig
from train.crafter.ad_buffer import AchievementBuffer, HARD_ACHIEVEMENTS
from train.crafter.env import VecCrafterEnv, SubprocVecCrafterEnv
from train.crafter.ppo_loss import ppo_loss
from train.crafter.rollout import RolloutBuffer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO + Achievement Distillation on Crafter")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ad-coef", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default="runs/crafter_ppo_ad")
    p.add_argument("--vec", choices=["serial", "subproc"], default="subproc",
                   help="subproc: 多 env 分摊到子进程并行(高吞吐);serial: 单进程串行。")
    p.add_argument("--n-workers", type=int, default=0,
                   help="subproc 子进程数(0=自动 min(n_envs,cpu-2))。")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg = PPOADConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        ad_coef=args.ad_coef,
    )

    # ── 模型与优化器 ──────────────────────────────────────────────────────────
    model = ActorCritic(
        encoder_depths=cfg.encoder_depths,
        encoder_kernel=cfg.encoder_kernel,
        encoder_stride=cfg.encoder_stride,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ActorCritic 参数量: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    # ── 环境、缓冲区 ──────────────────────────────────────────────────────────
    if args.vec == "subproc":
        envs = SubprocVecCrafterEnv(
            n_envs=cfg.n_envs, device=str(device), seed=args.seed,
            n_workers=(args.n_workers or None), demo_len=cfg.demo_len)
        print(f"向量环境: SubprocVecCrafterEnv "
              f"({cfg.n_envs} env / {envs.n_workers} 子进程并行)")
    else:
        envs = VecCrafterEnv(n_envs=cfg.n_envs, device=str(device), seed=args.seed)
        print(f"向量环境: VecCrafterEnv (单进程串行)")
    rollout = RolloutBuffer(
        n_envs=cfg.n_envs,
        n_steps=cfg.n_steps,
        obs_shape=(3, 64, 64),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        device=str(device),
    )
    ad_buf = AchievementBuffer(
        cap_per_achievement=cfg.ad_buffer_cap,
        device=str(device),
    )

    # ── 训练状态 ──────────────────────────────────────────────────────────────
    obs = envs.reset()
    done = torch.zeros(cfg.n_envs, device=device)
    ep_reward = torch.zeros(cfg.n_envs, device=device)
    ep_len = torch.zeros(cfg.n_envs, dtype=torch.long, device=device)

    finished_ep_rewards: list[float] = []
    finished_ep_lens: list[int] = []

    total_steps = 0
    n_updates = cfg.total_timesteps // (cfg.n_envs * cfg.n_steps)
    start_time = time.time()

    print(
        f"\nPPO+AD on Crafter | device={device} | n_envs={cfg.n_envs} | "
        f"n_steps={cfg.n_steps} | total_ts={cfg.total_timesteps:,} | "
        f"n_updates={n_updates}\n"
    )

    for update in range(1, n_updates + 1):
        # ── 收集轨迹 ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            for _ in range(cfg.n_steps):
                action, log_prob, _, value = model.get_action_and_value(obs)
                next_obs, reward, next_done, _, new_ach = envs.step(action)

                rollout.add(obs, action, log_prob, reward, done, value)

                # AD 缓冲区:新成就 → 存示范
                for env_idx, ach_name, obs_hist, act_hist in new_ach:
                    demo_obs = obs_hist[-cfg.demo_len:]
                    demo_act = act_hist[-cfg.demo_len:]
                    ad_buf.add_demo(ach_name, demo_obs, demo_act)

                ep_reward += reward
                ep_len += 1
                for i in range(cfg.n_envs):
                    if next_done[i]:
                        finished_ep_rewards.append(ep_reward[i].item())
                        finished_ep_lens.append(ep_len[i].item())
                        ep_reward[i] = 0.0
                        ep_len[i] = 0

                obs = next_obs
                done = next_done
                total_steps += cfg.n_envs

            last_value = model.get_action_and_value(obs)[3]

        rollout.compute_gae(last_value, done)

        # ── PPO + AD 更新 ─────────────────────────────────────────────────────
        model.train()
        pg_losses, v_losses, ents, ad_losses = [], [], [], []

        for _ in range(cfg.n_epochs):
            for mb_obs, mb_act, mb_lp, mb_adv, mb_ret, mb_val in \
                    rollout.get_minibatches(cfg.minibatch_size):

                _, new_lp, entropy, new_val = model.get_action_and_value(mb_obs, mb_act)
                loss, pg_l, v_l, ent_l = ppo_loss(
                    new_lp, mb_lp, mb_adv, mb_ret, new_val,
                    clip_coef=cfg.clip_coef,
                    vf_coef=cfg.vf_coef,
                    ent_coef=cfg.ent_coef,
                    entropy=entropy,
                )

                # Achievement Distillation BC 损失
                ad_obs, ad_act = ad_buf.sample(cfg.ad_batch_size)
                ad_l = torch.zeros(1, device=device).squeeze()
                if ad_obs is not None:
                    logits, _ = model(ad_obs)
                    ad_l = torch.nn.functional.cross_entropy(logits, ad_act)
                    loss = loss + cfg.ad_coef * ad_l

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_l.item())
                v_losses.append(v_l.item())
                ents.append(ent_l.item())
                ad_losses.append(ad_l.item())

        # ── 日志 ──────────────────────────────────────────────────────────────
        if update % cfg.log_interval == 0:
            sps = int(total_steps / (time.time() - start_time + 1e-6))
            recent = finished_ep_rewards[-100:] if finished_ep_rewards else [0.0]
            recent_len = finished_ep_lens[-100:] if finished_ep_lens else [0]
            covered = ad_buf.covered_names()
            hard = [a for a in covered if a in HARD_ACHIEVEMENTS]
            print(
                f"upd={update:5d} | steps={total_steps:>9,} | sps={sps:>5,} | "
                f"ep_rew={np.mean(recent):6.3f} | ep_len={int(np.mean(recent_len)):>5} | "
                f"pg={np.mean(pg_losses):+.4f} | vf={np.mean(v_losses):.4f} | "
                f"ent={np.mean(ents):.4f} | ad={np.mean(ad_losses):.4f} | "
                f"ad_cov={ad_buf.coverage()}/22 | hard={len(hard)}/{len(HARD_ACHIEVEMENTS)} "
                f"{hard if hard else ''}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if update % cfg.save_interval == 0:
            path = os.path.join(ckpt_dir, f"ckpt_{update:06d}.pt")
            torch.save({
                "update": update,
                "total_steps": total_steps,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ep_rewards": finished_ep_rewards,
            }, path)
            print(f"  → checkpoint 已保存: {path}")

    # ── 最终保存 ──────────────────────────────────────────────────────────────
    final_path = os.path.join(args.run_dir, "final.pt")
    torch.save({
        "update": n_updates,
        "total_steps": total_steps,
        "model_state": model.state_dict(),
        "ep_rewards": finished_ep_rewards,
    }, final_path)
    print(f"\n训练完成。最终模型: {final_path}")
    if finished_ep_rewards:
        print(f"最近 100 ep 平均奖励: {np.mean(finished_ep_rewards[-100:]):.4f}")
    envs.close()


if __name__ == "__main__":
    main()
