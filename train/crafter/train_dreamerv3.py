"""DreamerV3 Crafter 训练主程序 (train/crafter/train_dreamerv3.py)。

使用方法(从仓库根目录执行):
    python -m train.crafter.train_dreamerv3 --size small --total-steps 200000
    python -m train.crafter.train_dreamerv3 --size tiny --total-steps 4000   # 冒烟

对外接口:
    main() — 解析 CLI,装配 DreamerV3 + 环境 + 回放,运行"采集 ↔ 世界模型/想象更新"循环。

世界模型与想象 actor-critic 用各自优化器分开更新(actor/critic 不回传梯度到世界模型)。
模型结构走 net.dreamerv3.DreamerV3Config(--size 预设),训练旋钮走 CLI。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from net.dreamerv3 import build_dreamerv3
from train.crafter.env import VecCrafterEnv
from train.crafter.ad_buffer import ACHIEVEMENTS
from train.crafter.dreamer_buffer import SequenceReplay

N_ACTIONS = 17

# --size 预设:结构规模(部件超参覆盖 DreamerV3Config 默认)。
SIZE_PRESETS = {
    "tiny": dict(dyn_deter=128, dyn_stoch=8, dyn_discrete=8, dyn_hidden=128,
                 units=128, mlp_layers=1, enc_depths=(16, 32, 64, 128),
                 dec_depths=(128, 64, 32, 16), horizon=8),
    "small": dict(dyn_deter=256, dyn_stoch=24, dyn_discrete=24, dyn_hidden=256,
                  units=256, mlp_layers=2, enc_depths=(24, 48, 96, 192),
                  dec_depths=(192, 96, 48, 24), horizon=15),
    "default": dict(),   # 用 DreamerV3Config 全量默认(deter=512, 32×32, units=512)
}


def parse_args():
    p = argparse.ArgumentParser(description="DreamerV3 on Crafter")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size", choices=list(SIZE_PRESETS), default="small")
    p.add_argument("--total-steps", type=int, default=200_000, help="总环境交互步数")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--prefill", type=int, default=2000, help="随机策略预填步数")
    p.add_argument("--train-every", type=int, default=8,
                   help="每 train_every 次迭代做一次梯度更新")
    p.add_argument("--updates-per", type=int, default=1, help="每次训练触发的梯度步数")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--capacity", type=int, default=0,
                   help="每 env 回放容量;0 = ceil(total_steps/n_envs)")
    p.add_argument("--model-lr", type=float, default=1e-4)
    p.add_argument("--ac-lr", type=float, default=3e-5)
    p.add_argument("--log-interval", type=int, default=50, help="按更新次数计")
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--run-dir", default="runs/crafter_dreamerv3")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    # 行缓冲:重定向到文件(nohup ... > log)时 Python stdout 默认全缓冲会让进度日志
    # 长时间不落盘、看似卡死;改行缓冲后日志实时可见(等价于 python -u)。
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 模型 ────────────────────────────────────────────────────────────────
    agent = build_dreamerv3(device=str(device), num_actions=N_ACTIONS,
                            obs_shape=(3, 64, 64), **SIZE_PRESETS[args.size])
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"DreamerV3[{args.size}] 参数量: {n_params:,}")

    wm_opt = torch.optim.Adam(agent.world_model.parameters(), lr=args.model_lr, eps=1e-8)
    actor_opt = torch.optim.Adam(agent.behavior.actor.parameters(), lr=args.ac_lr, eps=1e-5)
    value_opt = torch.optim.Adam(agent.behavior.value.parameters(), lr=args.ac_lr, eps=1e-5)

    # ── 环境与回放 ──────────────────────────────────────────────────────────
    envs = VecCrafterEnv(n_envs=args.n_envs, device=str(device), seed=args.seed)
    capacity = args.capacity or (args.total_steps // args.n_envs + args.seq_len + 1)
    replay = SequenceReplay(capacity=capacity, n_envs=args.n_envs,
                            obs_shape=(3, 64, 64), num_actions=N_ACTIONS)

    # ── 训练状态 ────────────────────────────────────────────────────────────
    obs = envs.reset()
    is_first = torch.ones(args.n_envs, device=device)
    state = None
    ep_reward = torch.zeros(args.n_envs, device=device)
    ep_ach = [set() for _ in range(args.n_envs)]
    finished_rewards, finished_ach_counts = [], []

    total_steps = 0
    n_updates = 0
    start_time = time.time()
    print(f"\nDreamerV3 on Crafter | device={device} | n_envs={args.n_envs} | "
          f"capacity={capacity} | total={args.total_steps:,}\n")

    iteration = 0
    while total_steps < args.total_steps:
        iteration += 1
        prefilling = len(replay) < args.prefill

        # ── 选动作 ──────────────────────────────────────────────────────────
        if prefilling:
            action_idx = torch.randint(0, N_ACTIONS, (args.n_envs,), device=device)
            action_onehot = F.one_hot(action_idx, N_ACTIONS).float()
            state = None
        else:
            action_idx, action_onehot, state = agent.policy(
                obs, state, is_first, training=True)

        next_obs, reward, done, infos, _ = envs.step(action_idx)
        cont = 1.0 - done
        replay.add(obs, action_onehot, reward, cont, is_first)

        # ── episode 统计 ────────────────────────────────────────────────────
        ep_reward += reward
        for i in range(args.n_envs):
            for ach in ACHIEVEMENTS:
                if infos[i].get("achievements", {}).get(ach, 0) > 0:
                    ep_ach[i].add(ach)
            if done[i] > 0:
                finished_rewards.append(ep_reward[i].item())
                finished_ach_counts.append(len(ep_ach[i]))
                ep_reward[i] = 0.0
                ep_ach[i] = set()

        obs = next_obs
        is_first = done.float()
        total_steps += args.n_envs

        # ── 训练 ────────────────────────────────────────────────────────────
        if not prefilling and replay.can_sample(args.seq_len) and \
                iteration % args.train_every == 0:
            for _ in range(args.updates_per):
                b_obs, b_act, b_rew, b_cont, b_first = replay.sample(
                    args.batch_size, args.seq_len, device)

                wm_loss, post, m = agent.world_model.loss(
                    b_obs, b_act, b_rew, b_cont, b_first)
                wm_opt.zero_grad(set_to_none=True)
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.world_model.parameters(), 1000.0)
                wm_opt.step()

                post_sg = {k: v.detach() for k, v in post.items()}
                a_loss, v_loss, bm = agent.behavior.loss(post_sg, agent.world_model)
                actor_opt.zero_grad(set_to_none=True)
                value_opt.zero_grad(set_to_none=True)
                (a_loss + v_loss).backward()
                torch.nn.utils.clip_grad_norm_(agent.behavior.actor.parameters(), 100.0)
                torch.nn.utils.clip_grad_norm_(agent.behavior.value.parameters(), 100.0)
                actor_opt.step()
                value_opt.step()
                agent.behavior.update_slow()
                n_updates += 1

                if n_updates % args.log_interval == 0:
                    sps = int(total_steps / (time.time() - start_time + 1e-6))
                    rr = finished_rewards[-100:] or [0.0]
                    aa = finished_ach_counts[-100:] or [0]
                    print(
                        f"upd={n_updates:6d} | steps={total_steps:>9,} | sps={sps:>5,} | "
                        f"ep_rew={np.mean(rr):6.3f} | ach/ep={np.mean(aa):4.2f} | "
                        f"wm={m['wm_total']:7.1f}(img={m['image']:6.1f} "
                        f"rew={m['reward']:.3f} kl_d={m['kl_dyn']:.2f} kl_r={m['kl_rep']:.2f}) | "
                        f"actor={bm['actor']:+.3f} value={bm['value']:.3f} "
                        f"ent={bm['entropy']:.3f} imR={bm['imag_reward']:+.3f}"
                    )

        # ── checkpoint ──────────────────────────────────────────────────────
        if total_steps % args.save_interval < args.n_envs and not prefilling:
            path = os.path.join(ckpt_dir, f"ckpt_{total_steps:08d}.pt")
            torch.save({"total_steps": total_steps, "model_state": agent.state_dict(),
                        "ep_rewards": finished_rewards}, path)

    final = os.path.join(args.run_dir, "final.pt")
    torch.save({"total_steps": total_steps, "model_state": agent.state_dict(),
                "ep_rewards": finished_rewards}, final)
    print(f"\n训练完成。最终模型: {final}")
    if finished_rewards:
        print(f"最近 100 ep 平均奖励: {np.mean(finished_rewards[-100:]):.4f} | "
              f"平均成就数: {np.mean(finished_ach_counts[-100:]):.2f}")


if __name__ == "__main__":
    main()
