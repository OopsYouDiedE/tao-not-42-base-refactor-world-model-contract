"""YOLO-World-Dreamer Crafter 训练主程序 (train/crafter/train_yoloworld.py)。

使用方法(从仓库根目录执行):
    python -m train.crafter.train_yoloworld --size crafter --total-steps 1000000 --device cuda
    python -m train.crafter.train_yoloworld --size tiny --total-steps 4000 --device cpu  # 冒烟

对外接口:
    main() — 解析 CLI,装配 YoloWorld + 环境 + 含目标回放,运行
             "采集(逐 env 不同语言目标)↔ 世界模型线 / 双头行为线更新"循环。

两条线各自优化器分开更新(行为线不回传梯度到世界模型)。结构走 YoloWorldConfig(--size 预设),
训练旋钮走 CLI。任务语言条件见 train/crafter/crafter_tasks.py,设计见 knowledge/yoloworld.md。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from net.yoloworld import build_yoloworld
from train.crafter.env import VecCrafterEnv
from train.crafter.ad_buffer import ACHIEVEMENTS, N_ACHIEVEMENTS
from train.crafter.yoloworld_buffer import GoalSequenceReplay
from train.crafter.crafter_tasks import build_ach_embed, GoalSampler
from train.minecraft.task_text import TaskTextEncoder

N_ACTIONS = 17

# --size 预设:结构规模(覆盖 YoloWorldConfig 默认)。
# crafter = 能学会 Crafter 的 DreamerV3 已验证档(~1.7e7 参数);small/tiny 供 CPU 冒烟。
SIZE_PRESETS = {
    "tiny": dict(dyn_deter=128, dyn_stoch=8, dyn_discrete=8, dyn_hidden=128,
                 units=128, mlp_layers=1, enc_depths=(16, 32, 64, 128),
                 dec_depths=(128, 64, 32, 16), n_candidates=64, plan_horizon=6,
                 query_dim=32, head_hidden=128, n_rollout=8, n_explore=4),
    "small": dict(dyn_deter=256, dyn_stoch=24, dyn_discrete=24, dyn_hidden=256,
                  units=256, mlp_layers=2, enc_depths=(24, 48, 96, 192),
                  dec_depths=(192, 96, 48, 24), n_candidates=128, plan_horizon=12,
                  query_dim=48, head_hidden=192, n_rollout=24, n_explore=8),
    "crafter": dict(),   # YoloWorldConfig 全量默认(deter=512, 32×32, units=512, K=256, H=16)
}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-World-Dreamer on Crafter")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size", choices=list(SIZE_PRESETS), default="small")
    p.add_argument("--total-steps", type=int, default=1_000_000, help="总环境交互步数")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--prefill", type=int, default=2000, help="随机策略预填步数")
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--updates-per", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=32, help="世界模型训练序列长(16/32)")
    p.add_argument("--n-start", type=int, default=0,
                   help="行为线每次子采样的起点数;0 = 用全部 B·L(控 CPU 算力)")
    p.add_argument("--her-ratio", type=float, default=0.5, help="HER 事后重标窗口比例")
    p.add_argument("--capacity", type=int, default=0,
                   help="每 env 回放容量;0 = ceil(total_steps/n_envs)")
    p.add_argument("--model-lr", type=float, default=1e-4)
    p.add_argument("--beh-lr", type=float, default=3e-5)
    p.add_argument("--actor-entropy", type=float, default=None,
                   help="覆盖计划熵正则 λ_H(策略钉在最大熵=随机时调小;坍缩时调大);None=用预设")
    p.add_argument("--task-encoder", choices=["minilm", "mock"], default="minilm")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=20000)
    p.add_argument("--run-dir", default="runs/crafter_yoloworld")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _enable_fast_math(device):
    """吞吐优化(只在训练入口设全局开关,net/ 保持纯净)。

    - 关 torch.distributions 参数校验(RSSM/候选采样沿步反复构造分布,校验是纯开销)。
    - CPU:set_num_threads = 物理核数,吃满所有核(矢量化热路径 + 大批 rollout)。
    - GPU:TF32 + cudnn.benchmark,加速 flop-bound 卷积编/解码器与大 batch matmul。
    """
    torch.distributions.Distribution.set_default_validate_args(False)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _ach_multihot(infos, device):
    """infos → [n_envs, U] float multi-hot(该步累计已解锁成就)。"""
    out = torch.zeros(len(infos), N_ACHIEVEMENTS, device=device)
    for i, info in enumerate(infos):
        a = info.get("achievements", {})
        for j, name in enumerate(ACHIEVEMENTS):
            if a.get(name, 0) > 0:
                out[i, j] = 1.0
    return out


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device(args.device)
    _enable_fast_math(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 模型 ────────────────────────────────────────────────────────────────
    overrides = dict(SIZE_PRESETS[args.size])
    if args.actor_entropy is not None:
        overrides["actor_entropy"] = args.actor_entropy
    agent = build_yoloworld(device=str(device), num_actions=N_ACTIONS,
                            obs_shape=(3, 64, 64), n_achievements=N_ACHIEVEMENTS,
                            n_start=args.n_start, **overrides)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"YoloWorld[{args.size}] 参数量: {n_params:,}")

    # ── 任务语言条件(冻结句编码器 → 成就嵌入矩阵 E)──────────────────────
    encoder = TaskTextEncoder(kind=args.task_encoder, device="cpu")
    E = build_ach_embed(encoder, device=str(device))             # [U, d_g]
    agent.set_ach_embed(E)
    sampler = GoalSampler(E, n_envs=args.n_envs, device=str(device), seed=args.seed)

    wm_opt = torch.optim.Adam(agent.world_model.parameters(), lr=args.model_lr, eps=1e-8)
    beh_params = list(agent.proposal.parameters()) + list(agent.behavior.critic.parameters())
    beh_opt = torch.optim.Adam(beh_params, lr=args.beh_lr, eps=1e-5)

    # ── 环境与回放 ──────────────────────────────────────────────────────────
    envs = VecCrafterEnv(n_envs=args.n_envs, device=str(device), seed=args.seed)
    capacity = args.capacity or (args.total_steps // args.n_envs + args.seq_len + 1)
    replay = GoalSequenceReplay(capacity=capacity, n_envs=args.n_envs,
                                obs_shape=(3, 64, 64), num_actions=N_ACTIONS,
                                n_achievements=N_ACHIEVEMENTS)

    # ── 训练状态 ────────────────────────────────────────────────────────────
    obs = envs.reset()
    is_first = torch.ones(args.n_envs, device=device)
    state = None
    ep_reward = torch.zeros(args.n_envs, device=device)
    ep_ach = [set() for _ in range(args.n_envs)]
    finished_rewards, finished_ach_counts = [], []

    total_steps, n_updates = 0, 0
    start_time = time.time()
    print(f"\nYoloWorld on Crafter | device={device} | n_envs={args.n_envs} | "
          f"capacity={capacity} | total={args.total_steps:,} | her={args.her_ratio}\n")

    iteration = 0
    while total_steps < args.total_steps:
        iteration += 1
        prefilling = total_steps < args.prefill
        task_emb = sampler.task_emb()                            # [n_envs, d_g]

        # ── 选动作 ──────────────────────────────────────────────────────────
        if prefilling:
            action_idx = torch.randint(0, N_ACTIONS, (args.n_envs,), device=device)
            action_onehot = torch.nn.functional.one_hot(action_idx, N_ACTIONS).float()
            state = None
        else:
            action_idx, action_onehot, state = agent.policy(
                obs, state, is_first, task_emb, training=True)

        next_obs, reward, done, infos, _ = envs.step(action_idx)
        cont = 1.0 - done
        ach_mh = _ach_multihot(infos, device)
        replay.add(obs, action_onehot, reward, cont, ach_mh,
                   sampler.goal_idx(), is_first)

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
        sampler.resample(done)                                   # done 的 env 换语言目标

        obs = next_obs
        is_first = done.float()
        total_steps += args.n_envs

        # ── 训练(两条线)──────────────────────────────────────────────────
        if not prefilling and replay.can_sample(args.seq_len) and \
                iteration % args.train_every == 0:
            for _ in range(args.updates_per):
                b_obs, b_act, b_rew, b_cont, b_ach, b_goal, b_first = replay.sample(
                    args.batch_size, args.seq_len, device, her_ratio=args.her_ratio)

                # 世界模型线
                wm_loss, post, m = agent.world_model.loss(
                    b_obs, b_act, b_rew, b_cont, b_ach, b_first)
                wm_opt.zero_grad(set_to_none=True)
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.world_model.parameters(), 1000.0)
                wm_opt.step()

                # 双头行为线(对 WM 特征 stop-grad)
                flat = lambda x: x.reshape(-1, *x.shape[2:])
                start = {k: v.detach() for k, v in post.items()}
                start = {k: flat(v) for k, v in start.items()}
                te = E[flat(b_goal)]                              # [B·L, d_g]
                Ns = start["deter"].shape[0]
                if args.n_start and args.n_start < Ns:
                    sub = torch.randperm(Ns, device=device)[:args.n_start]
                    start = {k: v[sub] for k, v in start.items()}
                    te = te[sub]
                b_loss, bm = agent.behavior.loss(start, te, agent.proposal,
                                                 agent.world_model)
                beh_opt.zero_grad(set_to_none=True)
                b_loss.backward()
                torch.nn.utils.clip_grad_norm_(beh_params, 100.0)
                beh_opt.step()
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
                        f"rew={m['reward']:.2f} ach={m['ach']:.3f} kl_d={m['kl_dyn']:.2f}) | "
                        f"actor={bm['actor']:+.3f} cls={bm['cls']:.3f} algn={bm['align']:.3f} "
                        f"val={bm['value']:.2f} ent={bm['entropy']:.2f} Rbest={bm['ret_best']:+.3f}"
                    )

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
