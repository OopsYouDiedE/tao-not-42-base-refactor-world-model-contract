"""DreamerV3 Crafter 训练主程序 (train/crafter/train_dreamerv3.py)。

使用方法(从仓库根目录执行):
    python -m train.crafter.train_dreamerv3 --size small --total-steps 200000
    python -m train.crafter.train_dreamerv3 --size tiny --total-steps 4000   # 冒烟
    python -m train.crafter.train_dreamerv3 --goal --size small ...          # 文本目标条件化

对外接口:
    main() — 解析 CLI,装配 DreamerV3 + 环境 + 回放,运行"采集 ↔ 世界模型/想象更新"循环。

世界模型与想象 actor-critic 用各自优化器分开更新(actor/critic 不回传梯度到世界模型)。
模型结构走 net.dreamerv3.DreamerV3Config(--size 预设),训练旋钮走 CLI。

--goal:启用文本目标条件化(Phase 1)。把 22 个成就当文本目标,用冻结 MiniLM 编码后经
"文本点乘"条件化 actor(见 net/dreamerv3/behavior.py::GoalActorHead、train/crafter/goal.py)。
奖励保持 vanilla,目标只条件化行为。指标加:出现过的最高分数 + 仅 fresh(从零)评测的成就均值。
"""
import argparse
import glob
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
    "medium": dict(dyn_deter=384, dyn_stoch=32, dyn_discrete=32, dyn_hidden=384,
                   units=384, mlp_layers=2, enc_depths=(32, 64, 128, 256),
                   dec_depths=(256, 128, 64, 32), horizon=15),
    "default": dict(),   # 用 DreamerV3Config 全量默认(deter=512, 32×32, units=512)
}


def parse_args():
    p = argparse.ArgumentParser(description="DreamerV3 on Crafter")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size", choices=list(SIZE_PRESETS), default="small")
    p.add_argument("--total-steps", type=int, default=200_000, help="总环境交互步数")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--prefill", type=int, default=2000, help="随机策略预填步数")
    p.add_argument("--train-every", type=int, default=1,
                   help="每 train_every 次迭代做一次梯度更新")
    p.add_argument("--updates-per", type=int, default=2, help="每次训练触发的梯度步数")
    # L4 吞吐优化默认:大 batch 喂饱 GPU、短 seq 摊薄 RSSM 沿 T 的逐步开销
    # (见 knowledge/dreamer.md §2.5)。train ratio = updates_per×batch×seq/(train_every×n_envs)。
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--capacity", type=int, default=0,
                   help="每 env 回放容量;0 = ceil(total_steps/n_envs)")
    p.add_argument("--model-lr", type=float, default=1e-4)
    p.add_argument("--ac-lr", type=float, default=3e-5)
    p.add_argument("--actor-entropy", type=float, default=None,
                   help="覆盖 actor_entropy(默认用 config 的 3e-4)")
    p.add_argument("--log-interval", type=int, default=50, help="按更新次数计")
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--run-dir", default="runs/crafter_dreamerv3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="", help="检查点路径:加载 model_state + total_steps 续训")
    p.add_argument("--keep-ckpts", type=int, default=5,
                   help="只保留最近 N 个检查点(防写满磁盘);0 = 全留")
    # ── 文本目标条件化(Phase 1)─────────────────────────────────────────────
    p.add_argument("--goal", action="store_true", help="启用文本目标条件化")
    p.add_argument("--goal-encoder", default="minilm", choices=["minilm", "mock"],
                   help="目标文本编码器(minilm 冻结句向量 / mock 哈希伪嵌入)")
    p.add_argument("--n-eval-envs", type=int, default=4, help="fresh 评测并行环境数")
    p.add_argument("--eval-interval", type=int, default=0,
                   help="按更新次数;0 = 关闭 fresh 评测(从零测成就均值)")
    p.add_argument("--eval-steps", type=int, default=400, help="每次 fresh 评测每 env 的步预算")
    # ── 稀疏候选规划器 + 蒸馏(Phase 2)─────────────────────────────────────
    p.add_argument("--plan", action="store_true", help="启用稀疏候选规划器(MPC)采集/评测")
    p.add_argument("--plan-candidates", type=int, default=64, help="候选动作序列数 N")
    p.add_argument("--plan-horizon", type=int, default=8, help="候选序列长度 L")
    p.add_argument("--goal-align-coef", type=float, default=1.0, help="候选打分目标对齐权重 α")
    p.add_argument("--distill-coef", type=float, default=0.0,
                   help="规划器选中动作蒸回密集 actor 的损失权重(>0 启用)")
    # ── Go-Explore 状态缓存课程(Phase 3)──────────────────────────────────
    p.add_argument("--p-resume", type=float, default=0.0,
                   help="训练 env done 时从状态缓存空降的概率(>0 启用课程)")
    p.add_argument("--cache-cap", type=int, default=50, help="每 stage 桶缓存的快照数上限")
    return p.parse_args()


def _enable_fast_math():
    """训练侧吞吐优化(只在训练入口设全局开关,net/ 保持纯净不设)。

    - 关闭 torch.distributions 参数校验:RSSM observe 沿 T 步反复构造 OneHot 分布,
      默认的 simplex/有限性校验是纯 CPU 开销,占 observe 大头(实测 T=64 observe 363→166 ms)。
    - 开 TF32 + cudnn.benchmark:加速 flop-bound 的卷积编/解码器(固定形状下 benchmark 选最优 kernel)。
    """
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _unlocked_set(info):
    """从 Crafter info 取已解锁成就名集合。"""
    ach = info.get("achievements", {})
    return {a for a in ACHIEVEMENTS if ach.get(a, 0) > 0}


@torch.no_grad()
def evaluate_fresh(agent, goalspace, n_envs, eval_steps, device, use_goal, seed,
                   planner=None):
    """从零(fresh reset)跑 eval_steps/env 步,返回(成就均值, 最高 episode 奖励)。

    只用于"采集/评测"度量真实从初始可达成就数,不写回放、不参与训练、不从课程缓存恢复。
    planner 不为 None 时用规划器贪心行动(反映真实采集策略),否则用密集 actor 贪心。
    """
    envs = VecCrafterEnv(n_envs=n_envs, device=str(device), seed=seed)
    obs = envs.reset()
    is_first = torch.ones(n_envs, device=device)
    state = None
    ep_ach = [set() for _ in range(n_envs)]
    ep_rew = torch.zeros(n_envs, device=device)
    fin_ach, fin_rew = [], []
    if use_goal:
        goal_ids = torch.tensor(
            [goalspace.next_frontier(set()) for _ in range(n_envs)], device=device)

    seen = set()                      # 本次评测从零解锁过的所有成就(覆盖)
    for _ in range(eval_steps):
        goal_emb = goalspace.embedding(goal_ids) if use_goal else None
        if planner is not None:
            latent = agent.encode_latent(obs, state, is_first)
            a_idx, a_oh = planner.act(latent, goal_emb, training=False)
            state = (latent, a_oh)
        else:
            a_idx, _, state = agent.policy(obs, state, is_first, training=False, goal=goal_emb)
        obs, reward, done, infos, _ = envs.step(a_idx)
        ep_rew += reward
        for i in range(n_envs):
            ep_ach[i] |= _unlocked_set(infos[i])
            seen |= ep_ach[i]
            if done[i] > 0:
                fin_ach.append(len(ep_ach[i]))
                fin_rew.append(ep_rew[i].item())
                ep_ach[i] = set()
                ep_rew[i] = 0.0
                if use_goal:
                    goal_ids[i] = goalspace.next_frontier(set())
            elif use_goal:
                goal_ids[i] = goalspace.next_frontier(ep_ach[i])
        is_first = done.float()

    ep_counts = fin_ach or [len(s) for s in ep_ach]
    mean_ach = float(np.mean(ep_counts))
    max_ach = int(max(ep_counts))                  # 单局最高解锁成就数(从零)
    max_rew = max(fin_rew) if fin_rew else float(ep_rew.max().item())
    return mean_ach, max_rew, max_ach, seen


def main():
    args = parse_args()
    # 行缓冲:重定向到文件(nohup ... > log)时 Python stdout 默认全缓冲会让进度日志
    # 长时间不落盘、看似卡死;改行缓冲后日志实时可见(等价于 python -u)。
    sys.stdout.reconfigure(line_buffering=True)
    _enable_fast_math()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 目标空间(可选)──────────────────────────────────────────────────────
    goalspace = None
    goal_overrides = {}
    if args.goal:
        from train.crafter.goal import GoalSpace
        goalspace = GoalSpace(encoder_kind=args.goal_encoder, device=str(device))
        goal_overrides = dict(use_goal=True, goal_text_dim=goalspace.text_dim)
        print(f"目标条件化已启用 | encoder={args.goal_encoder} | "
              f"{goalspace.n} 个成就目标 | text_dim={goalspace.text_dim}")

    # ── 模型 ────────────────────────────────────────────────────────────────
    if args.actor_entropy is not None:
        goal_overrides["actor_entropy"] = args.actor_entropy
    agent = build_dreamerv3(device=str(device), num_actions=N_ACTIONS,
                            obs_shape=(3, 64, 64),
                            **SIZE_PRESETS[args.size], **goal_overrides)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"DreamerV3[{args.size}] 参数量: {n_params:,}")

    wm_opt = torch.optim.Adam(agent.world_model.parameters(), lr=args.model_lr, eps=1e-8)
    actor_opt = torch.optim.Adam(agent.behavior.actor.parameters(), lr=args.ac_lr, eps=1e-5)
    value_opt = torch.optim.Adam(agent.behavior.value.parameters(), lr=args.ac_lr, eps=1e-5)

    # ── 稀疏候选规划器(可选)──────────────────────────────────────────────
    planner = None
    if args.plan:
        from net.dreamerv3.planner import Planner
        planner = Planner(agent, args.plan_candidates, args.plan_horizon,
                          agent.cfg.discount, args.goal_align_coef, args.goal)
        print(f"稀疏规划器已启用 | N={args.plan_candidates} L={args.plan_horizon} "
              f"α={args.goal_align_coef} distill={args.distill_coef}")

    # ── 续训(可选):加载 model_state + total_steps ──────────────────────
    resume_total_steps, resume_cov = 0, {"fresh": set(), "train": set()}
    resume_max = {"fresh": 0, "train": 0, "score": 0.0}
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["model_state"])
        resume_total_steps = int(ckpt.get("total_steps", 0))
        resume_cov["fresh"] = set(ckpt.get("fresh_cov", []))
        resume_cov["train"] = set(ckpt.get("train_cov", []))
        resume_max["fresh"] = int(ckpt.get("fresh_max_ach", 0))
        resume_max["train"] = int(ckpt.get("train_max_ach", 0))
        resume_max["score"] = float(ckpt.get("best_score", 0.0))
        args.prefill = 0   # 已有训练好的策略,直接用策略采集(靠 replay.can_sample 等填够)
        print(f"续训:从 {args.resume} 加载 | 起始 total_steps={resume_total_steps:,} | "
              f"已知覆盖 fresh={len(resume_cov['fresh'])}/22 train={len(resume_cov['train'])}/22")

    # ── 状态缓存课程(可选)+ 环境与回放 ──────────────────────────────────
    state_cache = None
    if args.p_resume > 0:
        from train.crafter.state_cache import StateCache
        state_cache = StateCache(cap_per_stage=args.cache_cap, seed=args.seed)
        print(f"Go-Explore 课程已启用 | p_resume={args.p_resume} | cache_cap={args.cache_cap}")
    envs = VecCrafterEnv(n_envs=args.n_envs, device=str(device), seed=args.seed,
                         state_cache=state_cache, p_resume=args.p_resume)
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
    best_score = resume_max["score"]      # 出现过的最高 episode 奖励
    fresh_ach_mean, fresh_max = 0.0, 0.0  # 最近一次 fresh 评测结果
    # 成就覆盖/最高(单调累计,跨续训保留):fresh=从零评测,train=训练侧(含课程空降)
    fresh_cov, train_cov = resume_cov["fresh"], resume_cov["train"]
    fresh_max_ach, train_max_ach = resume_max["fresh"], resume_max["train"]
    if args.goal:
        goal_ids = torch.tensor(
            [goalspace.next_frontier(set()) for _ in range(args.n_envs)], device=device)
    else:
        goal_ids = None

    total_steps = resume_total_steps
    n_updates = 0
    start_time = time.time()
    print(f"\nDreamerV3 on Crafter | device={device} | n_envs={args.n_envs} | "
          f"capacity={capacity} | total={args.total_steps:,} | goal={args.goal}\n")

    iteration = 0
    while total_steps < args.total_steps:
        iteration += 1
        # prefill 以环境步计(total_steps),而非回放行数(len(replay) 每迭代 +1,= total_steps/n_envs);
        # 二者差 n_envs 倍,若用行数会让随机预填超采样 n_envs 倍。
        prefilling = total_steps < args.prefill

        # ── 选动作 ──────────────────────────────────────────────────────────
        if prefilling:
            action_idx = torch.randint(0, N_ACTIONS, (args.n_envs,), device=device)
            action_onehot = F.one_hot(action_idx, N_ACTIONS).float()
            state = None
        else:
            goal_emb = goalspace.embedding(goal_ids) if args.goal else None
            if args.plan:
                latent = agent.encode_latent(obs, state, is_first)
                action_idx, action_onehot = planner.act(latent, goal_emb, training=True)
                state = (latent, action_onehot)
            else:
                action_idx, action_onehot, state = agent.policy(
                    obs, state, is_first, training=True, goal=goal_emb)

        next_obs, reward, done, infos, _ = envs.step(action_idx)
        cont = 1.0 - done
        replay.add(obs, action_onehot, reward, cont, is_first, goal_ids)

        # ── episode 统计 + 目标推进 ─────────────────────────────────────────
        ep_reward += reward
        for i in range(args.n_envs):
            ep_ach[i] |= _unlocked_set(infos[i])
            if done[i] > 0:
                finished_rewards.append(ep_reward[i].item())
                finished_ach_counts.append(len(ep_ach[i]))   # 含课程起点,仅诊断
                best_score = max(best_score, ep_reward[i].item())
                train_cov |= ep_ach[i]                       # 训练侧成就覆盖(含课程到达)
                train_max_ach = max(train_max_ach, len(ep_ach[i]))
                ep_reward[i] = 0.0
                # 若是从课程缓存空降复活,继承该存档点的已解锁集(目标/前沿据此续接)。
                resumed = infos[i].get("resumed_unlocked")
                ep_ach[i] = set(resumed) if resumed is not None else set()
                if args.goal:
                    goal_ids[i] = goalspace.next_frontier(ep_ach[i])
            elif args.goal:                          # 前沿移动则重采下一目标
                goal_ids[i] = goalspace.next_frontier(ep_ach[i])

        obs = next_obs
        is_first = done.float()
        total_steps += args.n_envs

        # ── 训练 ────────────────────────────────────────────────────────────
        if not prefilling and replay.can_sample(args.seq_len) and \
                iteration % args.train_every == 0:
            for _ in range(args.updates_per):
                b_obs, b_act, b_rew, b_cont, b_first, b_goal = replay.sample(
                    args.batch_size, args.seq_len, device)

                wm_loss, post, m = agent.world_model.loss(
                    b_obs, b_act, b_rew, b_cont, b_first)
                wm_opt.zero_grad(set_to_none=True)
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.world_model.parameters(), 1000.0)
                wm_opt.step()

                b_goal_emb = goalspace.embedding(b_goal) if args.goal else None
                post_sg = {k: v.detach() for k, v in post.items()}
                a_loss, v_loss, bm = agent.behavior.loss(
                    post_sg, agent.world_model, goal=b_goal_emb)
                total_ac = a_loss + v_loss
                # 蒸馏(密集头带稀疏头):在真实回放态上把"已执行(规划器选中)动作"
                # 监督回密集 actor(feat detach ⇒ 梯度只到 actor,不回世界模型)。
                if args.distill_coef > 0:
                    feat_d = agent.world_model.dynamics.get_feat(post_sg)
                    ddist = agent.behavior.actor_dist(feat_d, b_goal_emb)
                    distill = -ddist.log_prob(b_act).mean()
                    total_ac = total_ac + args.distill_coef * distill
                    bm["distill"] = distill.item()
                actor_opt.zero_grad(set_to_none=True)
                value_opt.zero_grad(set_to_none=True)
                total_ac.backward()
                torch.nn.utils.clip_grad_norm_(agent.behavior.actor.parameters(), 100.0)
                torch.nn.utils.clip_grad_norm_(agent.behavior.value.parameters(), 100.0)
                actor_opt.step()
                value_opt.step()
                agent.behavior.update_slow()
                n_updates += 1

                # ── fresh 评测(从零测成就均值)──────────────────────────────
                if args.eval_interval and n_updates % args.eval_interval == 0:
                    fresh_ach_mean, fresh_max, fmax_ach, fseen = evaluate_fresh(
                        agent, goalspace, args.n_eval_envs, args.eval_steps,
                        device, args.goal, args.seed + 12345, planner=planner)
                    best_score = max(best_score, fresh_max)
                    fresh_cov |= fseen
                    fresh_max_ach = max(fresh_max_ach, fmax_ach)

                if n_updates % args.log_interval == 0:
                    sps = int(total_steps / (time.time() - start_time + 1e-6))
                    rr = finished_rewards[-100:] or [0.0]
                    aa = finished_ach_counts[-100:] or [0]
                    print(
                        f"upd={n_updates:6d} | steps={total_steps:>9,} | sps={sps:>5,} | "
                        f"ep_rew={np.mean(rr):6.3f} | ach/ep={np.mean(aa):4.2f} | "
                        f"max_score={best_score:5.2f} | fresh_ach={fresh_ach_mean:4.2f} | "
                        f"fmaxA={fresh_max_ach} fcov={len(fresh_cov)}/22 "
                        f"tmaxA={train_max_ach} tcov={len(train_cov)}/22 | "
                        f"wm={m['wm_total']:7.1f}(img={m['image']:6.1f} "
                        f"rew={m['reward']:.3f} kl_d={m['kl_dyn']:.2f} kl_r={m['kl_rep']:.2f}) | "
                        f"actor={bm['actor']:+.3f} value={bm['value']:.3f} "
                        f"ent={bm['entropy']:.3f} imR={bm['imag_reward']:+.3f} "
                        f"dist={bm.get('distill', 0.0):.3f}"
                    )

        # ── checkpoint ──────────────────────────────────────────────────────
        if total_steps % args.save_interval < args.n_envs and not prefilling:
            payload = {"total_steps": total_steps, "model_state": agent.state_dict(),
                       "ep_rewards": finished_rewards, "best_score": best_score,
                       "fresh_cov": sorted(fresh_cov), "train_cov": sorted(train_cov),
                       "fresh_max_ach": fresh_max_ach, "train_max_ach": train_max_ach}
            torch.save(payload, os.path.join(ckpt_dir, f"ckpt_{total_steps:08d}.pt"))
            print(f"[ckpt {total_steps:,}] fresh_cov({len(fresh_cov)}/22)={sorted(fresh_cov)} | "
                  f"train_cov({len(train_cov)}/22)={sorted(train_cov)}")
            if args.keep_ckpts > 0:   # 只留最近 N 个,防写满磁盘
                for old in sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))[:-args.keep_ckpts]:
                    os.remove(old)

    final = os.path.join(args.run_dir, "final.pt")
    torch.save({"total_steps": total_steps, "model_state": agent.state_dict(),
                "ep_rewards": finished_rewards, "best_score": best_score,
                "fresh_cov": sorted(fresh_cov), "train_cov": sorted(train_cov),
                "fresh_max_ach": fresh_max_ach, "train_max_ach": train_max_ach}, final)
    print(f"\n训练完成。最终模型: {final}")
    print(f"从零覆盖 fresh_cov({len(fresh_cov)}/22)={sorted(fresh_cov)} | 单局最高 fmaxA={fresh_max_ach}")
    print(f"训练侧覆盖 train_cov({len(train_cov)}/22)={sorted(train_cov)} | 单局最高 tmaxA={train_max_ach}")
    if finished_rewards:
        print(f"最近 100 ep 平均奖励: {np.mean(finished_rewards[-100:]):.4f} | "
              f"平均成就数: {np.mean(finished_ach_counts[-100:]):.2f} | 最高分数: {best_score:.2f}")


if __name__ == "__main__":
    main()
