"""吞吐瀑布 profiler：把端到端 sps 拆成四个桶。

桶：
  ① 纯环境步进 (env.step 扣掉重置)
  ② 编码器前向(收集阶段)
  ③ 地形检测重置 (CraftgroundEnvWithTerrainCheck.reset，含重试风暴)
  ④ PPO 更新 (envs 空转的那段)

配置与正式 run 对齐：4环境 / n_steps=256 / ppo_batch_size=64 / 编码器解冻(稳态)。
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from train.craftground import env_interface as EI
from train.craftground.env_interface import CraftgroundVecEnvWithInterface, CraftgroundEnvWithTerrainCheck
from net.encoders.yolo_backbone_encoder import YoloBackboneEncoder
from train.craftground.train_ppo_ad import PPOActorCritic, AchievementHead, compute_gae_advantages
from train.craftground.achievements import ALL_ACHIEVEMENTS

DEVICE = "cuda"
N_ENVS = 4
N_STEPS = 256
BATCH = 64
PPO_EPOCHS = 4
PPO_CLIP = 0.2
PROF_ROLLOUTS = 3

# ── 地形检测重置计时(monkeypatch)────────────────────────────────
_reset_acc = {"t": 0.0, "n": 0}
_orig_reset = CraftgroundEnvWithTerrainCheck.reset
def _timed_reset(self):
    t0 = time.time()
    r = _orig_reset(self)
    _reset_acc["t"] += time.time() - t0
    _reset_acc["n"] += 1
    return r
CraftgroundEnvWithTerrainCheck.reset = _timed_reset


def sync():
    torch.cuda.synchronize()


def main():
    print(f"[prof] 构建 {N_ENVS} 环境 + 编码器(解冻) + AC ...", flush=True)
    env = CraftgroundVecEnvWithInterface(nproc=N_ENVS, device=DEVICE,
                                         max_episode_steps=1000, use_terrain_check=True, seed=0)
    encoder = YoloBackboneEncoder(output_dim=512, pretrained=True, device=DEVICE)
    # 解冻 backbone，匹配 run 大部分时间的稳态
    for p in encoder.backbone.parameters():
        p.requires_grad = True
    actor_critic = PPOActorCritic(512, 27, 256).to(DEVICE)
    ach_head = AchievementHead(512, len(ALL_ACHIEVEMENTS), 256).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(actor_critic.parameters()) + list(ach_head.parameters()), lr=3e-4)

    obs = env.reset()
    _reset_acc["t"] = 0.0; _reset_acc["n"] = 0  # 清掉启动 reset

    buckets = {"env": 0.0, "enc": 0.0, "update": 0.0, "reset": 0.0}
    t_wall0 = time.time()
    total_env_steps = 0

    for r in range(PROF_ROLLOUTS):
        obs_list, action_list, reward_list, done_list = [], [], [], []
        logprob_list, value_list, ach_r_list, ach_list = [], [], [], []
        reset_before = _reset_acc["t"]

        for step in range(N_STEPS):
            sync(); t0 = time.time()
            with torch.no_grad():
                feats = encoder(obs)
                logits, values = actor_critic(feats)
                probs = torch.softmax(logits, -1)
                actions = torch.multinomial(probs, 1).squeeze(1)
                logp = torch.log_softmax(logits, -1).gather(1, actions.unsqueeze(1)).squeeze(1)
            sync(); buckets["enc"] += time.time() - t0

            t0 = time.time()
            obs, rewards, dones, infos = env.step(actions)
            buckets["env"] += time.time() - t0  # 含 in-loop 重置

            obs_list.append(obs.clone().detach()); action_list.append(actions)
            reward_list.append((rewards.squeeze(1) + infos["achievement_rewards"].squeeze(1)))
            done_list.append(dones.squeeze(1)); logprob_list.append(logp)
            value_list.append(values.squeeze(1)); ach_list.append(infos["achievements"].detach())
            total_env_steps += N_ENVS

        buckets["reset"] += _reset_acc["t"] - reset_before  # 本轮内重置耗时

        # ── PPO 更新 ──
        sync(); t0 = time.time()
        with torch.no_grad():
            last_v = actor_critic(encoder(obs))[1].squeeze(1)
        adv, ret = compute_gae_advantages(reward_list, value_list + [last_v], done_list, 0.99, 0.95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ds = TensorDataset(torch.cat(obs_list), torch.cat(action_list).unsqueeze(1),
                           torch.cat(logprob_list), ret, adv, torch.cat(ach_list))
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
        for _ in range(PPO_EPOCHS):
            for b_obs, b_act, b_lp, b_ret, b_adv, b_ach in dl:
                f = encoder(b_obs)
                lg, v = actor_critic(f)
                lpn = torch.log_softmax(lg, -1).gather(1, b_act).squeeze(1)
                ratio = torch.exp(lpn - b_lp)
                pl = -torch.min(ratio * b_adv, torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP) * b_adv).mean()
                vl = 0.5 * ((v.squeeze(1) - b_ret) ** 2).mean()
                pr = torch.softmax(lg, -1)
                el = (pr * torch.log_softmax(lg, -1)).sum(-1).mean()
                al = nn.BCEWithLogitsLoss()(ach_head(f), b_ach.float())
                loss = pl + 0.5 * vl + 0.01 * el + al
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(actor_critic.parameters()) + list(ach_head.parameters()), 1.0)
                optimizer.step()
        sync(); buckets["update"] += time.time() - t0
        print(f"[prof] rollout {r+1}/{PROF_ROLLOUTS} 完成", flush=True)

    wall = time.time() - t_wall0
    pure_env = buckets["env"] - buckets["reset"]
    sps = total_env_steps / wall
    print("\n" + "=" * 60)
    print(f"端到端: {total_env_steps} env步 / {wall:.1f}s = {sps:.1f} sps")
    print("=" * 60)
    parts = [("① 纯环境步进", pure_env), ("③ 地形检测重置", buckets["reset"]),
             ("② 编码器前向(收集)", buckets["enc"]), ("④ PPO 更新(env空转)", buckets["update"])]
    accounted = sum(p[1] for p in parts)
    for name, t in parts:
        print(f"  {name:20s}: {t:6.1f}s  ({100*t/wall:4.1f}%)")
    print(f"  {'其余(GAE/堆叠/采样等)':20s}: {wall-accounted:6.1f}s  ({100*(wall-accounted)/wall:4.1f}%)")
    print(f"\n  地形重置次数: {_reset_acc['n']}  平均每次 {buckets['reset']/max(1,_reset_acc['n']):.2f}s")
    env.close()


if __name__ == "__main__":
    main()
