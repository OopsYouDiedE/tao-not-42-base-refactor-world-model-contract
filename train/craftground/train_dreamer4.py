#!/usr/bin/env python3
"""CraftGround 在线 Dreamer4 世界模型训练(交互流上的世界模型学习)。

在线阶段:随机探索策略在 Minecraft 1.21(CraftGround)里采集交互流,边采集边训练
net/dreamer4.WorldModel——tokenizer 重建 + shortcut forcing 流匹配 + reward/cont 头
(在线数据有 RewardShaper 的稠密内在奖励与 episode 终止,离线 VPT 没有)。
可用 --init 从离线 VPT 预训练 checkpoint 热启动(动作接口 22 维连续 → 27 维
one-hot 不同,action_proj 与 reward/cont 头重新学,其余权重迁移)。

评估(held-out 环境):最后一个环境的数据只进评估、不进训练,
  - psnr_gen / psnr_recon / psnr_persist(同离线口径)
  - reward NLL 与 |reward 预测 - 真值| MAE、cont 准确率

使用方法(需 DISPLAY 指向可渲染的 X,见 scripts/gpu_run.sh):
    python -m train.craftground.train_dreamer4 --n-envs 3 --total-env-steps 24000 \
        --init runs/minecraft_d4_offline/best.pt
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from net.dreamer4 import Dreamer4Config, WorldModel
from train.craftground.env_interface import CraftgroundVecEnvWithInterface
from train.craftground.env import DISCRETE_TO_V2


def parse_args():
    p = argparse.ArgumentParser(description="CraftGround 在线 Dreamer4 世界模型训练")
    p.add_argument("--n-envs", type=int, default=3,
                   help="并行环境数;最后一个环境的数据只用于评估(held-out)")
    p.add_argument("--total-env-steps", type=int, default=24000)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--token-dim", type=int, default=256)
    p.add_argument("--dyn-layers", type=int, default=4)
    p.add_argument("--dyn-heads", type=int, default=8)
    p.add_argument("--collect-per-iter", type=int, default=64,
                   help="每轮迭代的向量步数(env 步 = 该值 × n_envs)")
    p.add_argument("--updates-per-iter", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup-env-steps", type=int, default=2000,
                   help="回放攒到该 env 步数后才开始更新")
    p.add_argument("--enc-base", type=int, default=32,
                   help="tokenizer 编码器基础通道(各级 = b,2b,4b,8b;解码器倒序)")
    p.add_argument("--shortcut-hidden", type=int, default=512)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16",
                   help="混合精度(评估始终 fp32;fp16 走 GradScaler)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sc-weight", type=float, default=1.0)
    p.add_argument("--d-min", type=float, default=0.125)
    p.add_argument("--gen-steps", type=int, default=4)
    p.add_argument("--init", default=None,
                   help="离线 VPT 预训练 checkpoint(train/minecraft/train_dreamer4 的 best.pt);"
                        "动作维不同,action_proj/reward/cont 重新初始化")
    p.add_argument("--eval-interval", type=int, default=10, help="每多少轮迭代评估一次")
    p.add_argument("--n-eval-batches", type=int, default=4)
    p.add_argument("--run-dir", default="runs/craftground_d4_online")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class StreamReplay:
    """逐环境的顺序交互流回放(uint8 CPU 存储,采样连续窗口转 GPU)。

    每个环境一条流:obs [N,3,H,W] uint8 / action idx / reward / done。
    采样窗口不跨 done(世界重置处动力学不连续;done 稀疏,损失可忽略)。
    """

    def __init__(self, n_envs, capacity, img_size):
        self.n = n_envs
        self.cap = capacity
        self.obs = [torch.zeros(capacity, 3, img_size, img_size, dtype=torch.uint8)
                    for _ in range(n_envs)]
        self.act = [torch.zeros(capacity, dtype=torch.long) for _ in range(n_envs)]
        self.rew = [torch.zeros(capacity) for _ in range(n_envs)]
        self.done = [torch.zeros(capacity) for _ in range(n_envs)]
        self.len = [0] * n_envs

    def add(self, i, obs, act, rew, done):
        """obs [3,H,W] uint8(转移后的观测,与 act/rew/done 同步 append)。"""
        if self.len[i] >= self.cap:
            return
        j = self.len[i]
        self.obs[i][j] = obs
        self.act[i][j] = act
        self.rew[i][j] = rew
        self.done[i][j] = done
        self.len[i] += 1

    def sample(self, env_ids, batch, seq_len, num_actions, device, rng):
        """从指定环境集合采样 [B,T] 窗口。

        Returns:
            img [B,T,3,H,W] float01 | act_onehot [B,T,A] | rew [B,T-1] | cont [B,T-1]
            或 None(可采数据不足)。
        """
        imgs, acts, rews, conts = [], [], [], []
        tries = 0
        while len(imgs) < batch and tries < batch * 20:
            tries += 1
            i = env_ids[rng.integers(len(env_ids))]
            if self.len[i] < seq_len + 1:
                continue
            s = int(rng.integers(0, self.len[i] - seq_len))
            dn = self.done[i][s: s + seq_len - 1]
            if dn.any():                       # 不跨 episode 边界
                continue
            imgs.append(self.obs[i][s: s + seq_len])
            acts.append(self.act[i][s: s + seq_len])
            rews.append(self.rew[i][s: s + seq_len - 1])
            conts.append(1.0 - self.done[i][s: s + seq_len - 1])
        if len(imgs) < max(2, batch // 2):
            return None
        img = torch.stack(imgs).to(device).float() / 255.0
        act = F.one_hot(torch.stack(acts), num_actions).float().to(device)
        return img, act, torch.stack(rews).to(device), torch.stack(conts).to(device)


@torch.no_grad()
def evaluate(wm, replay, eval_ids, args, device, rng):
    """held-out 环境:生成/重建/持续性 PSNR + reward/cont 头质量。"""
    wm.eval()
    agg, n = {}, 0
    for _ in range(args.n_eval_batches):
        s = replay.sample(eval_ids, args.batch_size, args.seq_len,
                          len(DISCRETE_TO_V2), device, rng)
        if s is None:
            break
        img, act, rew, cont = s
        m = wm.eval_next_frame(img, act, gen_steps=args.gen_steps)
        b, t1 = rew.shape
        tokens, _ = wm.tokenizer.encode(img)
        ctx = wm.dynamics(tokens[:, :-1], act[:, :-1])
        feat = ctx.reshape(b, t1, -1)
        rd = wm.reward_dist(feat)
        m["reward_nll"] = float(-rd.log_prob(rew.unsqueeze(-1)).mean())
        m["reward_mae"] = float((rd.mode().squeeze(-1) - rew).abs().mean())
        m["cont_acc"] = float(((wm.cont_dist(feat).mean.squeeze(-1) > 0.5).float()
                               == cont).float().mean())
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    wm.train()
    return {k: v / n for k, v in agg.items()} if n else None


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_actions = len(DISCRETE_TO_V2)
    print("=" * 78, flush=True)
    print("🌐 CraftGround 在线 Dreamer4 世界模型(随机探索采集)")
    print(f"   n_envs={args.n_envs}(env {args.n_envs-1} held-out) "
          f"| 目标 {args.total_env_steps:,} env 步 | init={args.init}")
    print("=" * 78, flush=True)

    env = CraftgroundVecEnvWithInterface(
        nproc=args.n_envs, device=args.device,
        max_episode_steps=args.max_episode_steps,
        use_terrain_check=False, seed=args.seed)
    obs = env.reset()                                    # [N,3,384,640] float01

    b_ = args.enc_base
    cfg = Dreamer4Config(
        obs_shape=(3, args.img_size, args.img_size), num_actions=num_actions,
        token_dim=args.token_dim, dyn_layers=args.dyn_layers, dyn_heads=args.dyn_heads,
        enc_depths=(b_, 2 * b_, 4 * b_, 8 * b_),
        dec_depths=(8 * b_, 4 * b_, 2 * b_, b_),
        shortcut_hidden=args.shortcut_hidden)
    wm = WorldModel(cfg).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        sd = {k: v for k, v in ck["wm"].items()
              if not k.startswith(("dynamics.action_proj", "reward", "cont"))}
        missing, unexpected = wm.load_state_dict(sd, strict=False)
        assert not unexpected, f"init checkpoint 含未知权重: {unexpected[:4]}"
        print(f"♻️  已从离线预训练热启动: {args.init}"
              f"(重新初始化 action_proj/reward/cont,共 {len(missing)} 个张量)", flush=True)
    print(f"✅ Dreamer4: {sum(p.numel() for p in wm.parameters())/1e6:.2f}M 参数", flush=True)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.amp == "fp16")
    print(f"⚙️  混合精度: {args.amp if use_amp else 'off(fp32)'}", flush=True)

    optimizer = torch.optim.AdamW(wm.parameters(), lr=args.lr, weight_decay=1e-5)
    replay = StreamReplay(args.n_envs, args.total_env_steps // args.n_envs + args.seq_len,
                          args.img_size)
    train_ids = list(range(args.n_envs - 1)) if args.n_envs > 1 else [0]
    eval_ids = [args.n_envs - 1]

    os.makedirs(args.run_dir, exist_ok=True)

    def save_ckpt(tag, step, metrics=None):
        path = os.path.join(args.run_dir, f"{tag}.pt")
        torch.save({"wm": wm.state_dict(), "optimizer": optimizer.state_dict(),
                    "env_steps": step, "cfg": vars(args), "metrics": metrics}, path)
        print(f"💾 已保存 {path}({step:,} env 步)", flush=True)

    def downsize(o):
        """[N,3,384,640] float01 → [N,3,img,img] uint8(中心裁成方形再缩放)。"""
        h, w = o.shape[2], o.shape[3]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        sq = o[:, :, top:top + side, left:left + side]
        small = F.interpolate(sq, size=(args.img_size, args.img_size),
                              mode="bilinear", align_corners=False)
        return (small * 255.0).clamp_(0, 255).to(torch.uint8).cpu()

    env_steps, it, best_gen = 0, 0, -float("inf")
    recent_losses = deque(maxlen=50)
    t0 = time.time()
    wm.train()
    try:
        while env_steps < args.total_env_steps:
            # ── 采集(随机均匀策略) ────────────────────────────────
            for _ in range(args.collect_per_iter):
                actions = torch.randint(0, num_actions, (args.n_envs,), device=device)
                obs, rew, done, infos = env.step(actions)
                small = downsize(obs)
                total_r = (rew.squeeze(1) + infos["achievement_rewards"].squeeze(1)).cpu()
                for i in range(args.n_envs):
                    replay.add(i, small[i], int(actions[i]), float(total_r[i]),
                               float(done[i, 0]))
                env_steps += args.n_envs

            # ── 世界模型更新 ─────────────────────────────────────
            if env_steps >= args.warmup_env_steps:
                for _ in range(args.updates_per_iter):
                    s = replay.sample(train_ids, args.batch_size, args.seq_len,
                                      num_actions, device, rng)
                    if s is None:
                        break
                    img, act, rew_b, cont_b = s
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        total, m = wm.loss(img, act, reward=rew_b, cont=cont_b,
                                           d_min=args.d_min, sc_weight=args.sc_weight)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    recent_losses.append(m)

            it += 1
            if it % 5 == 0:
                sps = env_steps / max(time.time() - t0, 1e-6)
                if recent_losses:
                    avg = {k: np.mean([x[k] for x in recent_losses])
                           for k in recent_losses[0]}
                    loss_str = (f"recon={avg['recon']:.4f} flow={avg['flow']:.4f} "
                                f"sc={avg['sc']:.4f} rew={avg.get('reward', 0):.4f} "
                                f"cont={avg.get('cont', 0):.4f}")
                else:
                    loss_str = "(回放预热中)"
                print(f"[iter {it:4d} | {env_steps:,} env步 | {sps:.1f} sps] {loss_str}",
                      flush=True)

            if it % args.eval_interval == 0 and env_steps >= args.warmup_env_steps:
                e = evaluate(wm, replay, eval_ids, args, device, rng)
                if e:
                    print(f"    📊 held-out env: gen={e['psnr_gen']:.2f}dB "
                          f"recon={e['psnr_recon']:.2f}dB persist={e['psnr_persist']:.2f}dB "
                          f"| reward NLL={e['reward_nll']:.3f} MAE={e['reward_mae']:.3f} "
                          f"cont_acc={e['cont_acc']:.3f}", flush=True)
                    if e["psnr_gen"] > best_gen:
                        best_gen = e["psnr_gen"]
                        save_ckpt("best", env_steps, e)
    except KeyboardInterrupt:
        print("\n⏹️  训练中断", flush=True)
    finally:
        save_ckpt("final", env_steps)
        env.close()

    print(f"✅ 完成:{env_steps:,} env 步,耗时 {(time.time()-t0)/60:.1f} 分钟,"
          f"best psnr_gen={best_gen:.2f}dB", flush=True)


if __name__ == "__main__":
    main()
