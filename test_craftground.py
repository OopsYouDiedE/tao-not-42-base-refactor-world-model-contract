#!/usr/bin/env python
"""快速测试 Craftground 环境集成。"""

import sys
import torch
from train.craftground.env import MinecraftCraftgroundEnv, CraftgroundVecEnv

print("=" * 60)
print("🧪 测试 Craftground 环境集成")
print("=" * 60)

# 测试 1: 单环境
print("\n[1/3] 测试 MinecraftCraftgroundEnv...")
try:
    env = MinecraftCraftgroundEnv(seed=0, max_steps=10)
    obs = env.reset()
    print(f"  ✅ Reset OK: obs.shape={obs.shape}, dtype={obs.dtype}")

    for step in range(3):
        action = 0  # noop
        obs, rew, done, info = env.step(action)
        print(f"  Step {step+1}: obs.shape={obs.shape}, rew={rew}, done={done}")

    env.close()
    print("  ✅ 单环境测试 PASSED")
except Exception as e:
    print(f"  ❌ 单环境测试 FAILED: {e}")
    sys.exit(1)

# 测试 2: 向量化环境
print("\n[2/3] 测试 CraftgroundVecEnv...")
try:
    vecenv = CraftgroundVecEnv(nproc=2, device="cpu", max_episode_steps=10)
    obs = vecenv.reset()
    print(f"  ✅ Reset OK: obs.shape={obs.shape}, dtype={obs.dtype}")

    for step in range(3):
        actions = torch.tensor([0, 1], dtype=torch.long)  # 2 envs
        obs, rewards, dones, infos = vecenv.step(actions)
        print(f"  Step {step+1}: obs.shape={obs.shape}, rew.shape={rewards.shape}, done.shape={dones.shape}")

    vecenv.close()
    print("  ✅ 向量化环境测试 PASSED")
except Exception as e:
    print(f"  ❌ 向量化环境测试 FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！Craftground 环境已就绪")
print("=" * 60)
