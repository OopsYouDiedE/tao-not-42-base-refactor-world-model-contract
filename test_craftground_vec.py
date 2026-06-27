#!/usr/bin/env python3
"""测试 CraftgroundVecEnv 是否正常工作"""

import sys
sys.path.insert(0, '.')

import torch
import time

def test_vec_env():
    from train.craftground.env import CraftgroundVecEnv

    print("=" * 60)
    print("测试 CraftgroundVecEnv")
    print("=" * 60)

    print("\n[1/5] 创建环境 (1 个进程)...")
    env = CraftgroundVecEnv(nproc=1, device='cpu', max_episode_steps=100)
    print("✓ 环境创建成功")

    print("\n[2/5] Reset 环境...")
    t0 = time.time()
    obs = env.reset()
    print(f"✓ Reset 成功 ({time.time()-t0:.1f}s)")
    print(f"   观测形状: {obs.shape}")
    print(f"   观测范围: [{obs.min():.3f}, {obs.max():.3f}]")

    print("\n[3/5] 执行 10 步...")
    for i in range(10):
        actions = torch.randint(0, 27, (1,))
        obs, rewards, dones, infos = env.step(actions)
        print(f"   步骤 {i+1}: reward={rewards[0].item():.3f}, done={dones[0].item()}")
    print("✓ 步进成功")

    print("\n[4/5] 检查 info 内容...")
    print(f"   info keys: {list(infos.keys())}")
    if 'achievements' in infos:
        print(f"   成就向量形状: {infos['achievements'].shape}")
    print("✓ Info 正常")

    print("\n[5/5] 关闭环境...")
    env.close()
    print("✓ 关闭成功")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！Craftground 环境运行正常")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_vec_env()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
