"""渲染后端 A/B 基准：1 环境固定动作，测 sps / 每步 ms。

用法：DISPLAY=:99 python bench_render.py  (llvmpipe / CPU)
      DISPLAY=:0  python bench_render.py  (NVIDIA 3090 / GPU)
"""
import os
import time
import sys

from train.craftground.env import MinecraftCraftgroundEnv

DISPLAY = os.environ.get("DISPLAY", "?")
WARMUP = 20
MEASURE = 200
ACTION = 1  # forward，固定动作

print(f"[bench] DISPLAY={DISPLAY}  warmup={WARMUP} measure={MEASURE}", flush=True)

t_boot0 = time.time()
env = MinecraftCraftgroundEnv(seed=0, max_steps=100000, port=9100)
env.reset()
print(f"[bench] 环境启动+reset 耗时 {time.time()-t_boot0:.1f}s", flush=True)

# 预热（JIT / 区块加载稳定）
for _ in range(WARMUP):
    env.step(ACTION)

# 计时
t0 = time.time()
for i in range(MEASURE):
    env.step(ACTION)
dt = time.time() - t0

sps = MEASURE / dt
ms = dt / MEASURE * 1000
print(f"[bench] DISPLAY={DISPLAY}  ->  {sps:.1f} sps  |  {ms:.1f} ms/step  ({MEASURE} steps in {dt:.1f}s)", flush=True)

env.close()
