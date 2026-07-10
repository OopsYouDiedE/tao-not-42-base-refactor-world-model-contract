# CraftGround PPO+AD 首轮 run 结论（2026-06-27；§3 于 2026-07-10 在 L4 机回填 ZEROCOPY/EGL 实测）

## 1. 训练结果
- **配置**：4 环境 / n_steps=256 / ppo_batch_size=64 / RAW 编码 / GPU 渲染(DISPLAY=:0) / expandable_segments
- **结果**：1M 步、**5.93 小时**、最终 **4/16 成就**（root + mine_wood + punch_tree + mine_stone）
- **判定**：学会了"砍木→挖石"科技链；但 4/16 是"曾经解锁过"的弱指标，不代表稳定复现。

## 2. ⚠️ 头号问题：模型没保存
- `runs/craftground_ppo_ad_v1/` 无任何 `.pt`，5.93 小时权重丢失。
- 训练脚本缺 `torch.save`（对比 `runs/crafter_*/final.pt` 都有）。
- **任何后续 run 之前必须先补 checkpoint 保存。**

## 3. GPU 渲染（2026-07-10 更新：ZEROCOPY 已在 L4 机跑通，本节数据整体回填）

### 3.1 渲染路径基准（L4 机,2026-07-10,`tests/bench_render_craftground.py`）
单环境 640×360、同一 seed=0 动作序列、各 500 step。**同卡有 BC 训练全程并行**
（`bc_vpt_warmstart`,5258MiB,GPU util 49–98%,8 个 CPU dataloader worker）——
三臂受同等争抢,横向可比,绝对值偏低。

| 臂 | X server | 编码 | steps/s | reset 秒 | obs 落点 | 进程 RSS |
|---|---|---|---|---|---|---|
| a | Xvfb :99（llvmpipe,CPU） | RAW | **13.1** | 479.5¹ | cpu uint8 | 5523 MB |
| b | Xorg :1（NVIDIA L4） | RAW | **37.3** | 110.6 | cpu uint8 | 3247 MB |
| c | Xorg :1（NVIDIA L4） | ZEROCOPY | **25.3** | 108.0 | **cuda:0 torch.uint8** | 3323 MB |

¹ 含首次 gradle 构建 + Minecraft 1.21 资产下载;后两臂为缓存后冷启动。
显存增量:ZEROCOPY 臂 python 侧 253MiB(CUDA context+IPC)+ java 侧 220MiB;BC 训练全程 5258MiB 不受影响。

- **GPU 渲染收益复现**:a→b 提速 2.8×(13.1→37.3),与旧 3090 机的 4×(51→207 sps)同向;
  倍数与绝对值低于旧机,归因于本机 CPU 更弱且 BC 训练争抢 CPU/GPU。
- **ZEROCOPY 跑通但本基准下比 RAW 慢 32%**(25.3 vs 37.3)。机制:上游实现每帧做
  `cudaGraphicsGLRegisterImage`/unregister(`MinecraftEnv/src/main/cpp/framebuffer_capturer_cuda.cpp:93` 起)
  加 python 侧每帧 `clone()[:, :, :3].flip(0)`,interop 固定开销吃掉零拷贝节省。注意口径:RAW 的 sps
  **不含** H2D 上传(训练时还要 `.to(cuda)`,640×360×3≈0.7MB/帧),ZEROCOPY 的 obs 已在 GPU;
  端到端训练差距会小于 32%,但按本数据 **GRPO rollout 首选 Xorg+RAW**,ZEROCOPY 不是吞吐解药。
- **对 GRPO rollout 预算**(一组 4 rollout × 400 ticks = 1600 tick):Xvfb 122s / Xorg+RAW 43s /
  ZEROCOPY 63s。GPU 渲染相对 Xvfb 省约 65% 采集墙钟。

### 3.2 ZEROCOPY 病灶清单（全部已定位,复现路径见基准脚本）
1. **旧病灶(kwin 改窗口尺寸触发 `assert(width==textureWidth)`,capturer_cuda.cpp:132)**:
   用无 WM 的无头 Xorg 即避开,本轮 500 step 未触发。**"留作后续"状态解除。**
2. craftground 2.6.15 打包 bug:`environment/` 下相对导入 `.craftground_native`,而 .so 在包根
   → 误报 "install craftground[cuda]"。运行时 shim:`sys.modules` 预注册(见基准脚本
   `patch_craftground_native()`,不改 site-packages)。
3. craftground 2.6.15 属性名 bug:`initialize_zerocopy` 写 `self.observation_tensor_type`,
   `convert` 分派读 `self.internal_type` → 首帧必 ValueError。同函数运行时补上。
   两处均为上游 bug,换机器/重装后**必须经 `patch_craftground_native()` 再用 ZEROCOPY**。

### 3.3 L4 机无头 Xorg 配置差异（对 `xorg.conf.headless` 的两处本机修正）
- BusID:`nvidia-smi --query-gpu=pci.bus_id` = 00000000:00:03.0 → `PCI:0:3:0`。
- **L4 是虚拟显示型数据中心 GPU,不支持 `Option "UseDisplayDevice" "None"`**
  (Xorg 报 "not supported with virtual display" 拒起)→ 改用
  `Option "AllowEmptyInitialConfiguration" "True"`。
- Colab 式布局的系统层前置(重启后需重做):`/etc/ld.so.conf.d/` 收录 `/usr/lib64-nvidia` + `ldconfig`;
  Xorg 配置 `ModulePath` 指向 `/usr/lib64-nvidia/xorg/modules`(nvidia_drv.so/libglxserver_nvidia 所在);
  apt 装 xserver-xorg-core、openjdk-21-jdk、libglew-dev、libgl-dev。
  生成的完整配置在 `runs/zerocopy_bench/xorg.conf.l4`(gitignored,照 3.3 两条可再生)。

### 3.4 EGL 无 X 路线（第四臂探测,两层判决分开）
- **驱动层:可用。** 无 DISPLAY 全程,`eglQueryDevicesEXT` 枚举到 L4 →
  `EGL_PLATFORM_DEVICE_EXT` display → surfaceless GL 4.6.0 context → FBO 清屏回读
  [0,255,0,255] 全过(driver 580.82.07;需补 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`)。
  复跑:`python tests/bench_render_craftground.py --arm egl-probe`。
- **CraftGround 栈:不支持,缺口在 Minecraft/GLFW 窗口层,不在驱动、也不在 capturer。**
  Minecraft 1.21 经 LWJGL/GLFW 建窗,GLFW 初始化就要求 x11/wayland 平台;
  craftground 全源码(Kotlin/C++/mixin)无任何 EGL/offscreen 分支,
  `WindowOffScreenMixin` 只有被注释掉的 hide-window 提示。要走 EGL 需上游特性
  (GLFW null-platform 无 GL context 能力,或 mixin 强制 `GLFW_EGL_CONTEXT_API` 仍依赖窗口系统),
  不属于我们侧可修——已按"不硬改第三方"放弃,候选项记 `docs/next_session.md`。

## 4. 吞吐瀑布（实测，端到端 43.9 sps）
| 桶 | 占比 |
|---|---|
| ④ PPO 更新（环境空转） | **58.5%** 🔴 |
| ① 纯环境步进（4个串行） | 33.2% |
| ② 编码器前向（收集） | 8.2% |
| ③ 地形检测重置 | 0%（profiler 窗口 768<1000 步未捕获；真实摊薄约 5-10%） |

**关键发现**：墙钟最大浪费是 **PPO 更新时 4 个 Minecraft 干等（58.5%）**，根因是解冻的 11.8M YOLO 编码器在更新里被前向+反向跑 64 遍/rollout。

## 5. 优化优先级（按实测收益）
1. **🔴 异步 Actor-Learner（IMPALA/V-trace）**：环境在更新时继续采集，吃掉 58.5% 空转 → 墙钟近乎翻倍。同时回答"PPO 转部分 offline"。
2. **🟡 砍更新成本**：ppo_epochs 4→2；或编码器冻结期缓存特征跳过重编码。
3. **🟡 并行 4 个环境**：解掉 33% 的串行步进。

## 6. 评测正确性（回答"我们到底学到没有"）
当前"X/16 曾解锁"是最弱证据。要真验证：
1. **随机策略 baseline**（金标准，没有它所有成就数无意义）
2. **per-episode 成功率**（滑窗，替代累积"曾解锁"）
3. 存 rollout 视频肉眼看行为
