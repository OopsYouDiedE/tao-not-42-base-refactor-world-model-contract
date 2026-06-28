# CraftGround PPO+AD 首轮 run 结论（2026-06-27）

## 1. 训练结果
- **配置**：4 环境 / n_steps=256 / ppo_batch_size=64 / RAW 编码 / GPU 渲染(DISPLAY=:0) / expandable_segments
- **结果**：1M 步、**5.93 小时**、最终 **4/16 成就**（root + mine_wood + punch_tree + mine_stone）
- **判定**：学会了"砍木→挖石"科技链；但 4/16 是"曾经解锁过"的弱指标，不代表稳定复现。

## 2. ⚠️ 头号问题：模型没保存
- `runs/craftground_ppo_ad_v1/` 无任何 `.pt`，5.93 小时权重丢失。
- 训练脚本缺 `torch.save`（对比 `runs/crafter_*/final.pt` 都有）。
- **任何后续 run 之前必须先补 checkpoint 保存。**

## 3. GPU 渲染（本轮已落地）
- Minecraft 渲染从 CPU(llvmpipe) 切到 3090(DISPLAY=:0)，单环境实测 **4× 提速**（51→207 sps）。
- 修了 craftground 打包 bug（`environment/craftground_native.py` 转发 shim）。
- ZEROCOPY 未启用：需专用无头 Xorg 避开 kwin 改窗口尺寸（`width==textureWidth` 断言），留作后续。

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
