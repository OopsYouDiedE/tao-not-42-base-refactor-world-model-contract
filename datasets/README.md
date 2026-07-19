# 数据集

本目录保存可被不同训练流程复用的数据读取与原始数据契约。数据文件本身仍放在
已忽略的 `runs/data/`，不进入 Git。

- `vpt/`：Minecraft VPT `mp4 + jsonl` 视频读取与原始动作编码。

行为克隆优化循环属于 `train/minecraft/`，CraftGround 执行动作契约属于
`rl_training_environments/craftground/`。
