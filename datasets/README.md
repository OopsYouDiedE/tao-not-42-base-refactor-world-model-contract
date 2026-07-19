# 数据集

本目录保存可被不同训练流程复用的数据读取与原始数据契约。数据文件本身仍放在
已忽略的 `runs/data/`，不进入 Git。

- `vpt/`：Minecraft VPT `mp4 + jsonl` 流式视频读取、滚动下载与原始动作编码。

`rolling_download.py` 接受 JSON/JSONL 下载清单，以 `.part` 临时文件和原子重命名
发布完整文件对，并维持有界磁盘缓存。`video_dataset.py` 只使用 clip 级 uint8
缓存，不提供全量 RAM 预载旁路。

行为克隆优化循环属于 `train/minecraft/`，CraftGround 执行动作契约属于
`rl_training_environments/craftground/`。
