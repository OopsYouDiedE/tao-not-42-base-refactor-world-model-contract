# 数据集

本目录保存可被不同训练流程复用的数据读取与原始数据契约。数据文件本身仍放在
已忽略的 `runs/data/`，不进入 Git。

- `vpt/`：Minecraft VPT `mp4 + jsonl` 流式读取，以及 MineStudio
  `image + action` 和可选 `meta_info` LMDB 课程读取与原始动作编码。

`rolling_download.py` 接受 JSON/JSONL 下载清单，以 `.part` 临时文件和原子重命名
发布完整文件对，并维持有界磁盘缓存。`video_dataset.py` 只使用 clip 级 uint8
缓存，不提供全量 RAM 预载旁路。

MineStudio 每个课程阶段必须全量下载 `action/**`，只允许 `image/**` 按 LMDB 分片
轮换。`meta_info/**` 是默认关闭的可选诊断目标，启用时同样必须全量下载。各模态的
分片编号没有配对含义；`minestudio_dataset.py` 扫描已启用模态的 episode 索引后按
episode 和帧数对齐。
`minestudio_download.py` 不依赖 CUDA，支持通过 `--data-root` 定位挂载数据盘、
`--modalities` 选择顶层模态、`--image-shard-index` 预取多个图像分片，或使用
`--all-image-shards` 下载该阶段全部图像。相同 revision 与目标目录可安全重复执行。

行为克隆优化循环属于 `train/minecraft/`，CraftGround 执行动作契约属于
`rl_training_environments/craftground/`。
