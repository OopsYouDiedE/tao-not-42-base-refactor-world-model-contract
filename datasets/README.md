# 数据集

`minestudio/download.py` 根据 `7xx / 9xx / 10xx` 范围完整下载选定 MineStudio
模态，默认下载 `image + action`。下载结果保存在 Git ignored 的 `runs/data/` 或用户
指定的数据盘，不做训练期滚动删除。

`minestudio/dataset.py` 扫描全部本地图像、动作及可选元数据 LMDB，以 episode
名称和帧数对齐后返回固定连续窗口。不同模态的 `part-*` 编号没有配对含义。
