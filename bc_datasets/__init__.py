"""行为克隆数据集构建的命名空间包。

每个子包对应一个数据来源，目前只有 ``minestudio``（MineStudio 的 Minecraft 轨迹，
源自 OpenAI VPT contractor data）。新增来源时作为它的兄弟子包加入，不要塞进现有子包。

产物落在 ``runs/bc_datasets/``（Git ignored）；本包只放代码。
"""
