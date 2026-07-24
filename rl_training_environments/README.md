# 强化学习训练环境

本目录按后端组织在线强化学习环境。每个后端包同时拥有运行时适配、训练入口和
引擎资产，禁止跨后端引用具体实现。

- `godot/`：Godot 共享内存环境、SB3 适配、训练入口与 `engine/` 工程。
- `craftground/`：CraftGround 在线环境、奖励塑形、动作契约、回放与世界快照。
- `solaris/`：mineflayer + prismarine-viewer headless 渲染数据源，含 vendored `engine/`
  controller 渲染路、负 y 地形补丁与动作起止截图验收。

目录名使用正确英文拼写 `environments`。
