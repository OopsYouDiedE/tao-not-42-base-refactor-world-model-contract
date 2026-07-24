"""solaris 在线环境：mineflayer 无头 bot + prismarine-viewer headless 渲染的行为数据源。

用 mineflayer 驱动 bot 在真实 Java Minecraft 服务器上执行 episode，经 vendored 的
`engine/controller`（Node.js）连续录制官方图形风格的 mp4 + 逐帧 22 维动作契约 json。
Python 侧对外接口：

    acceptance_boundary_shots — 逐帧动作 json + mp4 → 每个动作起止帧截图（验收后处理）。

引擎为 vendored 第三方源（`engine/`，Apache-2.0），依赖靠 npm install 重建，
渲染前需应用 `viewer_patches/`（1.18+ 负 y 地形修复）并对齐 Minecraft 1.21.4。
完整复现见本目录 README.md 与仓库根 SETUP_WSL_SMOKE.md。
"""
