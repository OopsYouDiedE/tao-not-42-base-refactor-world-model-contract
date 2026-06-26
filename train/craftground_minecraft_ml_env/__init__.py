"""Craftground Minecraft ML 环境定义与接口。

本模块提供 Minecraft RL 的标准化环境定义：
  - 观测空间（RGB 图像）
  - 动作空间（离散，27 维）
  - 成就空间（Minecraft 官方成就）
  - 奖励设计（稀疏 + 成就触发）

对外接口：
    MINECRAFT_ACHIEVEMENTS — Minecraft 1.21 全成就列表
    MinecraftRLEnv — 标准化环境类
"""
