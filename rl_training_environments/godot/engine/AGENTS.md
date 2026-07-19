# Godot 子系统开发规范 (AGENTS.md)

> 本文是 Godot Meta RL 子系统的局部规则，对 AI 助手与人类开发者同等适用。
> 通用写作纪律沿用仓库根 [../../../AGENTS.md](../../../AGENTS.md)；
> 本文只补充 Godot 侧特有的约束。子系统概览见 [README.md](README.md)，方法级说明见 [code_analysis.md](code_analysis.md)。

---

## 1. 分层与放置

- **引擎侧资产**（C# / GDScript / 场景 / 工程配置）放在本文件夹 `engine/`。
- **Python 通信、进程、SB3 适配与训练入口**放在父目录 `rl_training_environments/godot/`。
- 后续 CraftGround 环境放入同级 `rl_training_environments/craftground/`，不与 Godot 实现互相 import。
- 新增环境任务继承 `environment_model_base.gd`（`EnvironmentModelBase`），放入 `meta_environments/`；通用逻辑下沉到基类，任务专属逻辑通过基类虚函数实现。

## 2. 通信契约不变量

- 观测 / 动作的形状与 dtype 是 Python 与 Godot 双方的硬契约，见 [README.md](README.md) 协议表；任一侧改动必须同步另一侧与 `code_analysis.md`。
- 握手用轮询计数器 seqlock：收到对方应答前不得推进，保证帧号严格 +1、无丢帧。
- 共享内存走文件后端 mmap，不依赖 Windows 命名内核对象，以保持跨平台。

## 3. 跨平台

- 改动渲染 / 共享内存相关代码后，须在 Windows 与 Linux 无头（Xvfb + Vulkan 软渲染）两端确认能回读到非零像素。
- `--headless` 哑渲染器不产出像素，不能用于验证图像回读。

## 4. 文档同步

- 修改方法签名、握手协议或目录结构后，必须同步更新 [code_analysis.md](code_analysis.md)（方法调用图）与 [README.md](README.md)（协议表）。
- 历史活动（改了什么、为什么改）写进 git commit message，不写进文档。
