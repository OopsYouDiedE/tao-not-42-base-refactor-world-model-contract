# Godot 训练项目开发规范

## 范围

- 引擎资产放在 assets/godot_meta_rl/。
- 通用 Python 通信与进程基础设施放在 utils/godot_rl/。
- 训练装配与任务适配放在 train/godot_meta_rl/。
- 测试放在 tests/，训练产物放在已忽略的 runs/。

## 依赖

- 依赖只能由生产代码中的实际 import 或引擎工程文件证明。
- Godot 侧使用 Godot 4.6.1 .NET 和 .NET 8；Python 侧通过 mmap 通信。
- 禁止添加 godot-python 或其他未被代码使用的 Python binding。
- 不自动猜测 Linux 发行版的 Godot、显示服务或驱动包名。

## 协议

- 图像、元数据、动作和 seqlock 布局是 Python/C# 双方的硬契约。
- 修改协议时必须同步 assets/godot_meta_rl/README.md、
  assets/godot_meta_rl/code_analysis.md 与对应测试。
- 图像训练必须使用能够真实渲染像素的显示/Vulkan 环境；不得用
  --headless 哑渲染结果作为图像回读验证。

## 写作与提交

- Markdown 与代码说明使用中文。
- 函数和类 docstring 声明张量或数组的 Shape、Dtype 与单位。
- 废弃模块物理删除，不保留空壳。
- 每轮修改完成后运行测试并提交 Git commit。
