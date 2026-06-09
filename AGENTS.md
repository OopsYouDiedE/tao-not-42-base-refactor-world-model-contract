# 助手约束与开发规范 (AGENTS.md)

> **文档性质**:对 AI 助手和人类开发者均有效的硬性约束。只记录**规则**,不记录历史活动(历史活动归 git log)。

---

## 1. 文档规范

- 所有项目文档(`.md` 文件)必须使用**中文**撰写和更新。
- **SSOT 原则**:代码是逻辑的唯一事实来源。Markdown 只解释宏观架构、算法设计思想与物理直觉,禁止在 Markdown 中描述易变的微观实现逻辑。
- 所有 Class / Function 必须用 NumPy / Google 风格 Docstring 明确声明 Shape 和 Dtype 契约。
- 任何对 `net/` 或 `train/` 的目录结构、类名、核心方法签名的修改,**必须同步更新** `knowledge/` 对应专题。
- 废弃模块必须**物理删除**原文件,禁止留空文件或仅写注释,并在 commit message 中显式描述迁移路线。

---

## 2. 生产代码纯净原则

- 核心代码(`net/`、`train/`)必须保持纯净,**禁止**掺杂任何本地 Mock、调试数据加载或环境兼容降级逻辑。
- 所有依赖(`mamba_ssm`、`procgen`、`datasets` 等)**缺包直接报错**,不写 `try/except` 降级。
- CPU 兼容、Mamba Mock、离线数据 Mock **必须且仅允许**存在于 `tests/` 目录。

---

## 3. 测试目录结构

```
tests/
  conftest.py          全局 Mock 注入器(测试环境无 mamba_ssm 时自动注入分组卷积平替)
  mock_mamba.py        Mamba 算子的分组卷积平替实现
  unit/                纯单元测试(无需网络,无需 GPU:几何、损失计算)
  integration/         集成测试(离线数据流 + Mamba Mock 下的完整前向与反向)
  data/                本地微型测试数据(如 movi_e_static_sample.npz)
```

---

## 4. 环境矩阵

| 硬件环境 | 可运行内容 | 不可运行 |
| :--- | :--- | :--- |
| **Linux + CUDA(生产)** | 全量:net/ + train/ + procgen + mamba_ssm | — |
| **Windows + CUDA(开发)** | 仅 `tests/`(含 Mamba Mock);`blocks/` 单元测试 | net/ 生产代码(缺 mamba_ssm) |
| **CPU Only** | 🚫 不支持 | 一切 |

> `net/encoder.py` 直接 `from mamba_ssm import Mamba`,无任何降级。Windows 开发机只能通过 `tests/conftest.py` 的 Mock 注入跑测试,不能直接运行生产训练代码。

---

## 5. 训练数据与文件系统

- 所有下载数据、checkpoints、训练日志统一落在 `runs/`(已加入 `.gitignore`,**不入库**)。
- `runs/data/`       下载的数据集缓存
- `runs/checkpoints/` 模型权重
- `runs/logs/`       训练日志

---

## 6. 数值不变量(I1–I8,硬约束)

| # | 不变量 |
|---|---|
| I1 | 除法分母 `clamp(min=ε)`,ε ≥ 1e-4(绝不用 1e-12) |
| I2 | 不做无界 exp;softmax/Sinkhorn 走 log 域 |
| I3 | 所有头输出构造上有界(tanh/sigmoid/softplus+ε/exp-clamp) |
| I4 | 危险算子(除/normalize/exp/求逆/投影)强制 fp32 |
| I5 | 每个递归/残差更新增益受限(< 1 或 clamp) |
| I6 | 不稳定组合优化(Sinkhorn/匈牙利)只在损失里,不进前向 |
| I7 | 递归/rollout 路径用 LayerNorm/GroupNorm,不用 BatchNorm |
| I8 | 长链梯度受控;长程依赖不取消 detach,改用 stop-grad 对比损失 |

---

## 7. 写作规范

- 禁止使用情绪化或夸张修饰词汇(如"暴降"、"完美"、"惊人"、"瞬间"等)。
- 所有结论和成效必须用数据、收敛指标或物理原理说明。
