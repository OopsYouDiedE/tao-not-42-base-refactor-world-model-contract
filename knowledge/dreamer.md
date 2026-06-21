# DreamerV3(vendored)— 设计与放置说明

> ⚠️ **已退役（2026-06,统一世界基座清白重设计）**：`net/dreamer/`（vendored 的完整 DreamerV3 模型——
> networks.py 的 RSSM/编码解码器、models.py 的 WorldModel/ImagBehavior、_compat.py、config.py）**已整目录删除**
> （见 git 历史）。仅保留被 `blocks/` 复用的独立算子（GRUCell / Conv2dSamePad / ImgChLayerNorm /
> symlog·two-hot 分布 / static_scan·lambda_return）；MIT 署名与许可证随之移至 `blocks/NOTICE.dreamerv3`
> 与 `blocks/LICENSE.dreamerv3`。本文以下内容仅作已删 vendored 模型的历史记录（其中 `utils/nn.py` 等
> 映射目标早已不存在），勿据此查找文件。

## 1. 是什么 / 为什么

`net/dreamer/` 是与 `MinecraftWorldModel` 并列的**第二个世界模型**:Hafner 等的 DreamerV3
(《Mastering Diverse Domains through World Models》, arXiv:2301.04104)的完整 PyTorch 实现。

代码**原封不动照抄(verbatim)** 自社区移植 [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
(MIT,© 2023 NM512;许可证见 `net/dreamer/LICENSE`,归属见 `net/dreamer/NOTICE`)。

目的:让本仓**不再依赖外部 DreamerV3 源码**——无 git submodule、无 pip 包,实现代码物理
内置、可离线**完整加载并运行**(`build_dreamer` 一行构造 WorldModel + ImagBehavior,
CPU 即可跑前向/反向,见 `tests/unit/test_dreamer.py`)。

## 2. 放置映射(net→net/,blocks→blocks/,胶水→utils/)

按本仓"按是什么分层"的目录契约,把原仓三个大文件拆进对应层。**函数体保持 1:1 原样**,
只改了"物理拆分所必需的 import";未动任何数值逻辑。

| 原仓符号 | 落点 | 性质 |
|---|---|---|
| `networks.RSSM/MultiEncoder/MultiDecoder/ConvEncoder/ConvDecoder/MLP` | `net/dreamer/networks.py` | L2 网络 |
| `models.RewardEMA/WorldModel/ImagBehavior` | `net/dreamer/models.py` | L2 模型与装配 |
| `networks.GRUCell` | `blocks/dynamics.py` | L1 递归算子(与 ConvGRUCell 同列) |
| `networks.Conv2dSamePad/ImgChLayerNorm` | `blocks/conv.py` | L1 卷积算子 |
| `tools.symlog/symexp` + 各 `*Dist` 包装类 | `blocks/distributions.py` | L1 概率/变换算子 |
| `tools.static_scan/lambda_return` | `blocks/sequence.py` | L1 序列算子 |
| `tools.weight_init/uniform_weight_init/to_np/tensorstats/RequiresGrad/Optimizer` | `utils/nn.py` | 横向胶水(init/训练) |

两个**适配层**(本仓新增,非照抄):

- `net/dreamer/_compat.py`:把上面被拆走的名字重聚合成原 `tools` 命名空间,
  使 `networks.py`/`models.py` 仍以 `tools.X` 调用、函数体得以保持原样
  (`from net.dreamer import _compat as tools`)。
- `net/dreamer/config.py`:以纯 Python dict 复刻原仓 `configs.yaml` 的 `defaults` 段数值
  (`DREAMER_DEFAULTS`),并提供 gym-free 的 obs-space 垫片与 `build_dreamer` 装配入口
  ——替代原仓的 argparse/yaml/gym 外围装配(本仓 `net/` 不读文件、不引 gym)。

**未照抄**(与"加载模型"无关):`dreamer.py` 主训练循环、`envs/` 环境封装、`parallel.py`、
`exploration.py`、Logger/数据集读写/simulate 等运行时与数据管线。需要在线交互训练时再按需补。

## 3. 怎么加载

```python
from net.dreamer import build_dreamer
agent = build_dreamer(num_actions=17, obs_shapes={"image": (64, 64, 3)}, device="cuda:0")
agent.world_model   # 编码 + RSSM(离散 32×32 隐变量)+ 解码 + reward/cont 头
agent.behavior      # 想象中的 actor-critic(two-hot symexp 价值 + λ-return + 慢靶 critic)
```
默认结构超参逐字段等于原仓 `configs.yaml` 的 `defaults`(dyn_deter=512、dyn_stoch=32、
dyn_discrete=32、units=512 等),故加载后与原仓默认配置逐位一致。`build_dreamer(**overrides)`
可按字段覆盖,字典型字段(encoder/decoder/actor/critic/...)按 key 深合并。

## 4. 怎么确信"和上游一致"(可复现验证)

"原封不动 + 拆分不改数值"不是口头保证,而是可复现的三项实证(脚本见 git 历史中的
`runs/_verify_dreamer.py`,依赖临时 clone `runs/_vendor_src/dreamerv3-torch`,均 gitignored):

1. **类体逐字一致**:对 32 个被搬动的类/函数,`inspect.getsource(本仓版) == inspect.getsource(上游版)`
   全部成立 ⇒ 函数体一个字符都没改(只动了文件头 import 与模块 docstring,不进 getsource)。
2. **配置数值一致**:`DREAMER_DEFAULTS` 的 35 个字段与上游 `configs.yaml` 的 `defaults` 逐字段
   数值相等(注:pyyaml 把 `3e-5/1e-4` 等指数无符号标量读成**字符串**,本仓存的是正确浮点,
   数值意图一致)。
3. **权重逐位相等**:固定随机种子,分别用「本仓拆分后代码」与「上游原始源码」各构造一个
   WorldModel,46/46 参数张量 `torch.equal` 成立、state_dict 键序一致、总参数量相同
   ⇒ 结构与初始化与上游**逐位等价**。

注意 (3) 证的是"与上游 PyTorch 移植逐位一致",而上游 NM512/dreamerv3-torch 与官方 JAX 版
danijar/dreamerv3 的对齐由其自身复现实验背书(非本仓职责)。另:**未加载任何预训练 checkpoint**
——DreamerV3 按任务在线训练,无通用预训练权重;此处"权重"指结构与初始化方案,值为随机初始化。

## 5. 与本仓约定的张力(知情保留)

- vendored 代码保留了 `torch.cuda.amp.autocast/GradScaler`(新版 torch 仅告警 FutureWarning)、
  以及 `torch.cuda.amp`、`print` 调试输出。这些与本仓写作规范/I 约定不完全一致,但**为"原封不动"
  刻意保留**;不要为贴合 house style 去改 vendored 文件体(会破坏"可整体替换升级"的属性)。
  升级 DreamerV3 = 重新拉取上游、按第 2 节映射覆盖对应文件 + 同步 import。
- DreamerV3 的随机隐变量思想正是 `MinecraftWorldModel` 里 ξ 的来源(见 `net/world_model.py`
  的 ξ 段注释与 [[ultimate-goal-and-batch1-2026-06-12]] 等复盘);此处引入完整实现便于对照/借鉴。
