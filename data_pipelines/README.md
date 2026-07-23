# 数据集

`minestudio/download.py` 根据 `7xx / 9xx / 10xx` 范围完整下载选定 MineStudio
模态，默认下载 `image + action`。下载结果保存在 Git ignored 的 `runs/data/` 或用户
指定的数据盘，不做训练期滚动删除。

`minestudio/dataset.py` 扫描全部本地图像、动作及可选元数据 LMDB，以 episode
名称和帧数对齐后返回固定连续窗口。不同模态的 `part-*` 编号没有配对含义。

## 动作语言标注（annotations 子包）

`annotations/sonnet_action_annotator.py` 为 MineStudio 窗口的**真值动作**补充语言化理由，
产出 `Type/Key/Action/Explanation` 结构化标注。原理、用法与案例见
`annotations/AGENTS.md`。

### 采样与聚合约定

- 原始帧率 **20fps**，标注时降采样至 **5fps**（`stride=4`，每 4 帧取 1 帧画面）。
- 每个 5fps 步的动作在对应 4 帧上**聚合**：二值键取 OR（窗口内按下过即算），相机取
  偏离中性最大的分量（保留主导转向，滤掉往返抖动）。
- 去抖阈值：mu-law bin 偏离中性 `|Δ| ≤ 1`（`CAMERA_JITTER_BIN = 1`），低于此阈值
  的相机分量在喂给标注模型前已由 `describe_action` 过滤，不产生标注条目。

### 核验样本（10xx 窗口 #1370，5fps 降采样 8 步）

合成图见 `annotations/sample/window_1370_mosaic.png`（2行×4列，每帧下方标真值动作，
入库可直接打开）。真值序列见 `annotations/sample/window_1370_actions.json`，完整参考
标注见 `annotations/sample/window_1370_annotation.md`。

每帧 448×252，原始 20fps，stride=4，各区域像素均值如下（由 numpy 真算，可复现）：

| step | 真值动作（含抖动） | 去抖后关键动作 | 顶部均值 RGB | 中心均值 RGB |
|---:|---|---|---|---|
| 0 | `use cam(+1,-1)` | `use` | R60 G66 B49 | R139 G123 B84 |
| 1 | `F jump cam(-2,+3)` | `F jump, turn-left, look-down` | R59 G65 B49 | R144 G124 B83 |
| 2 | `F jump cam(+0,+1)` | `F jump` | R62 G66 B51 | R130 G114 B78 |
| 3 | `F jump cam(+2,+2)` | `F jump, turn-right, look-down` | R64 G67 B52 | R148 G124 B80 |
| 4 | `cam(+2,+3)` | `turn-right, look-down` | R64 G67 B51 | R106 G93 B63 |
| 5 | `B cam(+3,+3)` | `B, turn-right, look-down` | R67 G69 B53 | R125 G108 B73 |
| 6 | `B R h2 cam(+1,+0)` | `B, strafe-right, hotbar-2` | R67 G69 B54 | R134 G111 B71 |
| 7 | `B R h1 cam(+2,+1)` | `B, strafe-right, hotbar-1, turn-right` | R70 G71 B54 | R150 G121 B75 |

**颜色解读**（可对照 mosaic.png 核验）：
- 顶部 R60-70/G65-71/B49-54 → 暗绿色，户外白天草/叶，不是洞穴或蓝天。
- 中心 R>G>B 差值约 40-70 → 暖棕黄色，木材/泥土类方块（石头的 R≈G≈B 差值<20）。
- 整段多帧 pitch+（低头）+ use 动作 → 玩家俯视地面方块做交互操作。

采样与聚合逻辑即 `annotations/sonnet_action_annotator.py` 的 `sample_window` /
`aggregate_actions`（同一套 5fps 降采样与去抖），上表的帧与真值由其在窗口 #1370 上产出。
仓库内固化的 `annotations/sample/window_1370_mosaic.png` 已把 8 帧拼在一张图里，可直接
肉眼核对上述颜色判断，无需重新导出。

## Mineflayer 动作数据来源（mineflayer_actions 子包）

`mineflayer_actions/` 用 mineflayer 驱动无头 bot 在真实 Java Minecraft 服务器上**主动
执行**动作，记录每个动作的**起始 tick 与持续时长**，产出结构化动作序列。与 MineStudio
被动读取离线真值互补：这里可控地生成"哪个动作、何时开始、持续多久"的精确标注。

覆盖 6 类动作：移动 `F/B/L/R`、姿态 `jump/sneak/sprint`、转视角 `cam(dYaw,dPitch)`、
合成 `craft:*`、放置 `use`、破坏 `attack`；动作串表示与 `annotations` 约定对齐，时间基准
`bot.time.age`（20 tick/秒）与 MineStudio 20fps 对齐。原理、动作字段与运行前置见
`mineflayer_actions/AGENTS.md`，从零搭建服务器到采集的完整流程见
`mineflayer_actions/SETUP.md`，参考输出见 `mineflayer_actions/sample/session_actions.json`。

该子包是 Node.js 工具链（依赖见其 `package.json`），产物（server.jar、世界、
node_modules、采集 JSON）均为运行期数据，不入库。
