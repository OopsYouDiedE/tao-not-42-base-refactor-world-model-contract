# Mineflayer 动作数据来源

用 [mineflayer](https://github.com/PrismarineJS/mineflayer) 驱动一个无头 bot 在真实
Java Minecraft 服务器上**主动执行**动作,并记录每个动作的**起始 tick 与持续时长**,
产出结构化动作序列 JSON。

## 与 MineStudio 的区别与互补

- `data_pipelines/minestudio/`:**被动读取**离线 VPT 数据,动作是数据里的既有真值。
- 本子包:**主动执行**动作,可控地生成"哪个动作、何时开始、持续多久"的精确标注。

两者动作串表示对齐 `data_pipelines/annotations` 约定:移动 `F/B/L/R`、姿态
`jump/sneak/sprint`、破坏 `attack`、使用/交互 `use`、相机 `cam(dYaw,dPitch)`。

## 覆盖的动作类型

`ActionRecorder`(见 `action_recorder.js`)拦截 bot 方法,记录 6 类动作:

| 类型 | label 示例 | 记录方式 |
|---|---|---|
| 移动 | `F` `B` `L` `R` | `setControlState` 起停配对, 持续型 |
| 姿态 | `jump` `sneak` `sprint` | 同上, 持续型 |
| 转视角 | `cam(+90,+17)` | 拦截 `look`, 记录角度偏移(度) |
| 合成 | `craft:oak_planks` | 拦截 `craft`, 记录产物与数量 |
| 放置 | `use` | 拦截 `placeBlock`, 记录参照方块 |
| 破坏 | `attack` | 拦截 `dig`, 记录目标方块 |

每条记录字段:`type / label / startTick / endTick / durationTicks / startMs /
durationMs / detail`。时间基准是 `bot.time.age`(世界年龄, **20 tick/秒**),与
MineStudio 20fps 对齐;毫秒字段用挂钟时间,便于对齐真实录制。

## 运行前置

Node.js 18+;目标服务器为**真实 Java Minecraft**(flying-squid 等纯 JS 服务器不实现
服务端合成事务,`craft` 会超时,不能用)。服务器需满足:

- **创造模式**(`gamemode=creative`):`creative.setInventorySlot` 备料。
- **`spawn-protection=0`**:默认出生点 16 格内禁止非 OP 放置/破坏方块,否则
  `placeBlock` / `dig` 在出生点附近会以 `blockUpdate did not fire` 超时失败。

完整服务器搭建流程见 `SETUP.md`。

## 关键节点观测截图 (obs→action 配对)

仅记录动作时长还不够,要成为 VLA 训练数据还需**每个动作节点的观测帧**。
`frame_capturer.js` 基于 [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer)
无头渲染,按需对 bot 当前第一人称视角截取单帧 PNG(不同于连续录制 mp4)。

`example_keyframes.js` 把二者串起来:在**每个动作发生前**截取观测帧,执行动作并由
`ActionRecorder` 记录其起止,产出 `{ image, startTick, actions }` 的节点序列,即
observation(t) → action(t) 配对。输出 PNG + `manifest.json` + 可直接看的 `gallery.md`。

要点:

- **观测忠实于真实朝向**:截图不人为调整视角,bot 当时朝哪就拍哪,保证观测与动作
  执行时刻的朝向一致(哪怕拍到天空)。
- **持续渲染驱动 mesh**:worker 线程异步网格化,单纯 sleep 不推进;`FrameCapturer._pump`
  在 settle 期间反复 `update()+render()` 让 chunk 网格铺满,避免碎片/空白画面。
- **需虚拟显示**:无头渲染依赖 GL,用 `xvfb-run` 运行(见 `SETUP.md`)。

参考样本见 `sample/keyframes/`(7 节点: 移动/跳跃/转视角/合成/放置/破坏 + 末帧,
含 `gallery.md` 可直接浏览图文对照)。
