# Mineflayer 动作数据来源

用 [mineflayer](https://github.com/PrismarineJS/mineflayer) 驱动一个无头 bot 在真实
Java Minecraft 服务器上**主动执行**动作,并记录每个动作的**起始 tick 与持续时长**,
产出结构化动作序列 JSON。

## 定位

本子包**主动执行**动作,可控地生成"哪个动作、何时开始、持续多久"的精确标注,是当前
Minecraft 侧唯一的行为数据来源。

动作串表示对齐 `net/action_token_codec.py` 的规范动作:移动 `F/B/L/R`、姿态
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
durationMs / detail`。时间基准是 `bot.time.age`(世界年龄, **20 tick/秒**);毫秒字段用
挂钟时间,便于对齐真实录制。

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

`survival_mining.js` 是完整示例:**生存模式下从零走完一条真实挖矿链**——
走路探索 → 砍树采原木 → 合成木板/木棍/工作台 → 放置工作台 → 合成木镐 → 装备 →
找石头 → 用镐挖矿。在**每个关键动作发生前**截观测帧,执行动作并由 `ActionRecorder`
记录其起止,产出 `{ image, startTick, actions }` 节点序列,即 observation(t) → action(t)
配对。输出 PNG + `manifest.json` + 可直接看的 `gallery.md`。

要点:

- **真·从零 + 真实物理**:开局 `/clear` 清空库存(需 OP),原木靠真砍树得来;生存模式下
  徒手砍橡木约 60–80 tick(3–4 秒)、**木镐挖石头约 20–40 tick(~1 秒)且掉落圆石**
  (徒手挖石头则慢至 ~180 tick 且无掉落——可据此验证工具是否真握在手上)。
- **节点聚焦核心动作**:寻路(pathfinder 的移动/转视角微动作)在截图**之前**完成、不计入
  节点;节点内只记"看向目标 + 挖/合成/放置",所以每个挖矿节点就是干净的 `cam + attack`。
- **观测忠实于真实朝向**:截图不人为调整视角,bot 当时朝哪就拍哪,保证观测与动作
  执行时刻的朝向一致。
- **持续渲染驱动 mesh**:worker 线程异步网格化,单纯 sleep 不推进;`FrameCapturer._pump`
  在 settle 期间反复 `update()+render()` 让 chunk 网格铺满,避免碎片/空白画面。
- **需虚拟显示**:无头渲染依赖 GL,用 `xvfb-run` 运行(见 `SETUP.md`)。

参考样本见 `sample/mining/`(14 节点: 走路→砍树×3→合成×3→放工作台→合成木镐→
挖石头×4 + 末帧),`gallery.md` 可直接浏览"节点观测图 + 该节点后动作(含挖掘时长)"对照。

## 连续录制路的动作起止截图

除按节点截单帧外,solaris headless 渲染路会连续录制 mp4 + 逐帧动作 json(每帧 `action`
含布尔按键 + `camera:[dx,dy]`,`frame_count` 与 mp4 帧 1:1)。`action_boundary_shots.py`
对这种产物做后处理:逐帧检测每个动作的 `False↔True` 跳变,在开始/结束帧抽截图。

```bash
python -m data_pipelines.mineflayer_actions.action_boundary_shots \
  --json <逐帧动作.json> --mp4 <同序列录像.mp4> --out runs/xxx --contact-sheet
```

验收样本见 `sample/action_boundaries/`(8 个动作边界: hotbar 换镐 / camera 转头 / mine
挖矿的起止 + 录像 + contact sheet)。

## solaris 渲染前必读:负 y 地形补丁 + 版本对齐

solaris 用的 vendored `prismarine-viewer-colalab` 对 Minecraft 1.18+ 负 y 世界有三处 bug,
不打补丁地形不渲染(画面只有实体漂在虚空)。补丁与幂等应用脚本固化在
`solaris_viewer_patches/`,跑渲染前先 `bash solaris_viewer_patches/apply.sh <viewer 包根>`。
另需 server / bot / viewer 三者版本对齐到 **1.21.4**(否则地形呈竖刺畸变)。完整原理与 WSL
复现见仓库根 `SETUP_WSL_SMOKE.md` 第 7 节。
