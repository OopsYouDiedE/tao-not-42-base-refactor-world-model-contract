# 数据集

本包收拢可跨训练流程复用的数据读取与原始数据契约。当前 Minecraft 侧的行为数据来源是
`mineflayer_actions/`。

## Mineflayer 动作数据来源（mineflayer_actions 子包）

`mineflayer_actions/` 用 mineflayer 驱动无头 bot 在真实 Java Minecraft 服务器上**主动
执行**动作，记录每个动作的**起始 tick 与持续时长**，产出结构化动作序列。可控地生成
"哪个动作、何时开始、持续多久"的精确标注。

覆盖 6 类动作：移动 `F/B/L/R`、姿态 `jump/sneak/sprint`、转视角 `cam(dYaw,dPitch)`、
合成 `craft:*`、放置 `use`、破坏 `attack`；动作串表示与 `net/action_token_codec.py` 的
规范动作对齐，时间基准 `bot.time.age`（20 tick/秒）。原理、动作字段与运行前置见
`mineflayer_actions/AGENTS.md`，从零搭建服务器到采集的完整流程见
`mineflayer_actions/SETUP.md`，参考输出见 `mineflayer_actions/sample/session_actions.json`。

该子包还能在**每个动作节点截取观测帧**（`frame_capturer.js` 无头渲染），产出
observation(t)→action(t) 的图文数据集。完整示例 `survival_mining.js` 在**生存模式下从零**
走完一条真实挖矿链（走路→砍树→合成→放工作台→合成木镐→用镐挖石头），每个关键节点
配一张观测图与其后动作（含真实挖掘时长，木镐挖石头 ~20–40 tick 且掉落圆石）；参考样本见
`mineflayer_actions/sample/mining/`（14 节点，`gallery.md` 可直接浏览图文对照）。

该子包是 Node.js 工具链（依赖见其 `package.json`），产物（server.jar、世界、
node_modules、采集 JSON）均为运行期数据，不入库。
