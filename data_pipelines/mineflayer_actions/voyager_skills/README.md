# Voyager JS 技能原语

从 [MineDojo/Voyager](https://github.com/MineDojo/Voyager)（MIT，见 `LICENSE`）搬运的
mineflayer 技能原语，上游 commit `55e45a880755d0c8c66ca7fb5fe7962ac8974f89`。这些是
Voyager 提供给 LLM 生成代码时可直接调用的高层技能，与本子包"主动执行动作并记录时长"
的目标同层，用作动作库参考与复用。

## 目录

- `control_primitives/` —— 技能的**可执行实现**（12 个 `.js`）。
- `control_primitives_context/` —— 同名技能的**带签名与注释版**，Voyager 用它拼进 prompt
  告诉模型有哪些技能可调；`mineflayer.js` 是给模型的 mineflayer API 速览。

## 技能一览（control_primitives/）

| 文件 | 作用 |
|---|---|
| `mineBlock.js` | 就近寻找并采集指定方块 N 个 |
| `craftItem.js` | 有/无工作台合成指定物品 N 次 |
| `craftHelper.js` | `failedCraftFeedback`：合成失败时报缺料，供 `craftItem` 调用 |
| `placeItem.js` | 在指定位置放置方块 |
| `smeltItem.js` | 用熔炉冶炼指定物品 |
| `useChest.js` | 打开箱子存取物品 |
| `killMob.js` | 攻击并击杀指定生物 |
| `shoot.js` | 用远程武器射击目标 |
| `exploreUntil.js` | 朝某方向探索直到回调条件满足 |
| `givePlacedItemBack.js` | 放置后把方块收回（避免消耗） |
| `waitForMobRemoved.js` | 等待某生物实体消失 |

## 运行时依赖（重要）

这些技能**不能孤立运行**，依赖 Voyager 运行环境注入的全局与 bot 插件：

- `mcData` —— 全局注入的 `minecraft-data`（按 bot 版本）。
- `bot.collectBlock` —— [mineflayer-collectblock](https://github.com/PrismarineJS/mineflayer-collectblock) 插件。
- `bot.pathfinder` + `GoalLookAtBlock` 等 —— [mineflayer-pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder) 插件。
- `bot.save(name)` —— Voyager 自定义的事件记录钩子（上游 `lib/` 提供，本处未搬）。
- `bot.chat(...)` —— 用作反馈/日志通道。

接入本子包的 `record_session.js` / `survival_mining.js` 前，需先安装上述插件并提供
`mcData` 与 `bot.save` 的等价实现，或改写为调用本子包 `ActionRecorder` 的记录接口。
