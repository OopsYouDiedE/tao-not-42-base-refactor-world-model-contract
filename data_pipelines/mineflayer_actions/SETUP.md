# 环境搭建与运行流程

从零跑通"启动服务器 → bot 执行动作 → 记录动作时长"的完整步骤。所有产物(server.jar、
世界存档、node_modules、输出 JSON)都属运行期数据,放在仓库外或 `runs/` 下,不入库。

## 1. 安装 Node 依赖

```bash
cd data_pipelines/mineflayer_actions
npm install          # 按 package.json 安装 mineflayer 等
```

## 2. 下载并启动真实 Java 服务器(以 1.16.5 为例)

1.16.5 是本子包验证过的版本:服务端合成/挖掘/掉落逻辑完整,且
[prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) 无头渲染能正常
出画面(1.18+ 渲染器对高版本世界会退化为空)。

```bash
# 从 Mojang version manifest 解析 1.16.5 的 server.jar 地址并下载(需自行接受 EULA)
mkdir -p runs/mineflayer_server && cd runs/mineflayer_server
# server.jar 下载地址随版本 manifest 变化, 从官方 piston-meta 获取:
#   https://launchermeta.mojang.com/mc/game/version_manifest_v2.json
#   -> 找到 1.16.5 的 url -> 其 downloads.server.url 即 server.jar
curl -o server.jar "<从 manifest 解析出的 1.16.5 server.jar url>"
printf 'eula=true\n' > eula.txt
```

`server.properties` 关键字段(其余默认即可):

```properties
online-mode=false          # 允许离线用户名的 bot 接入
level-type=default         # 普通世界(有树有矿); 需超平坦改 flat
spawn-monsters=false        # 关怪物, 避免干扰 bot
difficulty=peaceful
spawn-protection=0         # 关键: 否则出生点附近 placeBlock/dig 被服务器拒绝
server-port=25565
# 模式二选一:
#   gamemode=creative  -> record_session.js 用 creative.setInventorySlot 备料
#   gamemode=survival  -> survival_mining.js 真实采集/挖掘/掉落 (挖掘时长才有意义)
gamemode=survival
```

生存模式挖矿示例要用 `/clear` 从零清库存, 需给 bot 上 OP。写 `ops.json`
(离线 UUID = MD5v3 of `"OfflinePlayer:<用户名>"`, 示例 bot 名为 `Miner`):

```bash
node -e 'const c=require("crypto"),h=c.createHash("md5").update("OfflinePlayer:Miner").digest();
h[6]=h[6]&0x0f|0x30;h[8]=h[8]&0x3f|0x80;const x=h.toString("hex");
const u=`${x.slice(0,8)}-${x.slice(8,12)}-${x.slice(12,16)}-${x.slice(16,20)}-${x.slice(20)}`;
require("fs").writeFileSync("ops.json",JSON.stringify([{uuid:u,name:"Miner",level:4}],null,2));console.log(u)'
```

启动(1.16.5 官方要求 Java 8, 实测 Java 17 亦可运行):

```bash
java -Xmx2G -jar server.jar --nogui
# 首次启动生成世界, 等日志出现 `Done (NNs)!` 即就绪
# ops.json 在启动时读取, 改动后需重启生效
```

## 3. 运行动作采集

```bash
cd data_pipelines/mineflayer_actions
node record_session.js --host localhost --port 25565 --version 1.16.5 \
                       --output runs/mineflayer_actions/session.json
```

脚本会:连服务器 → 寻路到平坦草地 → 依次执行移动/跳跃/转视角/合成/放置/破坏 →
把带 `startTick`、`durationTicks` 的动作序列写入 `--output`。

## 4. 生存模式挖矿链示例 (obs→action 配对)

生存模式下从零走完 走路→砍树→合成→放工作台→合成木镐→挖矿, 每个节点截观测帧配动作。
依赖无头 GL 渲染, 需虚拟显示与 `node-canvas-webgl`/`prismarine-viewer`/`three`
(见 package.json 的 optionalDependencies)。用 `xvfb-run` 运行:

```bash
xvfb-run -a -s "-screen 0 1280x720x24" \
  env OUT_DIR=runs/mineflayer_actions/mining \
  node survival_mining.js
```

产出:每个动作节点一张 PNG + `manifest.json`(节点→动作配对) + `gallery.md`(图文对照)。
验证要点:最终库存应含 `cobblestone`(木镐挖石头掉圆石)、挖石头时长 ~20–40 tick;
若时长 ~180 tick 且无圆石掉落, 说明木镐没握在手上(退化成徒手挖)。

## 5. 参考样本

- `sample/session_actions.json`:6 类动作冒烟序列(9 条),验证 startTick / durationTicks
  字段结构。
- `sample/mining/`:生存挖矿链关键节点数据集(14 节点),`gallery.md` 可直接浏览
  "节点观测图 + 该节点后动作(含挖掘时长)"的对照。二者均不依赖运行期数据。

## 已知坑位

- **合成超时**:纯 JS 服务器(flying-squid)不实现服务端合成,`bot.craft` 会一直等
  `updateSlot` 直到超时。必须用真实 Java 服务器。
- **放置/破坏被拒**:`spawn-protection` 非 0 时,出生点附近放置/破坏会以
  `blockUpdate did not fire` 超时失败。设 `spawn-protection=0`。
- **placeBlock 确认事件**:即使放置成功,`blockUpdate` 确认事件在部分版本也可能超时;
  记录器已容错(保留 err),方块是否真放上应由调用方 `blockAt` 校验。
- **pathfinder 阻塞**:复杂地形下 A* 可能长时间同步搜索,阻塞事件循环甚至触发
  keepalive 掉线;对 `goto` 加 `Promise.race` 超时兜底。
