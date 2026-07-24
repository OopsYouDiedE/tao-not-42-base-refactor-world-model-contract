# solaris 渲染器地形修复补丁

solaris headless 渲染路依赖的 vendored `prismarine-viewer-colalab`（three.js + node-canvas-webgl +
headless-gl）对 **Minecraft 1.18+ 负 y 世界**（地面 y=-60，世界高度 -64..320）有多处旧假设，
导致地形完全不渲染或渲染成竖刺。本目录固化了修复补丁，让别人无需重新排障即可复现正常渲染。

> prismarine-viewer-colalab 是 `engine/` 靠 `npm install` 装的 github fork 依赖（node_modules
> 不入库）。这些补丁打在它装出来的 `node_modules/prismarine-viewer-colalab` 副本上，因此以
> patch 形式固化，而非把第三方源 vendored 进来。

## 三个补丁

| 补丁 | 目标文件 | 问题 → 修复 |
|---|---|---|
| `worker.js.patch` | `viewer/lib/worker.js` | section 数组用裸 `Math.floor(y/16)` 索引，现代 prismarine-chunk 的 sections 是从 `chunk.minY`(-64) 起 0-based，负 y 落负索引=undefined → 地面 section 被当空跳过。改为 `Math.floor((y - (chunk.minY ?? 0)) / 16)`。 |
| `worldrenderer.js.patch` | `viewer/lib/worldrenderer.js` | `addColumn`/`removeColumn` 用 `for (y=0; y<256)` 标记 dirty section，漏掉负 y 地面。改为 `-64..320`。 |
| `models.js.patch` | `viewer/lib/models.js` | cullface 的 `neighbor.position.y < 0`（旧世界"底部以下当空气"假设）在负 y 世界短路了正常地形的面剔除。改为 `< -64`。 |

## 应用

```bash
bash apply.sh <prismarine-viewer-colalab 包根目录>
# 例(engine/ 下 npm install 后):
#   bash apply.sh ../engine/node_modules/prismarine-viewer-colalab
```

脚本幂等（已打过会跳过），打完重启 viewer 进程生效。

## 还有一步：版本对齐（不在补丁内）

打完三个补丁地形能生成几何，但若 **server 版本与 viewer 解码版本不一致**仍会呈竖刺畸变。
根因：viewer 的 `getVersion("1.21")` 归一到 **1.21.4**（supportedVersions 无裸 "1.21"），
用 1.21.4 的 section palette 解 1.21.0 的 chunk 数据 → 每方块画满六面。

**必须让 server、bot、viewer 三者版本一致**：用 PaperMC **1.21.4** server，bot 连时
`--mc_version 1.21.4`。实测对齐后渲染出正常 superflat 草地。详见仓库根 `SETUP_WSL_SMOKE.md` 第 7 节。

验收样本见 `../sample/acceptance/`（真实草地画面 + 动作起止截图）。
