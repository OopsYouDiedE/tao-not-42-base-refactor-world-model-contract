# solaris 环境规则

本文件在 `rl_training_environments/solaris/` 范围内追加生效，叠加仓库根 `AGENTS.md`。

## 1. 组成与 vendored 边界

- `engine/` 是 **vendored 的第三方 solaris-engine controller 渲染路**（上游 Apache-2.0，
  见 `engine/LICENSE`），只搬了渲染验收所需的 `controller/` + `package.json`。
- **不搬入**上游的 camera（GPU/NVENC 路）、spectator、Docker 编排（orchestrate /
  generate_compose）、postprocess 数据管线——它们与渲染验收无关。需要时从上游按需引入。
- `engine/node_modules/` 靠 `npm install` 重建，**不入库**（同 godot 只 vendored 源码）。
- `package.json` 有 3 个 github fork 依赖（mineflayer / mineflayer-pathfinder /
  prismarine-viewer-colalab），改依赖版本前须确认 fork 可达。

## 2. 渲染补丁纪律

- `viewer_patches/` 是对 `npm install` 出来的 `node_modules/prismarine-viewer-colalab`
  的修复补丁（Minecraft 1.18+ 负 y 世界地形渲染），以 `.patch` + 幂等 `apply.sh` 固化，
  **不把第三方源 vendored 进来**。
- 改补丁时同步更新 `viewer_patches/README.md` 与仓库根 `SETUP_WSL_SMOKE.md` 第 7 节。
- 三个补丁修 `worker.js` / `worldrenderer.js` / `models.js`，任何一处改动都要重跑
  `apply.sh` 幂等验证（从上游原始态可干净应用）。

## 3. 版本对齐铁律

- server、bot、viewer 三者必须同为 **Minecraft 1.21.4**。viewer 的 `getVersion("1.21")`
  会归一到 1.21.4；若 server 是 1.21.0 则 section palette 错配 → 地形竖刺畸变。
- 验收判据用 **Canny 边缘密度 edges%**（正常地形 9-13%，空/畸变画面 ~0.1%），
  **不得**用"帧非黑 mean>1"——空画面的天空色也非黑，会假阳性。

## 4. 动作契约

- 逐帧动作是 22 维契约：`forward/back/left/right/jump/sneak/sprint` +
  `camera:[Δyaw,Δpitch]` + `attack/use/mount/dismount/place_block/place_entity/mine` +
  `hotbar.1-9`，`frame_count` 与 mp4 帧严格 1:1。
- `acceptance_boundary_shots.py` 按此契约逐帧检测 `False↔True` 跳变抽动作起止帧，
  修改动作字段时同步该脚本的边界检测。

## 5. 产物边界

- 录像 mp4、逐帧 json、日志属运行期数据，落 gitignore 的 `runs/` 或 `engine/output*/`，
  不入库。`sample/acceptance/` 是固化的少量验收样本（同 mineflayer sample 先例），可入库。
