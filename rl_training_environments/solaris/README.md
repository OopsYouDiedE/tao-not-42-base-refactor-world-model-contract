# solaris 渲染环境

用 [mineflayer](https://github.com/PrismarineJS/mineflayer) 无头 bot 在真实 Java Minecraft
服务器上执行 episode，经 headless 的 [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer)
（three.js + node-canvas-webgl + headless-gl）连续渲染官方图形风格画面，输出 mp4 录像 +
逐帧 22 维动作契约 json（`forward/back/left/right/jump/sneak/sprint/camera[Δyaw,Δpitch]/
attack/use/mount/dismount/place_block/place_entity/mine/hotbar.1-9` + inventory + 位姿）。

与 CraftGround（Fabric mod）走的是两条独立渲染路：solaris 用 mineflayer 从客户端侧观测，
CraftGround 用服务端 mod。两者都产出「观测帧 + 动作」的行为数据。

## 目录

    engine/                        vendored solaris controller 渲染路(Apache-2.0，Node.js)
    engine/controller/             bot / episode-handlers / act_recorder / primitives / utils
    viewer_patches/                prismarine-viewer 负 y 地形渲染补丁 + 幂等 apply.sh
    acceptance_boundary_shots.py   逐帧动作 json + mp4 → 每个动作起止帧截图(验收后处理)
    sample/acceptance/             验收样本:动作起止截图 + 录像 + contact sheet + gallery

## 运行前置

- Node.js 18+；Java 21（跑 Minecraft server）；无头渲染需 `xvfb-run` + GL/canvas 的
  `-dev` 系统库（装机细节见仓库根 `SETUP_WSL_SMOKE.md`）。
- `engine/` 下 `npm install`（`package.json` 含 3 个 github fork 依赖，node_modules 不入库）。
- **应用地形补丁**（否则 Minecraft 1.18+ 负 y 世界地形不渲染，画面只有实体漂在虚空）：

  ```bash
  bash viewer_patches/apply.sh engine/node_modules/prismarine-viewer-colalab
  ```

- **版本对齐铁律**：server、bot、viewer 三者必须同为 **Minecraft 1.21.4**。用 1.21.0 server
  会因 viewer 把 `"1.21"` 归一到 1.21.4 解码而导致地形竖刺畸变（详见 `viewer_patches/README.md`）。

## 验收（动作起止截图 + 录像）

`engine/controller/main.js` 跑一个动作密集 episode（如 `mine`）→ act_recorder 输出 mp4 +
逐帧动作 json → 本目录脚本抽每个动作起止帧截图：

```bash
python -m rl_training_environments.solaris.acceptance_boundary_shots \
  --json <逐帧动作.json> --mp4 <同序列录像.mp4> \
  --out runs/solaris-acceptance --contact-sheet
```

产出逐个动作起止截图 + `contact_sheet.png` + `boundary_summary.json`。验收样本见
`sample/acceptance/`（真实 superflat 草地画面，动作 camera/hotbar/mine 的起止）。
WSL 从零复现（装机 / 补丁 / 版本对齐）见仓库根 `SETUP_WSL_SMOKE.md`。
