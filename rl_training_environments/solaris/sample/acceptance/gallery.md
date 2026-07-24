# solaris 渲染器 · 动作起止截图验收样本

`acceptance_boundary_shots.py` 的固化产物：对 solaris headless 渲染路（prismarine-viewer）
输出的「逐帧动作 JSON + 录像 mp4」做后处理，在每个动作的开始/结束帧抽截图。
2026-07-25 在 WSL2 Ubuntu(RTX 3070)实测，Minecraft 1.21.4 superflat 真实渲染。

## 产物

- `episode.mp4` — mine（挖矿）episode 的全程录像（640×360）。
- `contact_sheet.png` — 8 张动作起止截图的核验拼图。
- `<序号>_<动作>_<start|end>_f<帧>.png` — 逐个动作起止帧截图，左上角烧入动作标签。
- `boundary_summary.json` — 44 帧、8 个动作边界事件，涉及动作 `camera / hotbar.2 / hotbar.3 / mine`。

## 动作边界(逐帧检测 False↔True 跳变)

| 帧 | 事件 |
|---|---|
| 17 | hotbar.2 按下(切镐) |
| 18 | camera 开始转头(cam=[0,-0.3]) / hotbar.2 抬起 |
| 26 | mine 开始挖矿 / hotbar.3 按下 |
| 27 | camera 停止 / hotbar.3 抬起 |
| 29 | mine 结束 |

截图均为真实 superflat 草地画面（平整草方块、蓝天、地平线），Canny 边缘密度 9-13%
（空画面仅 ~0.1%，据此判定地形确实渲染）。

## 复现

solaris engine controller 已 vendored 进 `../../engine/`。`engine/` 下 `npm install`
→ 应用 `../../viewer_patches/apply.sh` → 连 1.21.4 server 跑 `controller/main.js` 出
mp4 + 逐帧动作 json，再后处理：

```bash
python -m rl_training_environments.solaris.acceptance_boundary_shots \
  --json <逐帧动作.json> --mp4 <同序列录像.mp4> \
  --out runs/solaris-acceptance --contact-sheet
```

WSL 从零复现、地形负 y 修复 + 版本对齐（必须用 1.21.4 server，否则地形呈竖刺畸变）
见 `../../README.md` 与仓库根 `SETUP_WSL_SMOKE.md`。
