# CraftGround 渲染器验收样本

`acceptance_sequence.py` 的固化产物：每秒截图 + 全程录像。2026-07-25 在 WSL2
Ubuntu(RTX 3070，xvfb 软渲染)实测跑通，Minecraft 1.21 + Fabric mod 真实渲染。

## 产物

- `sequence.mp4` — 200 步动作序列的全程录像（640×360，10fps）。
- `shots/shot_<秒>s_f<帧>.png` — 按挂钟时间每秒一张截图，共 9 张，文件名带秒数与帧号。
- `summary.json` — 验收指标：帧尺寸、总时长、截图清单、非黑帧数、每帧动作 ID 与均值。

## 画面

截图是真实第一人称 Minecraft 渲染：森林生物群系（树、树叶、草地）、天空、准星、
玩家手臂与底部 HUD（生命/饥饿条、物品栏），左上角烧入 `t=秒 step=步 act=动作ID`。

## 复现

```bash
xvfb-run -a python -m rl_training_environments.craftground.acceptance_sequence \
  --steps 200 --seed 0 --seconds-per-shot 1.0 --video-fps 10 \
  --output-dir runs/craftground-acceptance
```

系统依赖（JDK21 + GL dev 库 + xvfb）与首次 Gradle 冷编译 / Minecraft asset 下载的
坑见仓库根 `SETUP_WSL_SMOKE.md` 第二部分。
