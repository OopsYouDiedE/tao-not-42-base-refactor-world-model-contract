# AI 助手活动日志

> 按用户要求记录"助手做了什么"的操作流水（含环境配置与实验过程），最新条目在最下方。
> 分析结论仍按规范沉到 `knowledge/`，此处只记过程与事实。

## 2026-07-02（Colab L4，会话 1）

### 环境配置
- `python install_env.py --dreamer --ppo-ad --minecraft --dev`（Colab demo 标准组合 + minecraft）。
  - **故障 1**：headless 系统依赖的 `apt-get install` 在 `keyboard-configuration` 包弹出交互式
    debconf 键盘布局询问，挂起约 20 分钟。处置：杀掉进程树，`DEBIAN_FRONTEND=noninteractive
    dpkg --configure -a` 后重装系统依赖成功。**教训：install_env.py 的 apt 调用应设
    `DEBIAN_FRONTEND=noninteractive`（待修）。**
  - **故障 2**：`minecraft` extra 的 `minerl==0.4.4` 依赖 gym<0.20，gym 0.19 与新 setuptools
    不兼容（`extras_require` schema 错误），uv 构建失败。处置：离线 Dreamer4 管线不 import
    minerl，改装 `.[crafter,dev,dreamer,headless,ppo-ad]` + pillow，跳过 minerl。
  - 冒烟：`from net.dreamer4 import WorldModel` 与 `import crafter` 通过。
- 站点包里存在第三方 `tests` 包，遮蔽仓库 `tests/` 命名空间包 ⇒ `python -m tests.download_vpt_data`
  失效；改用 `PYTHONPATH=/content/repo python tests/download_vpt_data.py` 直跑。
- 设备监控：`scripts/sys_monitor.py --interval 5 --csv runs/logs/sys_monitor.csv` 已后台常驻。
- 定期保全：会话内每 10 分钟自动 commit+push 一次（防 Colab 中断丢工作）。

### 数据
- `tests/download_vpt_data.py --index find-cave-Jul-28 --n 32 --out runs/data/vpt_findcave`
  后台下载中（BASALT find-cave 承包商数据，与上一轮结论文档同源同规模）。
  首 2 段标定：|dx| p95=58 ⇒ 建议 camera_scale≈33（全量下完后按汇总分位数定，上轮为 29）。

### 实验计划（进行中）
- 目标：对比 e8c8904 引入的修改方式的效果，四口径评估（psnr_gen−persist、EV(Δz)、IG、
  8 步开环 rollout 优势）。
- 矩阵（统一预算 4000 步 / seq 16 / bf16 / seed 42，26M 结构，batch 按 GPU 打满调）：
  - A 基线（对齐修复后，motion_sample=1，无 delta_weight）
  - B `--motion_sample 4`
  - C `--delta_weight`
  - D `--motion_sample 4 --delta_weight`
- 结果与判定将写入 `knowledge/conclusion_minecraft_dreamer4_run.md`。
