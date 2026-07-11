# 安装指南(2026-07-10 重写)

> 旧版 611 行 uv 全模块指南(含已退役的 crafter/ppo-ad/dreamer 训练线)在 git 历史
> commit `1a29855` 及之前。本版只覆盖**当前 GRPO-Pixel 主线**实际需要的东西。
> Python >= 3.11;包管理推荐 uv。

## 1. 核心安装

```bash
# 自动检测平台(Colab / 本机 / 服务器),按需加系统依赖
python install_env.py

# 或手动
pip install -e .          # 仅核心依赖
uv pip install -e .[dev]  # 加开发工具
```

⚠️ pyproject 里的 `[crafter]` / `[ppo-ad]` / `[dreamer]` 可选组是**遗留组**——
对应训练代码已于 2026-07-10 删除(net/dreamer*、net/bc、net/ppo_ad、train/crafter),
勿再安装。

## 2. 当前主线的三个运行时依赖

1. **CraftGround 环境**(真 Minecraft Java 版):`obs["rgb"]` 取 RGB;headless 用
   Xvfb / 软渲染,根目录 `xorg.conf.headless` 备用。Colab 简单配置可跑。
   **裸机首次构建的两个坑(2026-07-11 L4 实测)**:Minecraft 1.21 要 **Java 21**
   且必须**完整版** `openjdk-21-jdk`——headless 版缺 AWT,cmake `FindJNI` 报
   `missing: AWT` 直接构建失败;外加 GL 开发库(libgl1-mesa-dev/libglew-dev/
   libglu1-mesa-dev/libglfw3-dev,install_env.py 的清单)。启动带
   `JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`。首次构建约 3 分钟,
   首次 reset 全程 ~166s。
2. **Omni 慢塔(NVFP4,本地 vLLM)**:启动脚本 `tests/serve_omni_nvfp4.sh`,
   四个 sm_120(RTX 5090)问题的修复已内联进脚本;实测口径见
   `knowledge/conclusion_omni_nvfp4_5090.md`(权重 21.5GiB / TTFT 0.154s)。
3. **Haiku 判官**:`claude` CLI(`claude -p --model haiku`),判官读图依赖 Read 权限,
   图片路径必须在工作区内(见 `docs/next_session.md §6` 的 fallback 陷阱)。

另:MiniLM 句向量(`sentence-transformers/all-MiniLM-L6-v2`)随核心依赖自动可用;
DINOv3 权重 gated,需 HF token(见 `net/backbone.py` 与 `utils/io.py` 的 HF_TOKEN 说明),
无 token 时降级 dinov2 开放权重。(YOLOE 已废弃删码,sm_120 NMS 绕过随之无关。)

## 3. 冒烟自检

```bash
python train/craftground/grpo_pixel.py --smoke   # 链路冒烟(groups=1/ticks=120)
```

CUDA/AMP 相关改动务必跑 micro CUDA 冒烟,单测全绿不代表 GPU 路径正常
(教训见记忆 cuda-amp-smoke 批次)。
