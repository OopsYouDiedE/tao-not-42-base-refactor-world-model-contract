# 安装

带 GPU 的 Ubuntu Linux 默认环境使用一条命令安装完整 JDK 21、CraftGround 构建依赖、Xvfb 和全部
Python 依赖：

```bash
bash scripts/bootstrap_gpu_craftground.sh
```

脚本使用 `uv pip install --system -e '.[cuda,craftground,dev]'` 完成一次 Python 依赖解析。完整 JDK
不可替换为 headless 变体，因为 CraftGround 原生库通过 CMake 查找 JNI AWT。Xvfb 为无桌面的远程
服务器提供真实 Minecraft 客户端所需的 X11 显示；它不替代 CraftGround 环境执行。

项目提供两种版本策略。默认使用仓库中的锁定直接依赖：

```bash
python scripts/bootstrap.py
```

需要检查兼容范围内最新版本时使用：

```bash
python scripts/bootstrap.py --latest
```

安装器自动根据 PyTorch 的真实 CUDA 可用性选择 CPU 或 CUDA。也可以显式指定：

```bash
python scripts/bootstrap.py --accelerator cpu
python scripts/bootstrap.py --accelerator cuda
```

CPU 安装从 PyTorch 官方 CPU 索引安装 `torch`，且不安装 Unsloth、Flash Attention、xFormers、CUDA
runtime 或其他 GPU 专用包。CUDA 安装使用 `requirements/locked-cuda.txt` 中的候选基线。该候选基线
必须在带 GPU 的 Linux 远程服务器完成真实行为克隆、相对优势训练和本地模型推理后，才能认定为已
验证锁定环境。

CUDA 候选基线使用 PyTorch 2.11、Transformers 5.5 和 TRL 0.24。Unsloth 2026.8 要求
PyTorch `<2.12`、Transformers `<=5.5`，且其数据集依赖与 TRL 1.x 不兼容；更新候选版本时必须先
使用 pip 完整解析全部锁定依赖，不能只验证单个包的版本范围。

当前锁文件固定直接依赖，尚未包含完整传递依赖哈希。权威 GPU Linux 验证完成后，应在对应环境生成
完整带哈希锁文件并提交仓库。
