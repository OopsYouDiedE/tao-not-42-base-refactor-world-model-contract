# 安装

安装、系统依赖和安装后验证全部由一键脚本完成。项目源码不包含环境检查与鉴权检查模块。

## GPU Linux 默认环境

带 GPU 的 Ubuntu Linux 使用一条命令安装完整 JDK 21、CraftGround 构建依赖、Xvfb 和全部 Python 依赖：

```bash
bash scripts/bootstrap_gpu_craftground.sh
```

脚本使用 `uv pip install --system -e '.[cuda,craftground,dev]'` 完成一次 Python 依赖解析，随后校验
JDK 21、`import craftground` 和真实 CUDA 计算。完整 JDK 不可替换为 headless 变体，因为 CraftGround
原生库通过 CMake 查找 JNI AWT。Xvfb 为无桌面的远程服务器提供真实 Minecraft 客户端所需的 X11
显示；它不替代 CraftGround 环境执行。

## 通用安装

不需要 CraftGround 时使用通用脚本。默认安装仓库中的锁定直接依赖：

```bash
bash scripts/bootstrap.sh
```

需要检查兼容范围内最新版本时使用：

```bash
bash scripts/bootstrap.sh --latest
```

脚本按已安装 PyTorch 的真实 CUDA 可用性自动选择 CPU 或 CUDA，也可以显式指定：

```bash
bash scripts/bootstrap.sh --accelerator cpu
bash scripts/bootstrap.sh --accelerator cuda
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

## 鉴权与配置

项目源码不检查也不读取本地鉴权状态。按需自行登录官方客户端：

```bash
gh auth login
hf auth login
```

公开数据集下载不要求 Hugging Face 登录；私有数据访问和发布必须使用具有相应权限的真实身份。登录
成功不构成上传授权；只有任务明确提供目标仓库、可见性和可用凭证时，才允许通过项目正式发布入口
上传。CI 和远程服务器可以按官方客户端合同注入 `GH_TOKEN` 或 `HF_TOKEN`；项目代码不主动读取、
打印或持久化这些变量。

教师模型 API 使用根目录 `.env` 或进程环境变量：

```dotenv
TEACHER_BACKEND=openai-compatible
TEACHER_API_URL=https://example.invalid/v1
TEACHER_API_KEY=replace-with-real-token
TEACHER_MODEL=model-name
TEACHER_TIMEOUT_SECONDS=240
```

配置优先级为显式非敏感参数、已有进程环境变量、`.env`、非敏感默认值。`.env` 默认不得覆盖已有
环境变量。Token 不得作为命令参数，不得写入日志、`runs/`、测试数据或 Git。
