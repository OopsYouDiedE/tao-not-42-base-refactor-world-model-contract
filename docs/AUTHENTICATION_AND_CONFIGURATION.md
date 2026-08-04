# 鉴权与配置

本地 GitHub 和 Hugging Face 分别使用官方 CLI 登录状态：

```bash
gh auth login
hf auth login
```

项目只通过 `gh auth status` 和 `hf auth whoami` 检查状态，不读取 CLI 凭证缓存，也不执行
`gh auth token`。公开数据集下载不要求 Hugging Face 登录；私有数据访问和发布必须使用具有相应权限的
真实身份。登录成功不构成上传授权，数据集发布仍必须显式传入目标仓库、可见性和
`confirm_publish=True`。

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

安装和环境检查允许跳过本地可选鉴权：

```bash
python scripts/bootstrap.py --skip-optional-auth
python scripts/check_environment.py --skip-github-auth
python scripts/check_environment.py --skip-huggingface-auth
```

CI 和远程服务器可以按官方客户端合同注入 `GH_TOKEN` 或 `HF_TOKEN`。项目代码不主动读取、打印或
持久化这些变量。
