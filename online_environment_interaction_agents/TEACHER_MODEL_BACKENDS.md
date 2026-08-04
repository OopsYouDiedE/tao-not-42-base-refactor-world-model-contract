# 教师模型轨迹生成后端

教师模型模块把最新观察转换为一段 `standard-input-action/v1` 动作。环境执行器负责执行动作、推进 tick、生成下一次观察、判断终止并把事实写入完整轨迹。模块不伪造环境结果，也不把模型计划直接视为已执行轨迹。

## 候选方案

| 方案 | 优点 | 局限 | 适用位置 |
| --- | --- | --- | --- |
| Codex CLI | 原生支持图片路径；调用隔离；本地调试方便；无需在项目中管理 HTTP 协议差异 | 启动进程开销较大；吞吐和并发受 CLI 限制；CLI 版本变化会影响参数；令牌用量不一定完整 | 研发基线、少量高价值轨迹、人工复现 |
| Claude CLI | 可复用本机 Claude 登录；非交互 JSON 输出便于审计；可通过只读工具读取观察图片 | 图片依赖 CLI 的 `Read` 工具；进程开销较大；会话和工具行为比纯 API 多一层；批量调度成本高 | Claude 能力对照、少量轨迹验证 |
| OpenAI Python/兼容 API | 长连接和并发调度容易；返回请求 ID 与用量；容器和远程任务稳定；便于限流、重试与监控 | 需要显式管理密钥、超时、限流和兼容差异；不同服务对图像消息与参数支持程度不同 | 正式批量生产、数据工厂、服务化执行 |

本实现以统一 `TeacherBackend` 隔离 CLI 与 API 调用差异，并由正式的
`TeacherTrajectoryExecutor` 完成协议解析、动作编译、环境执行和轨迹审计。项目不再保留一套仅生成并校验动作、却不执行环境的并行生成器。

## 建议

正式批量生成使用 `OpenAICompatibleBackend`。CLI 后端保留为可复现的研发基线和供应商能力对照。API 后端使用 Python 标准库实现 OpenAI Chat Completions 兼容请求，不额外引入仅用于传输的第三方客户端。

## API 示例

```python
from pathlib import Path

from online_environment_interaction_agents import (
    OpenAICompatibleBackend,
    OpenAICompatibleConfig,
    TeacherRequest,
    TeacherTrajectoryExecutor,
)

backend = OpenAICompatibleBackend(OpenAICompatibleConfig.from_environment())
executor = TeacherTrajectoryExecutor(environment, backend)
step = executor.execute_generation(
    TeacherRequest(
        system_prompt=Path(
            "online_environment_interaction_agents/TRAJECTORY_GENERATION_PROMPT.md"
        ).read_text(encoding="utf-8"),
        task_context="<trajectory_task>...</trajectory_task>",
        step_context="<trajectory_step>...</trajectory_step>",
        observation_paths=(Path("latest.png"),),
    ),
    observation=latest_observation,
    info=latest_info,
    remaining_action_ticks=32,
)
```

配置使用 `TEACHER_API_URL`、`TEACHER_API_KEY` 和 `TEACHER_MODEL`。`TEACHER_API_URL` 可以是 `https://host/v1`，也可以是完整的 `.../chat/completions` 地址。密钥只进入 Authorization 请求头，不写入审计记录。

四分支入口额外使用 `TEACHER_BACKEND` 选择 `openai-api`、`anthropic-api`、`codex-cli` 或
`claude-cli`。四个 arm 固定使用同一选择和同一模型，从而构成同策略多分支样本。跨供应商或 CLI/API
对照应作为独立能力评测运行，不能直接作为组内相对优势样本。CLI 可执行文件通过
`TEACHER_CLI_EXECUTABLE` 提供；入口不读取任何工具私有凭据文件或固定用户路径。

## CLI 示例

```python
from online_environment_interaction_agents import CLIConfig, CodexCLIBackend

backend = CodexCLIBackend(CLIConfig(model="gpt-5.6-sol", executable="codex"))
```

Claude 对照后端使用：

```python
from online_environment_interaction_agents import CLIConfig, ClaudeCLIBackend

backend = ClaudeCLIBackend(CLIConfig(model="claude-model-name", executable="claude"))
```

CLI 调用均使用参数数组和 `shell=False`。Codex 运行在临时只读、无持久会话环境中。Claude 只开放 `Read` 工具，用于读取显式列出的观察图片。

## 轨迹循环

| 顺序 | 责任方 | 输出 |
| --- | --- | --- |
| 1 | 环境执行器 | 最新观察、状态、剩余预算和终止状态 |
| 2 | 教师后端 | 一段待解析的标准动作协议文本 |
| 3 | 教师轨迹执行器 | 严格解析并实际执行 tick，记录奖励、异常和新观察 |
| 4 | 教师轨迹执行器 | 导出请求、原始输出、编译动作和环境事实 |
| 5 | 调度器 | 未终止时构造下一轮请求 |

轨迹执行器拒绝协议外说明文字和超预算动作。它不会把计划动作写成已执行事实；上层调度器可以把失败原因加入下一轮提示词，并采用有界重试。
