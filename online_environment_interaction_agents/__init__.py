"""读取环境观察、生成动作并记录交互轨迹的代理。"""

from .local_vision_policy import LocalPolicyGeneration, LocalVisionPolicyBackend
from .teacher_executor import (
    ExecutedTeacherGeneration,
    TeacherTrajectoryExecutor,
    export_trajectory_markdown,
)
from .teacher_trajectory import (
    AnthropicCompatibleBackend,
    AnthropicCompatibleConfig,
    ClaudeCLIBackend,
    CLIConfig,
    CodexCLIBackend,
    GeneratedTrajectoryStep,
    OpenAICompatibleBackend,
    OpenAICompatibleConfig,
    TeacherBackend,
    TeacherDecisionEnvelope,
    TeacherModelError,
    TeacherRequest,
    TeacherResponse,
    TeacherTrajectoryGenerator,
    parse_teacher_decision,
)

__all__ = [
    "AnthropicCompatibleBackend",
    "AnthropicCompatibleConfig",
    "CLIConfig",
    "ClaudeCLIBackend",
    "CodexCLIBackend",
    "ExecutedTeacherGeneration",
    "GeneratedTrajectoryStep",
    "LocalPolicyGeneration",
    "LocalVisionPolicyBackend",
    "OpenAICompatibleBackend",
    "OpenAICompatibleConfig",
    "TeacherBackend",
    "TeacherDecisionEnvelope",
    "TeacherModelError",
    "TeacherRequest",
    "TeacherResponse",
    "TeacherTrajectoryExecutor",
    "TeacherTrajectoryGenerator",
    "export_trajectory_markdown",
    "parse_teacher_decision",
]
