"""跨模块共用的基础设施。不反向依赖任何具体业务模块。"""

from .configuration import EnvironmentConfigurationError, load_env_file, require_env

__all__ = [
    "EnvironmentConfigurationError",
    "load_env_file",
    "require_env",
]
