"""旧导入路径的兼容转发；新代码使用 shared_tools.configuration。"""

from shared_tools.configuration import EnvironmentConfigurationError, load_env_file

__all__ = ["EnvironmentConfigurationError", "load_env_file"]
