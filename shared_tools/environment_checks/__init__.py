"""不修改系统状态的环境事实检查。"""

from .checks import CheckResult, check_environment, detect_accelerator

__all__ = ["CheckResult", "check_environment", "detect_accelerator"]
