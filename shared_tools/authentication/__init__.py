"""官方 CLI 和模型 API 鉴权合同。"""

from .cli import AuthenticationStatus, check_github_authentication, check_huggingface_authentication
from .secrets import SecretValue

__all__ = [
    "AuthenticationStatus",
    "SecretValue",
    "check_github_authentication",
    "check_huggingface_authentication",
]
