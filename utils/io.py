"""合并后的基础 IO 配置与环境助手。

提供 YAML 配置加载以及 HuggingFace 鉴权 Token 获取。
"""
import os
import yaml
from pathlib import Path

__all__ = ["load_yaml", "get_hf_token"]

_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_yaml(path):
    """读 yaml 文件为 dict。缺包(pyyaml)/缺文件直接报错(AGENTS §2:不写降级)。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _from_environ():
    for k in _ENV_KEYS:
        v = os.environ.get(k)
        if v:
            return v.strip()
    return None


def _from_colab(secret_name="HF_TOKEN"):
    """Colab Secret(🔑 面板)。非 Colab 环境 import 即失败 ⇒ 返回 None。"""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        v = userdata.get(secret_name)
        return v.strip() if v else None
    except Exception:
        # SecretNotFoundError / NotebookAccessError 等:静默回退到下一来源
        return None


def _from_dotenv(start=None):
    """从调用处向上找最近的 .env(本仓库根)。只解析 HF_TOKEN 这一行,不引第三方库。"""
    here = Path(start) if start else Path(__file__).resolve()
    for d in [here, *here.parents]:
        env = d / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key.strip() in _ENV_KEYS:
                    return val.strip().strip("'\"") or None
    return None


def _from_cached_login():
    try:
        import huggingface_hub as h
        return h.get_token()
    except Exception:
        return None


def get_hf_token(colab_secret_name="HF_TOKEN", set_environ=True):
    """解析 HuggingFace token(按优先级: 环境变量 -> Colab -> .env -> hf login)。

    找不到返回 None。
    colab_secret_name: Colab Secret 面板里的条目名(默认 HF_TOKEN)。
    set_environ: 解析到后写回 os.environ,使后续 HF 库调用自动鉴权。
    """
    tok = (_from_environ()
           or _from_colab(colab_secret_name)
           or _from_dotenv()
           or _from_cached_login())
    if tok and set_environ:
        for k in _ENV_KEYS:
            os.environ[k] = tok
    return tok


if __name__ == "__main__":
    t = get_hf_token()
    if t:
        masked = f"{t[:6]}…{t[-4:]}" if len(t) > 12 else "(short)"
        print(f"HF token resolved: {masked}")
    else:
        print("HF token NOT found (set Colab Secret 'HF_TOKEN' or repo-root .env)")
