"""HuggingFace token 双重解析:Colab Secret(云) / .env(本地).

设计动机:DINOv3 等 gated 模型下载需要 token,但 token 的来源随环境而异——
  - Colab notebook:用 Google 的 Secret(左侧🔑面板,名为 HF_TOKEN),`google.colab.userdata`
    读取;不落盘、不进 notebook,适合分享的云环境。
  - 本地 / 无 Google 环境:用仓库根的 .env 文件(HF_TOKEN=...),已被 .gitignore 忽略。

`get_hf_token()` 按以下优先级解析并把结果写回 os.environ['HF_TOKEN']/
['HUGGING_FACE_HUB_TOKEN'],使 transformers / huggingface_hub 后续调用自动带上:

  1. 显式环境变量 HF_TOKEN / HUGGING_FACE_HUB_TOKEN(调用方已手动设置 ⇒ 最高优先)
  2. Colab Secret(google.colab.userdata)——仅 Colab 可用
  3. 仓库根 .env 文件
  4. huggingface_hub 已登录的缓存 token(`hf auth login` 留下的)
"""
import os
from pathlib import Path

__all__ = ["get_hf_token"]

_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


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
    """解析 HuggingFace token(见模块 docstring 的优先级)。找不到返回 None。

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
