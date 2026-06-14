"""DINOv3 (HuggingFace, gated) 本地下载 + 前向冒烟测试。

下载 86M 的 ViT-B/16(facebook/dinov3-vitb16-pretrain-lvd1689m,768 维,patch=16),
跑一次真实前向,核对:patch token 形状 / register token 数 / 预处理归一化常数 /
本项目 128×128 帧下的 grid 尺寸 / GPU 显存与耗时。

token 从仓库根目录 .env 读取(HF_TOKEN=...),该文件已被 .gitignore 忽略。
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hf_token import get_hf_token


def main():
    repo = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    tok = get_hf_token()
    print(f"token loaded: {bool(tok)} | repo: {repo}")

    from transformers import AutoModel, AutoImageProcessor

    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(repo, token=tok)
    print(f"\n[processor] mean={proc.image_mean} std={proc.image_std} "
          f"size={getattr(proc, 'size', None)} rescale={getattr(proc, 'rescale_factor', None)}")

    model = AutoModel.from_pretrained(repo, token=tok).eval()
    dl = time.time() - t0
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] class={model.__class__.__name__} params={n_params/1e6:.1f}M "
          f"download+load={dl:.1f}s")
    print(f"[config] hidden={cfg.hidden_size} patch={cfg.patch_size} "
          f"layers={getattr(cfg,'num_hidden_layers',None)} "
          f"heads={getattr(cfg,'num_attention_heads',None)} "
          f"register_tokens={getattr(cfg,'num_register_tokens',None)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    n_reg = getattr(cfg, "num_register_tokens", 0) or 0

    for res in (224, 128):  # 224=官方分辨率;128=本项目 Minecraft 帧
        x = torch.rand(2, 3, res, res, device=dev)
        x = (x - torch.tensor(proc.image_mean, device=dev).view(1, 3, 1, 1)) \
            / torch.tensor(proc.image_std, device=dev).view(1, 3, 1, 1)
        if dev == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.no_grad():
            out = model(pixel_values=x)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t1) * 1000
        lhs = out.last_hidden_state                       # [B, 1+n_reg+P, hidden]
        n_patch = lhs.shape[1] - 1 - n_reg
        g = int(round(n_patch ** 0.5))
        patch_tokens = lhs[:, 1 + n_reg:, :]              # 丢 CLS + register
        vram = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0
        print(f"\n[fwd {res}x{res}] last_hidden_state={tuple(lhs.shape)} "
              f"-> patch_tokens={tuple(patch_tokens.shape)} grid={g}x{g} "
              f"(expect {res//cfg.patch_size}) | {dt:.0f}ms | peak VRAM {vram:.0f}MB")
        print(f"           patch feat: mean={patch_tokens.mean():.4f} "
              f"std={patch_tokens.std():.4f} norm/token={patch_tokens.norm(dim=-1).mean():.2f}")

    print("\nOK — DINOv3 ViT-B/16 本地可用。")


if __name__ == "__main__":
    main()
