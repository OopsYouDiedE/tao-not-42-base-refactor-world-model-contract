"""冻结视觉骨干的加载与离线冒烟回退。

对外接口:
    load_backbone(kind, repo_override=None) — HF transformers 加载冻结 DINOv2/v3,
        返回 (module, patch_size, enc_dim, n_register)。
    MockDINOv2(d) — 随机冻结卷积,模拟 DINOv2 patch token;**仅无网络冒烟用**
        (CLI `--encoder mock` 显式选择,非静默降级)。

视觉骨干统一走 HuggingFace transformers(torch.hub 路径已废弃):gated 权重(dinov3)
由 utils.hf_token 解析的 token 自动鉴权,不再手传 URL/.pth。
"""
import torch
import torch.nn as nn

_HF_REPOS = {
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",  # ViT-S/16,384,patch16,4 register,gated
    "dinov2": "facebook/dinov2-small",                      # ViT-S/14,384,patch14,0 register,开放
}


def load_backbone(kind, repo_override=None):
    """HF transformers 加载冻结视觉骨干。返回 (module, patch_size, enc_dim, n_register)。

    dinov3: ViT-S/16,hidden=384,patch=16(整除 128→8×8),4 register token。权重 gated
            ——HF 接受许可证后,token 经 Colab Secret(HF_TOKEN)或仓库根 .env 注入。
    dinov2: ViT-S/14,hidden=384,patch=14(128→126 削边),0 register。权重开放,降级备选。
    repo_override: 非空时覆盖默认 repo id(可换 ViT-B 等更大变体;enc_dim 自动取 hidden_size)。
    """
    from transformers import AutoModel
    from utils.hf_token import get_hf_token
    repo = repo_override or _HF_REPOS.get(kind)
    if repo is None:
        raise ValueError(f"未知 encoder: {kind}")
    try:
        model = AutoModel.from_pretrained(repo, token=get_hf_token())
    except Exception as ex:
        hint = ("DINOv3 权重 gated:在 HF 接受许可证后,把 token 放进 Colab Secret(HF_TOKEN)"
                "或仓库根 .env(HF_TOKEN=...)——见 utils/hf_token.py;离线冒烟改 --encoder mock。"
                if kind == "dinov3" else "需要网络访问 HuggingFace Hub;离线冒烟请改 --encoder mock。")
        raise RuntimeError(f"{kind} 从 HF 加载失败({repo}:{ex})。{hint}") from ex
    cfg = model.config
    n_reg = getattr(cfg, "num_register_tokens", 0) or 0
    return model, cfg.patch_size, cfg.hidden_size, n_reg


class MockDINOv2(nn.Module):
    """随机冻结卷积,模拟 DINOv2 输出 Patch Tokens——**仅供无网络的管线冒烟测试**。

    由 CLI `--encoder mock` 显式选择(非静默降级)。注意:视觉骨干统一冻结
    (见 world_model.extract_feats),mock 即随机特征投影;真实训练用 --encoder dinov3/dinov2。
    """
    def __init__(self, d=384):
        super().__init__()
        self.embed_dim = d
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8), nn.ReLU(),
            nn.Conv2d(64, d, kernel_size=4, stride=4), nn.ReLU()
        )
    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.net(x) # [B, d, h, w]
        B, d, h, w = feat.shape
        return feat.view(B, d, h*w).transpose(1, 2) # [B, M, d]
