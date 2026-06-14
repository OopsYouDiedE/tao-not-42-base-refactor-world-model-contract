"""冻结视觉骨干的加载(HuggingFace transformers)。

对外接口:
    load_backbone(kind, repo_override=None) — 加载冻结 DINOv2/v3,
        返回 (module, patch_size, enc_dim, n_register)。

视觉骨干统一走 HuggingFace transformers(torch.hub 路径已废弃):gated 权重(dinov3)
由 utils.hf_token 解析的 token 自动鉴权,不再手传 URL/.pth。
离线/无网络的管线冒烟用依赖注入的 mock 骨干(`MinecraftWorldModel(backbone=...)`,
见 tests/);按 AGENTS §2,生产 net/ 不提供任何 mock 骨干。
"""
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
                "或仓库根 .env(HF_TOKEN=...)——见 utils/hf_token.py;无 token 改 --encoder dinov2(开放权重)。"
                if kind == "dinov3" else "需要网络访问 HuggingFace Hub(首次下载后本地缓存)。")
        raise RuntimeError(f"{kind} 从 HF 加载失败({repo}:{ex})。{hint}") from ex
    cfg = model.config
    n_reg = getattr(cfg, "num_register_tokens", 0) or 0
    return model, cfg.patch_size, cfg.hidden_size, n_reg
