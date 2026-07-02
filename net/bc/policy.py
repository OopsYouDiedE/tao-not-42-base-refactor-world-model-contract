"""VPT 行为克隆策略:冻结视觉骨干 + 因果时序 Transformer + 相机/按键动作头。

结构(自回归策略,预测 a_t | o_{≤t}, a_{<t}):
    帧特征(骨干 CLS,冻结) ──proj──┐
    上一步动作 a_{t-1} ──embed──── ⊕ + 位置编码 → [MHABlock(causal) + FFN] × L
                                              → cam_head  [B,T,2,camera_bins]
                                              → key_head  [B,T,n_keys]
相机头输出 mu-law 分箱 logits(CE 监督;MSE 回归下"恒预测 0"是平凡解,
见 train/minecraft/vpt_action.py 注释),按键头输出独立二值 logits(BCE)。
损失与领域常量在 train/ 侧,本模块只有结构。
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from blocks.attention import MHABlock
from net.backbone import build_backbone
from net.bc.config import BCConfig


class _FFN(nn.Module):
    """Pre-LN 前馈残差块: x + W2(GELU(W1(LN(x))))。[B,L,d]→[B,L,d]。"""

    def __init__(self, d: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, mult * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mult * d, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class BCPolicy(nn.Module):
    """行为克隆策略(骨干冻结,时序骨干与动作头可训练)。

    Args:
        cfg: BCConfig 结构超参。
        injected_backbone: 依赖注入的 mock 骨干(仅 tests/,须自带 .embed_dim)。

    Shapes:
        encode_frames: img [B,T,3,H,W] float32 [0,1] → feats [B,T,enc_dim](no_grad)
        forward: feats [B,T,enc_dim], prev_action [B,T,action_dim]
                 → cam_logits [B,T,2,camera_bins], key_logits [B,T,n_keys]
    """

    def __init__(self, cfg: BCConfig, injected_backbone=None):
        super().__init__()
        self.cfg = cfg
        module, _patch, enc_dim, _n_reg, self.backbone_kind = build_backbone(
            cfg.backbone, injected=injected_backbone)
        self.backbone = module.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.enc_dim = enc_dim

        # DINO 系骨干的 ImageNet 归一化常数(骨干预处理约定,非领域常量)
        self.register_buffer("px_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("px_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        d = cfg.d
        self.feat_proj = nn.Linear(enc_dim, d)
        self.act_embed = nn.Linear(cfg.action_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        blocks: List[nn.Module] = []
        for _ in range(cfg.layers):
            blocks.append(MHABlock(d, heads=cfg.heads, causal=True, dropout=cfg.dropout))
            blocks.append(_FFN(d, dropout=cfg.dropout))
        self.trunk = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(d)
        self.cam_head = nn.Linear(d, cfg.n_mouse * cfg.camera_bins)
        self.key_head = nn.Linear(d, cfg.action_dim - cfg.n_mouse)

    @torch.no_grad()
    def encode_frames(self, img: torch.Tensor) -> torch.Tensor:
        """冻结骨干抽帧特征(CLS token)。img [B,T,3,H,W] [0,1] → [B,T,enc_dim]。"""
        B, T = img.shape[:2]
        flat = img.flatten(0, 1)
        if self.backbone_kind == "injected":
            feats = self.backbone(flat)
        else:
            flat = (flat - self.px_mean) / self.px_std
            feats = self.backbone(pixel_values=flat).last_hidden_state[:, 0]
        return feats.view(B, T, self.enc_dim)

    def forward(
        self, feats: torch.Tensor, prev_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """feats [B,T,enc_dim], prev_action [B,T,action_dim] → (cam_logits, key_logits)。"""
        B, T = feats.shape[:2]
        if T > self.cfg.max_len:
            raise ValueError(f"seq_len {T} 超过 max_len {self.cfg.max_len}")
        x = self.feat_proj(feats) + self.act_embed(prev_action) + self.pos[:, :T]
        for blk in self.trunk:
            x = blk(x)
        x = self.out_norm(x)
        cam = self.cam_head(x).view(B, T, self.cfg.n_mouse, self.cfg.camera_bins)
        key = self.key_head(x)
        return cam, key


def build_bc_policy(cfg: Optional[BCConfig] = None, injected_backbone=None) -> BCPolicy:
    """工厂:按 BCConfig 构建 BCPolicy(缺省配置=默认结构)。"""
    return BCPolicy(cfg or BCConfig(), injected_backbone=injected_backbone)
