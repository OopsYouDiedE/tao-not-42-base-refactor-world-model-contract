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


class CondPolicy(BCPolicy):
    """BCPolicy + 标量回报条件(return-conditioned 快头)。

    forward(feats, prev_action, ret):在输入相加处多一项 ret_embed(标量回报→d),
    其余(骨干/头/时序干)全复用父类。用于"用信号操纵执行能力"的条件化 BC。
    """

    def __init__(self, cfg: BCConfig, injected_backbone=None):
        super().__init__(cfg, injected_backbone)
        self.ret_embed = nn.Linear(1, cfg.d)

    def forward(self, feats, prev_action, ret):
        """feats [B,T,enc], prev_action [B,T,A], ret [B] → (cam_logits, key_logits)。"""
        B, T = feats.shape[:2]
        r = self.ret_embed(ret.view(B, 1, 1).expand(B, T, 1).to(feats.dtype))
        x = self.feat_proj(feats) + self.act_embed(prev_action) + r + self.pos[:, :T]
        for blk in self.trunk:
            x = blk(x)
        x = self.out_norm(x)
        cam = self.cam_head(x).view(B, T, self.cfg.n_mouse, self.cfg.camera_bins)
        return cam, self.key_head(x)


class TextCondPolicy(BCPolicy):
    """BCPolicy + 文本指令条件(ReST/RAFT 的策略)。

    text_embed 初始化为零 → 载入纯挖 BC 权重后策略起手忽略指令、照常执行;
    ReST 逐轮把判优优胜轨迹 + 其指令一起 BC 回灌,才把"指令→行为"的服从装进来。
    forward(feats, prev_action, text_emb):text_emb [B,text_dim] 每条轨迹一个指令向量。
    """

    def __init__(self, cfg: BCConfig, injected_backbone=None, text_dim: int = 384):
        super().__init__(cfg, injected_backbone)
        self.text_embed = nn.Linear(text_dim, cfg.d)
        nn.init.zeros_(self.text_embed.weight)      # 起手忽略指令 → 载 c2bc 即纯挖策略
        nn.init.zeros_(self.text_embed.bias)

    def forward(self, feats, prev_action, text_emb):
        """feats [B,T,enc], prev_action [B,T,A], text_emb [B,text_dim] → (cam, key)。"""
        B, T = feats.shape[:2]
        if T > self.cfg.max_len:
            raise ValueError(f"seq_len {T} 超过 max_len {self.cfg.max_len}")
        c = self.text_embed(text_emb.to(feats.dtype)).view(B, 1, self.cfg.d)
        x = self.feat_proj(feats) + self.act_embed(prev_action) + c + self.pos[:, :T]
        for blk in self.trunk:
            x = blk(x)
        x = self.out_norm(x)
        cam = self.cam_head(x).view(B, T, self.cfg.n_mouse, self.cfg.camera_bins)
        return cam, self.key_head(x)

    def load_c2bc(self, ckpt_path, device, strict_body=True):
        """载入纯挖 BC 权重(ftt_c2bc);text_embed 保持零初始化。返回 (missing, unexpected)。"""
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ck.get("policy", ck.get("model", ck))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        body_missing = [m for m in missing
                        if not m.startswith("backbone.") and not m.startswith("text_embed.")]
        if strict_body:
            assert not body_missing, f"非 backbone/text_embed 的缺失键: {body_missing[:8]}"
        assert not unexpected, f"意外键: {unexpected[:8]}"
        return missing, unexpected
