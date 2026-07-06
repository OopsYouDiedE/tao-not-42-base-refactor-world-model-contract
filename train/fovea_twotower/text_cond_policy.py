# -*- coding: utf-8 -*-
"""文字指示条件化快头(ReST/RAFT 的策略)。

BCPolicy + 一个文本条件头:MiniLM 句向量(384)──Linear→d 作条件项加进输入,
与帧特征、上一步动作、位置编码相加(与 C1 指令总线同构,只是快头形态=冻结
DINOv3 CLS + 因果 Transformer,非慢塔 ActionTower)。

text_embed 初始化为零 → 载入 ftt_c2bc 后,策略起手 ≈ 忽略指令、照常"看起来正确
地挖"(用户"初步训练做看起来正确的动作")。ReST 逐轮把 判优优胜轨迹 + 其指令
一起 BC 回灌,才把"指令→行为"的服从装进来。
"""
import torch
import torch.nn as nn

from net.bc.policy import BCPolicy


class TextCondPolicy(BCPolicy):
    """BCPolicy + 文本条件。forward(feats, prev_action, text_emb)。

    text_emb: [B,384] L2-归一化 MiniLM 句向量(每条轨迹一个指令,整段同一条件)。
    """

    def __init__(self, cfg, injected_backbone=None, text_dim=384):
        super().__init__(cfg, injected_backbone)
        self.text_embed = nn.Linear(text_dim, cfg.d)
        nn.init.zeros_(self.text_embed.weight)      # 起手忽略指令 → 载 c2bc 即纯挖策略
        nn.init.zeros_(self.text_embed.bias)

    def forward(self, feats, prev_action, text_emb):
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
