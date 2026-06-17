"""Minecraft 世界模型 (两步式离散词表与重构版)。

第一步：利用历史潜特征与历史动作序列自回归预测未来动作词表 Token。
第二步：利用预测出的词表 Token 与前一帧的细粒度 patch 潜表征进行 Cross-Attention 重构，还原当前帧的潜表征。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.encodings import ContinuousTimeEncoding, sinusoidal_time_encoding
from blocks.attention import PreLNAttn
from net.config import ModelConfig
from net.backbone import build_backbone
from net.dynamics import build_dynamics
from net.heads import ActionVocabHead


class MinecraftWorldModel(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.config = config
        self.d = config.d
        self.K = config.K
        self.J = config.J
        self.S = config.max_skip
        self.vocab_size = 512  # 与 ActionTokenizer/quantizer 默认词表大小一致

        # 视觉骨干
        self.backbone, self._patch, enc_dim, self._n_reg, self.encoder_kind = \
            build_backbone(config.backbone, injected=backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 特征可训练投影
        self.proj = nn.Linear(enc_dim, self.d)

        # 动作/词表嵌入
        self.vocab_embed = nn.Embedding(self.vocab_size, self.d)
        self.action_enc = nn.Linear(config.act_dim + 1, self.d)

        # 动力学核（主自回归网络）
        self.blocks = build_dynamics(config.dynamics, self.d)

        # 离散词表分类头
        self.heads = ActionVocabHead(self.d, self.vocab_size)

        # 动作效应 Cross-Attention 重建层（还原当前帧潜向量）
        self.recon_attn = PreLNAttn(self.d, heads=4, mode="cross")
        self.recon_proj = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, self.d)
        )

        # 时间与位置嵌入
        self.dt_enc = ContinuousTimeEncoding(self.d)
        self.act_pos = nn.Parameter(torch.randn(1, self.S, self.d) * 0.02)
        
        # Placeholders
        self.text_placeholder = nn.Parameter(torch.randn(1, 1, self.d))
        self.task_proj = nn.Linear(384, self.d)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_feats(self, img):
        if self.encoder_kind in ("dinov2", "dinov3"):
            H, W = img.shape[-2:]
            ps = self._patch
            H2, W2 = max(ps, (H // ps) * ps), max(ps, (W // ps) * ps)
            if (H2, W2) != (H, W):
                img = F.interpolate(img, size=(H2, W2), mode="bilinear", align_corners=False)
            img = (img - self._in_mean) / self._in_std
            lhs = self.backbone(pixel_values=img).last_hidden_state
            return lhs[:, 1 + self._n_reg:, :]
        return self.backbone(img)

    def encode_obs(self, img=None, feats=None):
        """在线感知编码：将图像特征投影至隐层空间。

        返回细粒度 patch 表征：[B, M, d]
        """
        if feats is None:
            feats = self.extract_feats(img)
        return self.proj(feats)

    def forward(self, z_ref, h, a_hist, a_cur, dt, t_vec, t_hist=None, hist_valid=None,
                task_emb=None, target_token_id=None):
        """两步式前向。

        z_ref: [B, M, d] 前一帧细粒度 patch 潜向量
        h: [B, 1, d] 脑内记忆状态
        a_hist: [B, J, A] 历史聚合动作
        a_cur: [B, S, A] 当前区间的动作时序（用来在第一步中辅助预测或重构）
        dt: [B] 执行步长
        target_token_id: [B] 训练时传入的 GT 动作词表 Token 索引，用以进行 Cross-Attention 特征重构训练。
        """
        B = z_ref.shape[0]
        J = a_hist.shape[1]
        
        # 1. 整理 text/dt/h token
        text_token = (self.text_placeholder.expand(B, -1, -1) if task_emb is None
                      else self.task_proj(task_emb.to(z_ref.dtype)).unsqueeze(1))
        h_token = h + sinusoidal_time_encoding(t_vec, self.d).to(h.dtype)
        dt_token = self.dt_enc(dt).to(z_ref.dtype).unsqueeze(1)

        # 2. 整理历史动作特征并注入时间信息
        if t_hist is None:
            t_hist = torch.zeros(B, J, device=z_ref.device)
        if hist_valid is None:
            hist_valid = torch.ones(B, J, device=z_ref.device, dtype=a_hist.dtype)
        ah = self.action_enc(torch.cat([a_hist, hist_valid.unsqueeze(-1)], dim=-1)) \
            + self.dt_enc(t_hist).to(z_ref.dtype).view(B, J, self.d)

        # 3. 整理当前区间的动作特征
        S = a_cur.shape[1]
        valid = (torch.arange(S, device=z_ref.device).unsqueeze(0)
                 < dt.unsqueeze(1)).to(a_cur.dtype).unsqueeze(-1)
        ac = self.action_enc(torch.cat([a_cur, valid], dim=-1)) + self.act_pos[:, :S]

        # 4. 对前一帧 patch 特征进行平均池化以提取帧级全局表征，送入 Transformer 预测
        z_global = z_ref.mean(dim=1, keepdim=True)  # [B, 1, d]

        # 拼接送入 Transformer
        X = torch.cat([z_global, text_token, h_token, dt_token, ah, ac], dim=1)
        X = self.blocks(X)

        # 提取更新后的隐状态 h_next 并预测词表 Token Logits
        out_h = X[:, 2:3, :]  # 取 h_token 对应的更新状态
        logits = self.heads(out_h).squeeze(1)  # [B, vocab_size]

        # 5. 特征重建（Cross-Attention 重构当前状态 z_t）
        if target_token_id is not None:
            token_embed = self.vocab_embed(target_token_id).unsqueeze(1)  # [B, 1, d]
        else:
            pred_token_id = logits.argmax(dim=-1)
            token_embed = self.vocab_embed(pred_token_id).unsqueeze(1)  # [B, 1, d]

        # 用上一帧的细粒度 patch 表征 z_ref (Query) 与动作 Token 的 Embedding (Key/Value) 做 Cross-Attention
        delta_z = self.recon_attn(z_ref, token_embed)
        delta_z = self.recon_proj(delta_z)
        z_recon = z_ref + delta_z  # 还原得到的当前帧 patch 潜向量

        return {
            "logits": logits,         # 预测动作词表的 Logits
            "h_next": out_h,           # 下一步的记忆状态
            "z_recon": z_recon         # 还原推断得到的当前帧潜向量预测值
        }
