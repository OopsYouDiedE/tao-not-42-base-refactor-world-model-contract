"""Minecraft 世界模型 (序列对齐的后果结构版)。

设计原则与数学推导见 knowledge/mental_world.md。核心:
  - 对齐发生在**编码空间**且是**序列↔序列**:把每帧扫成 patch token、加屏幕坐标 + 时间编码摊平成
    一个集合,用"初始帧编码 + 一段(图像+动作)token"去预测**同一个未来帧的潜向量**(数学 (1)/(2))。
  - 潜空间**因子化** z=(z_rev, z_inv):可逆相机/平移走生成元流 𝔤,不可逆事件走离散增量 𝒟(数学 (7))。
  - 目标编码取 **EMA 教师 + stop-grad**(JEPA,I8);动作只作**条件输入**,不作对齐目标。
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.encodings import ContinuousTimeEncoding, PositionalEmbed
from blocks.attention import PreLNAttn
from blocks.dynamics import GatedResidual
from blocks.regularization import StochLatent, BoundedActivation
from net.config import ModelConfig
from net.backbone import build_backbone
from net.dynamics import build_dynamics
from net.effect_tokenizer import GeneratorBank
from net.heads import EventVocabHead, AffordanceHead, SurpriseHead


class _MLP(nn.Module):
    """Pre-LN 前馈(供 GatedResidual 包裹;残差由外层增益门控)。"""
    def __init__(self, d, mult):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class _Adapter(nn.Module):
    """冻结骨干之上的可训练编码 adapter → 因子化潜 (z_rev, z_inv)。

    Linear(enc→d) → num_layers×(PreLNAttn 自注意 + GatedResidual(MLP)) → 两路头:
      z_rev: BoundedActivation('flow') 有界连续 [.,d_rev];
      z_inv: StochLatent(gaussian/categorical) 随机潜 [.,d_inv] + KL 信息瓶颈。
    """
    def __init__(self, enc_dim, d, d_rev, d_inv, cfg):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, d)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.blocks.append(PreLNAttn(d, heads=cfg.nhead, mode="self"))
            self.blocks.append(GatedResidual(_MLP(d, cfg.ffn_mult)))
        self.rev_head = nn.Linear(d, d_rev)
        self.rev_act = BoundedActivation("flow")
        self.inv_head = StochLatent(d, d_inv, kind=cfg.z_inv_kind)

    def forward(self, feats):
        """feats: [.., M, enc_dim] → (z [..,M,d], kl [..])。"""
        h = self.in_proj(feats)
        for blk in self.blocks:
            h = blk(h)
        z_rev = self.rev_act(self.rev_head(h))
        z_inv, kl = self.inv_head(h)
        z = torch.cat([z_rev, z_inv], dim=-1)
        return z, kl.mean(dim=-1)               # kl: [..,M] → [..]


class MinecraftWorldModel(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.config = config
        self.d = config.d
        self.d_rev = config.d_rev
        self.d_inv = config.d_inv
        self.J = config.J
        self.S = config.max_skip
        self.event_vocab_size = config.effect.event_vocab_size
        self.beta_kl = config.adapter.beta_kl
        self.ema_decay = config.adapter.ema_decay
        self.unfreeze_backbone_layers = config.unfreeze_backbone_layers

        # 视觉骨干(冻结)
        self.backbone, self._patch, enc_dim, self._n_reg, self.encoder_kind = \
            build_backbone(config.backbone, injected=backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self._maybe_unfreeze_backbone()

        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 编码 adapter(online)+ EMA 教师(目标编码,stop-grad)
        self.adapter = _Adapter(enc_dim, self.d, self.d_rev, self.d_inv, config.adapter)
        self.target_adapter = copy.deepcopy(self.adapter)
        for p in self.target_adapter.parameters():
            p.requires_grad_(False)

        # 时空 token 编码(数学 (1))
        self.pos2d = PositionalEmbed(self.d, kind="sine2d")
        self.dt_enc = ContinuousTimeEncoding(self.d)
        self.tok_in = nn.Linear(self.d, self.d)                 # W_in:patch 内容投影
        self.action_enc = nn.Linear(config.act_dim, self.d)     # W_a:动作条件 token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)

        # 序列↔序列预测器主干(复用 Transformer 动力学核)
        self.blocks = build_dynamics(config.dynamics, self.d)

        # 因子化预测头:𝔤 系数 / 𝒟 事件 logits + 事件解码 / no-op / surprise
        self.coef_head = nn.Linear(self.d, config.effect.n_generators)
        self.generators = GeneratorBank(self.d_rev, config.effect.n_generators)
        self.rev_act = BoundedActivation("flow")                # 预测 z_rev 有界(I3)
        self.event_head = EventVocabHead(self.d, self.event_vocab_size)
        self.event_decode = nn.Embedding(self.event_vocab_size, self.d_inv)
        self.affordance = AffordanceHead(self.d)
        self.surprise = SurpriseHead(self.d, self.d_inv)

        # Placeholders(任务文本条件,沿用旧接口)
        self.task_proj = nn.Linear(384, self.d)

    # ---- 骨干 ----
    def _maybe_unfreeze_backbone(self):
        """探针失败时的逃生口:解冻 backbone 顶部 N 层(默认 0=全冻)。"""
        n = self.unfreeze_backbone_layers
        if n <= 0:
            return
        # 解冻名字里层号落在顶部 n 个的 transformer block 参数(近似,对各 HF ViT 命名鲁棒)。
        names = list(self.backbone.named_parameters())
        idxs = sorted({int(t) for name, _ in names
                       for t in name.replace(".", " ").split() if t.isdigit()})
        top = set(idxs[-n:]) if idxs else set()
        for name, p in names:
            toks = [int(t) for t in name.replace(".", " ").split() if t.isdigit()]
            if any(t in top for t in toks):
                p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_feats(self, img):
        ctx = torch.enable_grad() if self.unfreeze_backbone_layers > 0 else torch.no_grad()
        with ctx:
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

    # ---- 编码 ----
    def encode(self, feats):
        """online 编码:feats [..,M,enc] → (z [..,M,d], kl [..])。"""
        return self.adapter(feats)

    def encode_obs(self, img=None, feats=None):
        """兼容接口:返回 online 潜 z [B,M,d](SIGReg/可视化用)。"""
        if feats is None:
            feats = self.extract_feats(img)
        z, _ = self.adapter(feats)
        return z

    @torch.no_grad()
    def encode_target(self, feats):
        """EMA 教师编码(目标,stop-grad)。采样的随机性作轻量目标增广。"""
        z, _ = self.target_adapter(feats)
        return z.detach()

    @torch.no_grad()
    def update_ema(self):
        """EMA 教师跟随 online adapter(每个优化步后调用)。"""
        d = self.ema_decay
        for tp, op in zip(self.target_adapter.parameters(), self.adapter.parameters()):
            tp.mul_(d).add_(op.detach(), alpha=1.0 - d)
        for tb, ob in zip(self.target_adapter.buffers(), self.adapter.buffers()):
            tb.copy_(ob)

    # ---- 时空 token 化(数学 (1)) ----
    def _grid_pos(self, M, device):
        """patch 网格屏幕坐标 sine2d → [M, d]。方形网格按 √M×√M,否则退化 1×M。"""
        hw = int(round(math.sqrt(M)))
        H, W = (hw, hw) if hw * hw == M else (1, M)
        emb = self.pos2d(H, W, device=device)          # [1,d,H,W]
        return emb.reshape(self.d, M).t()               # [M,d]

    def _frame_tokens(self, z_frames, t_frames):
        """z_frames [B,Nf,M,d], t_frames [B,Nf] → patch token 集合 [B,Nf*M,d]。"""
        B, Nf, M, d = z_frames.shape
        pos = self._grid_pos(M, z_frames.device).view(1, 1, M, d)
        u = self.tok_in(z_frames) + pos
        tt = self.dt_enc(t_frames.reshape(-1)).to(u.dtype).view(B, Nf, 1, d)
        return (u + tt).reshape(B, Nf * M, d)

    def _action_tokens(self, act, t_act, null=False):
        """act [B,Na,act_dim], t_act [B,Na] → 动作条件 token [B,Na,d];null=True 置零动作。"""
        if null:
            act = torch.zeros_like(act)
        g = self.action_enc(act)
        tt = self.dt_enc(t_act.reshape(-1)).to(g.dtype).view(act.shape[0], act.shape[1], self.d)
        return g + tt

    def _query_tokens(self, query_t, M, B, device, dtype):
        """未来 query token q=m+ρ(屏幕坐标)+τ(t*) → [B,M,d]。"""
        pos = self._grid_pos(M, device).view(1, M, self.d)
        tt = self.dt_enc(query_t).to(dtype).view(B, 1, self.d)
        return self.mask_token.to(dtype) + pos + tt

    # ---- 序列↔序列前向(数学 (2)/(7)) ----
    def forward(self, z_frames, t_frames, act, t_act, query_t, null=False):
        """对上下文 token 集合 + 未来 query 做一次掩码预测。

        z_frames : [B, Nf, M, d]  上下文帧的 online 潜(锚点 = z_frames[:,0])
        t_frames : [B, Nf]        各上下文帧的帧时刻
        act      : [B, Na, act_dim] 区间动作(条件输入;null=True 置零)
        t_act    : [B, Na]        各动作的帧时刻
        query_t  : [B]            目标未来帧时刻 t*

        返回 dict:z_hat/[B,M,d]、z_hat_rev、z_hat_inv、event_logits/[B,M,V]、
                   c/[B,M,n_gen]、e_norm_hat/[B,M]、surprise/[B,M]、hq/[B,M,d]。
        """
        B, Nf, M, d = z_frames.shape
        ctx = self._frame_tokens(z_frames, t_frames)
        a_tok = self._action_tokens(act, t_act, null=null)
        q = self._query_tokens(query_t, M, B, z_frames.device, z_frames.dtype)
        X = torch.cat([ctx, a_tok, q], dim=1)
        X = self.blocks(X)
        hq = X[:, -M:, :]                                       # query 槽位输出 [B,M,d]

        anchor = z_frames[:, 0]                                 # [B,M,d]
        a_rev, a_inv = anchor[..., :self.d_rev], anchor[..., self.d_rev:]

        c = torch.tanh(self.coef_head(hq))                      # 有界系数(可逆增量小)
        z_rev_hat = self.rev_act(a_rev + self.generators(a_rev, c))

        event_logits = self.event_head(hq)                      # [B,M,V]
        probs = event_logits.float().softmax(dim=-1)
        z_inv_hat = a_inv + (probs @ self.event_decode.weight.float()).to(a_inv.dtype)

        z_hat = torch.cat([z_rev_hat, z_inv_hat], dim=-1)
        e_norm_hat = self.affordance(hq)
        _, surprise = self.surprise(hq)
        return {
            "z_hat": z_hat, "z_hat_rev": z_rev_hat, "z_hat_inv": z_inv_hat,
            "event_logits": event_logits, "c": c,
            "e_norm_hat": e_norm_hat, "surprise": surprise, "hq": hq,
        }
