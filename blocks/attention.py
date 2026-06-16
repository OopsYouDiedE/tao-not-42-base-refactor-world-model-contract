"""L1 primitive 积木库 - 注意力与解码机制 (blocks/attention.py)"""
import torch
import torch.nn as nn


class PreLNAttn(nn.Module):
    """Pre-LN 多头注意 + 残差。mode∈{self,cross}。

    默认 need_weights=False ⇒ 走 PyTorch 融合 SDPA 快路径(need_weights=True 会强制
    物化注意力矩阵、禁用 flash/mem-efficient kernel,全模型显著拖慢)。
    可视化需要注意力图时把 store_attn 置 True:该次前向走慢路径,
    头平均注意力权重存入 last_attn(detach,[B, L_q, L_kv])。
    """

    def __init__(self, d, heads=4, mode="self"):
        super().__init__()
        assert mode in ("self", "cross")
        self.mode = mode
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.store_attn = False
        self.last_attn = None

    def forward(self, q, kv=None):
        if self.mode == "self" or kv is None:
            kv = q
        out, w = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
                           need_weights=self.store_attn)
        if self.store_attn and w is not None:
            self.last_attn = w.detach()
        return q + out


class ProtoDecode(nn.Module):
    """σ(clamp(einsum(coeff,proto),±15))。无参 (I3)。"""

    def forward(self, coeff, proto):
        logit = torch.einsum("bnk,bkhw->bnhw", coeff, proto)
        return torch.sigmoid(logit.clamp(-15.0, 15.0))
