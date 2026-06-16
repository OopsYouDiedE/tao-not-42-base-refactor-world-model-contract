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


class SlotCompetitiveAttn(nn.Module):
    """Slot-Attention 式竞争交叉注意力: softmax 沿 slot 维(而非 token 维)。

    PreLNAttn 的 softmax 沿 token 维，各 slot 的注意力互相独立，导致都涌向最显著
    区域。竞争归一化把 "哪个 slot 解释哪块输入" 变成逐 token 的零和分配
    ——一个 patch 的注意力质量被 slots 瓜分，排他绑定从结构上成立。
    聚合时再沿 token 维归一做加权平均。

    接口与 PreLNAttn 对齐: 残差输出 Z + proj(out)，store_attn/last_attn
    ([B, N, M]，聚合权重的头平均)——SlotBinder 与可视化无需区分两者。
    softmax/归一在 fp32(I4)，分母 clamp(I1)。
    """

    def __init__(self, d: int, heads: int = 4, eps: float = 1e-4):
        """
        Parameters
        ----------
        d : int
            通道维度，必须被 heads 整除。
        heads : int, optional
            注意力头数，默认为 4。
        eps : float, optional
            分母 clamp 的 epsilon 阈值，默认为 1e-4。
        """
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.eps = heads, d // heads, eps
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.store_attn = False
        self.last_attn = None

    def forward(self, Z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """执行竞争交叉注意力计算。

        Parameters
        ----------
        Z : torch.Tensor
            Slot 状态，Shape: [B, N, d]，Dtype: float32 或 float16
        P : torch.Tensor
            感知 token，Shape: [B, M, d]，Dtype: float32 或 float16

        Returns
        -------
        torch.Tensor
            注意力更新后的 Slot 状态，Shape: [B, N, d]，Dtype: 与 Z 相同
        """
        B, N, d = Z.shape
        M = P.shape[1]
        q = self.q(self.ln_q(Z)).view(B, N, self.h, self.dh).transpose(1, 2)   # [B,h,N,dh]
        kv = self.ln_kv(P)
        k = self.k(kv).view(B, M, self.h, self.dh).transpose(1, 2)             # [B,h,M,dh]
        v = self.v(kv).view(B, M, self.h, self.dh).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)).float() / (self.dh ** 0.5)          # [B,h,N,M] fp32
        attn = logits.softmax(dim=2)                       # 竞争:沿 slot 维归一
        w = attn / attn.sum(dim=-1, keepdim=True).clamp(min=self.eps)  # 聚合:沿 token 维归一
        # 槽间多样性用的注意力图(谁看哪块 patch);头平均、带梯度,供训练侧
        # slot_diversity_loss 软惩罚成对重叠。前向 w 仍是合法分布(非负、沿 token 和 1)
        # ——把"不同 slot 别盯同一块"写进损失,不在前向里硬改聚合权重(硬正交化会
        # 打破 out=w·v 的加权平均语义、并按 slot 序饿死后排槽)。
        self.attn_map = w.mean(dim=1)                                          # [B,N,M] fp32
        out = (w.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, d)
        if self.store_attn:
            self.last_attn = self.attn_map.detach()                            # [B,N,M] 可视化
        return Z + self.out(out)

