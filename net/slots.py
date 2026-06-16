"""实体槽绑定:把感知 token 绑定到一组持久潜向量(slot)上。

对外接口:
    SlotCompetitiveAttn — slot 维竞争交叉注意力(防 slot 冗余,聚合权重供多样性损失)。
    SlotBinder          — 贝叶斯滤波式门控绑定(逐 slot 增益 K·innovation)。
    build_binder(cfg, d) — 按 EncoderConfig 造 binder(competitive / preln)。

由 net.world_model 的在线/目标感知编码共用;可视化与 train 侧的 slot 多样性损失
读 SlotCompetitiveAttn.attn_map(竞争注意力图)。
"""
import torch
import torch.nn as nn

from blocks.attention import PreLNAttn
from net.config import EncoderConfig


class SlotCompetitiveAttn(nn.Module):
    """Slot-Attention 式竞争交叉注意力:softmax 沿 **slot 维**(而非 token 维)。

    PreLNAttn 的 softmax 沿 token 维,各 slot 的注意力互相独立 ⇒ 都涌向最显著
    区域(Minecraft 面板 E 实测:多个 slot 盯同一块画面,冗余)。竞争归一化把
    "哪个 slot 解释哪块输入"变成逐 token 的零和分配——一个 patch 的注意力质量
    被 slot 们瓜分,排他绑定从结构上成立。聚合时再沿 token 维归一做加权平均
    (Slot Attention 原版两步归一)。这是修 slot 冗余的对症位置:约束加在
    "谁看哪里"(注意力空间);对比表征空间的正交惩罚——治标,且与"实体间语义
    相关性应可由内积表达"冲突。

    接口与 PreLNAttn 对齐:残差输出 Z + proj(out),store_attn/last_attn
    ([B, N, M],聚合权重的头平均)——SlotBinder 与可视化无需区分两者。
    softmax/归一在 fp32(I4),分母 clamp(I1)。
    """

    def __init__(self, d, heads=4, eps=1e-4):
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

    def forward(self, Z, P):
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


class SlotBinder(nn.Module):
    """将全局与局部感知 Token 绑定到实体 Slot 上。

    贝叶斯滤波视角:Z' = Z + K·innovation。增益 K 逐 slot 由 (slot 状态, 感知修正量)
    条件化预测——被遮挡实体应 K→0 维持先验,新出现/观测清晰的实体应 K→1 接收观测。
    全模型共享单标量增益是该增益函数空间里最受限的一档,无法表达上述区分。
    gate 权重零初始化、bias=0.1 ⇒ 冷启动时逐 slot 增益恒等于旧版全局 sigmoid(0.1),
    已验证管线的初始行为不变。

    compete=True 用 SlotCompetitiveAttn(slot 维竞争,防 slot 冗余;Minecraft 管线用);
    默认 False 保持 PreLNAttn,既有 tao 管线行为不变。
    """

    def __init__(self, d, compete=False, heads=4):
        super().__init__()
        self.compete = compete
        self.attn = SlotCompetitiveAttn(d, heads=heads) if compete \
            else PreLNAttn(d, heads=heads, mode="cross")
        self.ln = nn.LayerNorm(d)
        # I5: 增益受限于 (0,1)。输入 [Z_i; δ_i] → 逐 slot 标量增益
        self.gate = nn.Linear(2 * d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.1)

    def forward(self, Z, P):
        # Z: [B, N, d] (Slots)
        # P: [B, M+1, d] (Perception tokens)

        # attn 内部有残差 Z + CrossAttn(Z, P)，但我们想用门控残差，
        # 为了复用 PreLNAttn 的层归一化和注意力，我们可以手动提取注意力输出
        # 因为 PreLNAttn 返回 q + attn_out，所以我们减去 q 得到 attn_out
        z_out = self.attn(Z, P)
        delta_Z = z_out - Z

        gate = torch.sigmoid(self.gate(torch.cat([Z, delta_Z], dim=-1)))  # [B, N, 1]
        Z_new = Z + gate * self.ln(delta_Z)  # I5

        return Z_new


def build_binder(cfg: EncoderConfig, d: int) -> SlotBinder:
    """按 EncoderConfig 造实体槽 binder。

    cfg.binder: 'competitive'→SlotCompetitiveAttn(slot 维竞争,防冗余,Minecraft 管线用);
                'preln'→PreLNAttn cross(既有 tao 管线行为)。heads 由 cfg.binder_heads 给。
    """
    if cfg.binder == "competitive":
        return SlotBinder(d, compete=True, heads=cfg.binder_heads)
    if cfg.binder == "preln":
        return SlotBinder(d, compete=False, heads=cfg.binder_heads)
    raise ValueError(f"未知 binder: {cfg.binder}")
