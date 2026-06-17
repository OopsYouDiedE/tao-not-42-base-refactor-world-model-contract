"""效应词表 𝔤 ⊕ 𝒟:全部从**潜空间变化**读出,不碰原始动作标签 (net/effect_tokenizer.py)。

替换旧的 `net/action_model.py`:旧 ActionTokenizer 对**原始动作**做 VQ(token=按了什么键);
本模块对**不可逆潜变化 Δz_inv** 做 VQ(token=世界发生了什么不可逆后果),对齐原则 1
(对齐发生在编码空间,动作只作条件输入)。

对外接口:
    EffectTokenizer — 𝒟:对 Δz_inv 量化为事件码 + commitment 损失,暴露码本供 soft-decode。
    GeneratorBank   — 𝔤:作用于 z_rev 的可逆连续生成元算子组,系数由预测器在线给出。
"""
import torch
import torch.nn as nn

from blocks.quantization import VectorQuantizer


class EffectTokenizer(nn.Module):
    """𝒟 不可逆事件词表:对 Δz_inv = z_inv_{t+1} − z_inv_t 量化(EMA+死码重启)。

    与世界模型分离、由训练侧联合优化(承袭旧 ActionTokenizer 的装配位)。`Δz_inv≈0`
    (相机/平移/no-op)自然落到模最小的一个码 → 约定为 null 事件码(`null_code` 属性)。
    """
    def __init__(self, d_inv: int, event_vocab_size: int, decay: float = 0.99):
        super().__init__()
        self.d_inv = d_inv
        self.quantizer = VectorQuantizer(dim=d_inv, n_embed=event_vocab_size, decay=decay)

    @property
    def codebook(self):
        """事件码本 embed: [event_vocab_size, d_inv](EMA buffer,对 autograd 近似常量)。"""
        return self.quantizer.embed

    @property
    def null_code(self):
        """模最小的码索引 = null 事件(Δz_inv≈0 路由到此)。"""
        return int(self.quantizer.embed.float().norm(dim=1).argmin().item())

    def forward(self, z_inv_t, z_inv_next):
        """
        Parameters
        ----------
        z_inv_t, z_inv_next : torch.Tensor
            前/后帧不可逆潜,Shape: [B, M, d_inv] 或 [B, d_inv],Dtype: float。

        Returns
        -------
        event_idx : torch.Tensor  事件码索引,Shape: [B],Dtype: int64。
        loss : torch.Tensor       commitment 损失,Shape: []。
        delta : torch.Tensor      量化前的 Δz_inv(池化后),Shape: [B, d_inv]。
        """
        d = z_inv_next - z_inv_t
        if d.dim() == 3:                       # [B,M,d_inv] → 帧级净效应(对 patch 取均值)
            d = d.mean(dim=1)
        _, event_idx, loss = self.quantizer(d)
        return event_idx, loss, d


class GeneratorBank(nn.Module):
    """𝔤 可逆连续生成元算子组:ẑ_rev = z_rev + Σ_j c_j · G_j(z_rev)。

    G_j 从锚点潜 z_rev 生成一组有界方向(tanh),与预测器给出的系数 c 线性组合成增量。
    可逆性靠有界增量近似(非严格 Lie 群),与 BoundedActivation('flow') 协同保证 z_rev 不发散(I3)。
    """
    def __init__(self, d_rev: int, n_generators: int):
        super().__init__()
        self.n_generators = n_generators
        self.d_rev = d_rev
        self.basis = nn.Linear(d_rev, n_generators * d_rev, bias=False)

    def forward(self, z_rev, c):
        """
        Parameters
        ----------
        z_rev : torch.Tensor  锚点可逆潜,Shape: [B, M, d_rev]。
        c : torch.Tensor      生成元系数,Shape: [B, M, n_generators]。

        Returns
        -------
        delta : torch.Tensor  可逆增量 Σ_j c_j·G_j(z_rev),Shape: [B, M, d_rev]。
        """
        B, M, _ = z_rev.shape
        g = torch.tanh(self.basis(z_rev)).view(B, M, self.n_generators, self.d_rev)
        return (c.unsqueeze(-1) * g).sum(dim=2)
