"""离散动作词表编码与执行模块 (net/action_model.py)。

对外接口:
    ActionTokenizer — 连续动作序列到离散 Token 的聚类编码器。
    ActionExecutor  — 基于时间编码和环境上下文的离散 Token 执行器。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import VectorQuantizer, ContinuousTimeEncoding


class ActionTokenizer(nn.Module):
    """连续动作序列到离散 Token 的自适应聚类编码器。"""
    
    def __init__(self, act_dim: int, hidden_dim: int, latent_dim: int, n_embed: int, decay: float = 0.99):
        """
        Parameters
        ----------
        act_dim : int
            原始动作维度。
        hidden_dim : int
            隐层维度。
        latent_dim : int
            潜空间特征维度。
        n_embed : int
            离散词表大小 (如 512 或 1024)。
        decay : float, optional
            Codebook EMA 衰减系数，默认 0.99。
        """
        super().__init__()
        # 使用 MLP 提取动作序列的连续特征，再做时间池化
        self.encoder = nn.Sequential(
            nn.Linear(act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.quantizer = VectorQuantizer(dim=latent_dim, n_embed=n_embed, decay=decay)

    def forward(self, a_seq, valid_mask=None):
        """对动作序列进行离散 Token 提取。

        Parameters
        ----------
        a_seq : torch.Tensor
            输入时序动作序列，Shape: [B, S, act_dim], Dtype: torch.float32。
        valid_mask : torch.Tensor, optional
            序列有效掩码，Shape: [B, S], Dtype: torch.float32 (1.0 = 有效, 0.0 = 无效)。

        Returns
        -------
        z_q : torch.Tensor
            量化后的潜动作表征向量，Shape: [B, latent_dim], Dtype: torch.float32。
        indices : torch.Tensor
            量化后的离散动作 Token 索引，Shape: [B], Dtype: torch.int64。
        loss : torch.Tensor
            自监督承诺损失，Shape: [], Dtype: torch.float32。
        """
        # 提取每一帧的连续表示
        h = self.encoder(a_seq)  # [B, S, latent_dim]
        
        # 结合 valid_mask 进行池化，排除无效填充对动作表示的污染
        if valid_mask is not None:
            # valid_mask: [B, S] -> [B, S, 1]
            mask = valid_mask.unsqueeze(-1)
            h = h * mask
            h_pooled = h.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-4)  # I1: clamp >= 1e-4
        else:
            h_pooled = h.mean(dim=1)  # [B, latent_dim]
            
        z_q, indices, loss = self.quantizer(h_pooled)
        return z_q, indices, loss


class ActionExecutor(nn.Module):
    """基于时间编码和环境上下文的离散 Token 执行器。"""
    
    def __init__(self, act_dim: int, latent_dim: int, state_dim: int, hidden_dim: int, max_skip: int):
        """
        Parameters
        ----------
        act_dim : int
            原始动作维度。
        latent_dim : int
            动作 Token 空间维度。
        state_dim : int
            环境状态表征维度。
        hidden_dim : int
            隐层维度。
        max_skip : int
            最大动作序列输出步长。
        """
        super().__init__()
        self.max_skip = max_skip
        
        # 连续时间编码器，将步长索引编码为正弦特征
        self.time_enc = ContinuousTimeEncoding(latent_dim)
        
        # 上下文投影：结合动作 Token 特征和环境上下文状态
        self.ctx_proj = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 解码层：输入 ctx 特征和当前时间步特征，还原出当前的真实动作
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, z_q, z_ref, dt):
        """执行动作 Token，还原为真实动作序列。

        Parameters
        ----------
        z_q : torch.Tensor
            选中的动作 Token embedding，Shape: [B, latent_dim], Dtype: torch.float32。
        z_ref : torch.Tensor
            当前环境上下文潜状态，Shape: [B, N, state_dim], Dtype: torch.float32。
        dt : torch.Tensor
            各样本的实际执行时间 (最大为 max_skip)，Shape: [B], Dtype: torch.float32/torch.int64。

        Returns
        -------
        a_recon : torch.Tensor
            重构后的动作时序序列，Shape: [B, max_skip, act_dim], Dtype: torch.float32。
        """
        B = z_q.shape[0]
        
        # 1. 对环境上下文状态进行平均池化，结合成一个统一的上下文环境向量
        # z_ref: [B, N, state_dim] -> [B, state_dim]
        z_env = z_ref.mean(dim=1)
        
        # 2. 将动作特征与环境特征拼接并投影，形成执行上下文特征
        # ctx_feat: [B, hidden_dim]
        ctx_feat = self.ctx_proj(torch.cat([z_q, z_env], dim=-1))
        
        # 3. 构造相对时间轴。对于每一帧偏移 j，使用相对时间编码
        # t_seq: [B, max_skip]
        device = z_q.device
        t_seq = torch.arange(self.max_skip, device=device).unsqueeze(0).expand(B, -1).float()
        
        # 将相对时间编码成特征
        # t_enc: [B, max_skip, latent_dim]
        t_enc = self.time_enc(t_seq).view(B, self.max_skip, -1)
        
        # 4. 拼合执行上下文与时间特征
        # ctx_feat_expanded: [B, max_skip, hidden_dim]
        ctx_feat_expanded = ctx_feat.unsqueeze(1).expand(-1, self.max_skip, -1)
        # combined: [B, max_skip, hidden_dim + latent_dim]
        combined = torch.cat([ctx_feat_expanded, t_enc], dim=-1)
        
        # 5. 解码还原出动作序列
        a_recon = self.decoder(combined)  # [B, max_skip, act_dim]
        
        # 6. 结合执行持续时间 dt 进行有效位处理
        # 超过 dt 帧的部分在物理上是无效的（填充动作，通常为 0）
        # valid_mask: [B, max_skip, 1]
        valid_mask = (torch.arange(self.max_skip, device=device).unsqueeze(0)
                      < dt.unsqueeze(1)).to(a_recon.dtype).unsqueeze(-1)
        a_recon = a_recon * valid_mask
        
        return a_recon
