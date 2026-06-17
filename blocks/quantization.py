"""Vector Quantizer 积木模块 (blocks/quantization.py)。

对外接口:
    VectorQuantizer — 带有 EMA 更新与随机重启机制的向量量化器。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """自适应向量量化器 (Vector Quantizer)。

    采用在线 EMA 质心更新与随机重启 (Random Restart) 机制，
    对输入特征进行离散量化并捕捉有限的物理聚类，支持长尾分布。
    """
    def __init__(self, dim: int, n_embed: int, decay: float = 0.99, eps: float = 1e-4, commitment_cost: float = 0.25):
        """
        Parameters
        ----------
        dim : int
            输入特征维度。
        n_embed : int
            词表大小 (如 512 或 1024)。
        decay : float, optional
            EMA 衰减系数，默认 0.99。
        eps : float, optional
            防除零不变量 epsilon，默认 1e-4 (I1)。
        commitment_cost : float, optional
            承诺损失 (Commitment Loss) 的权重系数，默认 0.25。
        """
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost

        # 词表参数使用 buffer 存储，因为是用 EMA 更新，而不是通过 optimizer 梯度更新
        embed = torch.randn(n_embed, dim)
        # 简单初始化：标准正态分布，使其在球面上分布
        embed.normal_()
        self.register_buffer("embed", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(n_embed))
        self.register_buffer("ema_w", embed.clone())
        
        # 记录每个 token 的累积使用次数，用于随机重启和监控
        self.register_buffer("usage_count", torch.zeros(n_embed))

    def forward(self, z):
        """前向传播进行量化。

        Parameters
        ----------
        z : torch.Tensor
            输入连续特征，Shape: [B, ..., dim], Dtype: torch.float32。

        Returns
        -------
        z_q : torch.Tensor
            量化后的特征，Shape: [B, ..., dim], Dtype: torch.float32。
        encoding_indices : torch.Tensor
            对应的词表 Token 索引，Shape: [B, ...], Dtype: torch.int64。
        loss : torch.Tensor
            承诺损失，Shape: [], Dtype: torch.float32。
        """
        # 保存原始 Shape
        orig_shape = z.shape
        z_flattened = z.reshape(-1, self.dim)  # [N, dim]
        
        # 为了计算精度与数值稳定，使用 fp32 危险算子 (I4)
        z_flattened_f32 = z_flattened.float()
        embed_f32 = self.embed.float()
        
        # 计算 L2 距离的平方: |z - e|^2 = |z|^2 + |e|^2 - 2 * z^T e
        d = (
            torch.sum(z_flattened_f32 ** 2, dim=1, keepdim=True)
            + torch.sum(embed_f32 ** 2, dim=1)
            - 2 * torch.matmul(z_flattened_f32, embed_f32.t())
        )  # [N, n_embed]
        
        # 找到最近邻
        encoding_indices = torch.argmin(d, dim=1)  # [N]
        
        # 进行 One-hot 编码
        encodings = F.one_hot(encoding_indices, self.n_embed).float()  # [N, n_embed]
        
        # 量化特征
        z_q = torch.matmul(encodings, self.embed).view(orig_shape)
        
        # 计算 commitment loss
        loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        
        # 直连估计器 (Straight-Through Estimator) 将梯度回传
        z_q = z + (z_q - z).detach()
        
        # 如果在训练模式，进行 EMA 更新与随机重启
        if self.training:
            # 统计本次 batch 每个 token 的被选次数
            # 注意：不更新计算图，只累积统计量
            with torch.no_grad():
                # 计算这个 batch 每个 embedding 的使用次数
                n_encodings = encodings.sum(0)  # [n_embed]
                
                # 更新使用计数
                self.usage_count.add_(n_encodings)
                
                # 1. 更新 EMA 聚类大小
                self.ema_cluster_size.lerp_(n_encodings, 1.0 - self.decay)
                
                # 2. 更新 EMA 特征加权和
                z_flattened_sum = torch.matmul(encodings.t(), z_flattened)  # [n_embed, dim]
                self.ema_w.lerp_(z_flattened_sum, 1.0 - self.decay)
                
                # 3. 计算新的质心
                # 使用 I1: 分母 clamp 到 self.eps (1e-4) 以上
                n = self.ema_cluster_size.clamp(min=self.eps)
                self.embed.copy_(self.ema_w / n.unsqueeze(1))
                
                # 4. 随机重启 (Random Restart) 机制
                # 如果某个 embedding 向量在此 batch 内完全没有被使用 (n_encodings == 0)，
                # 并且其长期的 EMA 计数也极小，我们就将它重新初始化为 batch 中的某个随机 z。
                dead_indices = (n_encodings == 0) & (self.ema_cluster_size < 1.0)
                if dead_indices.any() and z_flattened.shape[0] > 0:
                    dead_indices_where = torch.where(dead_indices)[0]
                    random_idx = torch.randint(0, z_flattened.shape[0], (len(dead_indices_where),), device=z.device)
                    new_embeds = z_flattened[random_idx]
                    
                    self.embed[dead_indices_where] = new_embeds
                    self.ema_w[dead_indices_where] = new_embeds
                    # 重新将大小设为 1，防止下一轮再被直接重置
                    self.ema_cluster_size[dead_indices_where] = 1.0
                    
        return z_q, encoding_indices.view(orig_shape[:-1]), loss
