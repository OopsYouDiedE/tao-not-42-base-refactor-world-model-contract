import torch

def compute_sinkhorn_matching(cost, epsilon=0.1, iters=10):
    """
    GPU 原生 Sinkhorn 算法求软分配概率矩阵 (Zero CPU-GPU Sync)
    
    Args:
        cost: 代价矩阵 [N, M]
        epsilon: 正则化参数
        iters: 迭代次数
        
    Returns:
        P_opt: 软分配概率矩阵 [N, M]
    """
    P = torch.exp(-cost / epsilon)
    u = torch.ones_like(P[:, 0])
    v = torch.ones_like(P[0, :])
    for _ in range(iters):
        u = 1.0 / (torch.matmul(P, v) + 1e-8)
        v = 1.0 / (torch.matmul(P.t(), u) + 1e-8)
    P_opt = u.unsqueeze(1) * P * v.unsqueeze(0)
    return P_opt

def sinkhorn_batched(cost, epsilon=0.1, iters=20):
    """批量 Sinkhorn。cost:[B,K,M] → 软分配 P:[B,K,M]。纯 tensor,GPU 零同步。"""
    P = torch.exp(-cost / epsilon)                       # [B,K,M]
    u = torch.ones_like(P[:, :, 0])                      # [B,K]
    v = torch.ones_like(P[:, 0, :])                      # [B,M]
    for _ in range(iters):
        u = 1.0 / ((P * v.unsqueeze(1)).sum(-1) + 1e-8)  # [B,K]
        v = 1.0 / ((P * u.unsqueeze(-1)).sum(1) + 1e-8)  # [B,M]
    return u.unsqueeze(-1) * P * v.unsqueeze(1)
