"""MinecraftWorldModel 自监督训练的损失函数 (两步式离散词表与重构版)。

提供：
  vocab_pred_loss       — 第一步离散词表自回归预测的交叉熵分类损失。
  z_recon_loss          — 第二步 Cross-Attention 特征重建的 MSE 均方误差损失。
"""
import torch
import torch.nn.functional as F

EPS = 1e-4


def vocab_pred_loss(logits, target_token_id):
    """离散动作词表预测分类交叉熵损失。

    logits: [B, vocab_size]
    target_token_id: [B]
    """
    return F.cross_entropy(logits, target_token_id)


def z_recon_loss(z_recon, z_tg):
    """特征重建 MSE 损失。

    z_recon: [B, M, d]
    z_tg: [B, M, d]
    """
    # 基础 MSE 损失
    mse = F.mse_loss(z_recon.float(), z_tg.float())

    # 归一化残差比（Persistence Ratio，1.0 代表不发生转移的 baseline）
    with torch.no_grad():
        denom = z_tg.square().mean()
        ratio = mse / denom.clamp(min=1e-4)

    return mse, ratio.item()
