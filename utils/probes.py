"""世界模型"真学习"诊断探针。

低 loss ≠ 学到了——这些探针专门测出退化捷径。最关键:`action_shuffle_sensitivity`
(动作被忽略则世界模型对规划无用)。判据阶梯见 knowledge/refactor_plan.md(canary)。
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def action_shuffle_sensitivity(predict_fn, Z, a_lat, target):
    """打乱动作后预测 loss 应**变差**;若几乎不变 ⇒ 模型在忽略动作。

    Args:
        predict_fn: (Z, a_lat) -> 预测张量(与 target 同形)。
        Z, a_lat, target: 同 batch。
    Returns:
        (loss_real, loss_shuffled, ratio),ratio=(shuf-real)/real,**>0 才说明动作没被忽略**。
    """
    loss_real = F.mse_loss(predict_fn(Z, a_lat), target)
    perm = torch.randperm(a_lat.shape[0], device=a_lat.device)
    loss_shuf = F.mse_loss(predict_fn(Z, a_lat[perm]), target)
    ratio = (loss_shuf - loss_real) / (loss_real + 1e-8)
    return loss_real.item(), loss_shuf.item(), ratio.item()


@torch.no_grad()
def latent_effective_rank(Z, eps=1e-9):
    """潜在有效秩(谱熵 exp)。坍缩(JEPA 退化)时趋于极低。Z:[B,...] 任意,展平到 [B,D]。"""
    x = Z.flatten(1).float()
    x = x - x.mean(0, keepdim=True)
    cov = (x.t() @ x) / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = ev / (ev.sum() + eps)
    p = p[p > eps]
    return torch.exp(-(p * p.log()).sum()).item()
