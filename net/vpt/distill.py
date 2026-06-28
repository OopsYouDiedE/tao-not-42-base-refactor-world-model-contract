"""VPT → CraftGround 蒸馏损失 (net/vpt/distill.py)。

对外接口:
    vpt_distill_loss — KL 散度 + 特征对齐 + 可选硬标签损失。
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def vpt_distill_loss(
    student_logits: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    temp: float = 2.0,
    alpha: float = 0.7,
    hard_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VPT teacher → student 知识蒸馏损失。

    Args:
        student_logits: (B, 27) 学生动作 logits
        student_hidden: (B, hidsize) 学生隐藏状态
        teacher_logits: (B, 27) teacher 投影后 logits
        teacher_hidden: (B, hidsize) teacher 投影后 hidden
        temp:  软化温度(默认2.0)
        alpha: 软标签权重 ∈ [0,1](默认0.7)
        hard_labels: (B,) 真实动作 id,可选

    Returns:
        total_loss: 标量张量
        metrics:    {'kl': float, 'feat': float, 'hard': float}

    损失组合: L = α(KL + MSE_feat) + (1-α)CE_hard
    """
    # KL 散度(软标签蒸馏)
    kl = F.kl_div(
        F.log_softmax(student_logits / temp, dim=-1),
        F.softmax(teacher_logits / temp, dim=-1),
        reduction='batchmean'
    ) * (temp ** 2)

    # 特征对齐
    feat = F.mse_loss(student_hidden, teacher_hidden.detach())

    # 硬标签(可选)
    hard = torch.tensor(0.0, device=student_logits.device)
    if hard_labels is not None:
        hard = F.cross_entropy(student_logits, hard_labels)

    total = alpha * (kl + feat) + (1 - alpha) * hard
    return total, {'kl': kl.item(), 'feat': feat.item(), 'hard': hard.item()}
