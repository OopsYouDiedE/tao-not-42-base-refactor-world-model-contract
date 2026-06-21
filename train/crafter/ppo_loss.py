"""PPO clip 损失计算 (train/crafter/ppo_loss.py)。

对外接口:
    ppo_loss — 返回总损失及各分项(供日志记录)。
"""
import torch
import torch.nn.functional as F


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_values: torch.Tensor,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    entropy: torch.Tensor | None = None,
):
    """PPO clip policy loss + clipped value loss + entropy bonus。

    Args:
        new_log_probs: (B,) float32 — 当前策略下动作的 log 概率。
        old_log_probs: (B,) float32 — 收集轨迹时的 log 概率(旧策略)。
        advantages:    (B,) float32 — 已标准化的 GAE 优势。
        returns:       (B,) float32 — λ-return。
        new_values:    (B,) float32 — 当前 critic 输出。
        clip_coef:     PPO clip ε。
        vf_coef:       价值损失权重。
        ent_coef:      熵奖励权重(负号内置,loss 减熵)。
        entropy:       (B,) float32,可选;None 时熵项为 0。

    Returns:
        total:   scalar — 总损失(供 backward)。
        pg_loss: scalar — 策略梯度 clip 损失。
        v_loss:  scalar — 价值 MSE 损失。
        ent:     scalar — 平均熵(正值)。
    """
    ratio = (new_log_probs - old_log_probs).exp()
    pg1 = -advantages * ratio
    pg2 = -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = torch.max(pg1, pg2).mean()

    v_loss = F.mse_loss(new_values, returns)

    ent = entropy.mean() if entropy is not None else torch.zeros(1, device=new_log_probs.device).squeeze()

    total = pg_loss + vf_coef * v_loss - ent_coef * ent
    return total, pg_loss, v_loss, ent
