import torch
import torch.nn.functional as F

from utils.matching import sinkhorn_batched

@torch.no_grad()
def update_ema_teacher(student_model, teacher_model, momentum=0.996):
    """更新 EMA 教师网络 (I8: 停止梯度)。"""
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data = param_t.data * momentum + param_s.data * (1.0 - momentum)


def info_nce_loss(z_pred, z_target, temperature=0.07):
    """控制变量反事实对比损失。fp32(I4), 分母安全(I1)。
    z_pred: [B, N, d] 预测特征
    z_target: [B, N, d] 教师网络输出目标特征
    """
    B, N, d = z_pred.shape
    # Flatten to [B*N, d]
    z_pred = z_pred.reshape(B * N, d)
    z_target = z_target.reshape(B * N, d)
    
    # normalize (I1: eps=1e-4)
    z_pred = F.normalize(z_pred, dim=-1, eps=1e-4)
    z_target = F.normalize(z_target, dim=-1, eps=1e-4)
    
    # logits (fp32 I4)
    logits = torch.matmul(z_pred.float(), z_target.float().T) / temperature
    labels = torch.arange(B * N, device=z_pred.device)
    
    return F.cross_entropy(logits, labels)


def gaussian_nll_loss(mu, sigma, target, active_mask):
    """高斯负对数似然损失 (JEPA 概率云对齐)。
    包含了两项的博弈:
    1. ||mu - target||^2 / sigma^2 : 迫使 mu 逼近 target。如果不准，模型会放大 sigma 逃避惩罚。
    2. log(sigma^2) : 惩罚过大的方差，迫使模型在可能的情况下尽可能确信 (反偷懒)。
    """
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=mu.device)
    
    # I1: 限制 sigma 的下界，防止除 0 和 log(0)
    sigma2 = (sigma ** 2).clamp(min=1e-4) 
    
    # 均方误差项
    mse = F.mse_loss(mu, target, reduction='none')
    
    # NLL = (mse / sigma^2) + log(sigma^2)
    nll = (mse / sigma2) + torch.log(sigma2)
    
    # 仅对存在概率 > 0.5 的活跃 Slot 计算
    valid_nll = (nll * active_mask.unsqueeze(-1)).sum(-1)
    
    return valid_nll.sum() / active_mask.sum()


def action_plan_loss(pred, gt, w_onset=2.0, w_dur=1.0, w_key=0.5, w_exist=1.0):
    """DETR 式集合匹配:K 个预测动作 ↔ M 个 GT 待击打动作。全批量,GPU 零同步。

    pred: {key_logits[B,K,n_keys], onset[B,K], duration[B,K], exist[B,K]}
    gt  : {onset[B,M], duration[B,M], track[B,M], valid[B,M]}  (来自 env.get_upcoming_actions)

    匹配代价 = w_onset·|Δonset| + w_dur·|Δdur| + w_key·(1-p_key)。匹配用 Sinkhorn(detach),
    再按 GT onset 升序做**单射贪心指派**(逐列 argmax 允许多个 GT 撞同一槽,会给该槽
    注入互相冲突的回归目标;贪心去重保证一一对应,需 M ≤ K——本仓库调用处 M == K)。
    匹配上的:回归 onset/时长 + 键分类 + exist→1;未匹配的:exist→0。
    指标返回 0 维 tensor(调用方按需 .item(),热路径不同步)。
    """
    on_p, dur_p, key_p, ex_p = pred["onset"], pred["duration"], pred["key_logits"], pred["exist"]
    gon, gdur, gtrk, gval = gt["onset"], gt["duration"], gt["track"], gt["valid"]
    B, K = on_p.shape
    M = gon.shape[1]
    n_keys = key_p.shape[-1]
    eps = 1e-6

    with torch.no_grad():
        cp = key_p.softmax(-1)                                       # [B,K,n_keys]
        key_cost = 1.0 - cp.gather(2, gtrk.clamp(0, n_keys - 1)      # [B,K,M]
                                   .unsqueeze(1).expand(B, K, M))
        cost = (w_onset * (on_p.unsqueeze(2) - gon.unsqueeze(1)).abs()
                + w_dur * (dur_p.unsqueeze(2) - gdur.unsqueeze(1)).abs()
                + w_key * key_cost)
        cost = cost + (1.0 - gval).unsqueeze(1) * 1e4               # 屏蔽无效 GT 列
        P = sinkhorn_batched(cost)                                  # [B,K,M]
        # 单射贪心指派:有效 GT 按 onset 升序优先挑槽,已占用槽被屏蔽 ⇒ sel 每行无重复。
        # (Sinkhorn 输出是双随机阵内点,逐列 argmax 不是到置换阵的投影,可能撞槽。)
        b_idx = torch.arange(B, device=P.device)
        order = torch.argsort(
            torch.where(gval > 0.5, gon, torch.full_like(gon, 1e9)), dim=1)  # 无效列垫后
        sel = torch.zeros(B, M, dtype=torch.long, device=P.device)
        taken = torch.zeros(B, K, dtype=torch.bool, device=P.device)
        for j in range(M):                                          # M 小(=K),纯 GPU 零同步
            col = order[:, j]                                       # [B] 当前指派的 GT 列
            score = P[b_idx, :, col].masked_fill(taken, -1.0)       # P≥0,-1 必不被选
            pick = score.argmax(dim=1)                              # [B]
            sel[b_idx, col] = pick
            taken[b_idx, pick] = True

    on_sel = on_p.gather(1, sel)                                    # [B,M]
    dur_sel = dur_p.gather(1, sel)
    key_sel = key_p.gather(1, sel.unsqueeze(-1).expand(B, M, n_keys))  # [B,M,n_keys]

    gsum = gval.sum().clamp(min=1.0)
    L_on = (F.smooth_l1_loss(on_sel, gon, reduction="none", beta=0.2) * gval).sum() / gsum
    L_dur = (F.smooth_l1_loss(dur_sel, gdur, reduction="none", beta=0.2) * gval).sum() / gsum
    ce = F.cross_entropy(key_sel.reshape(B * M, n_keys),
                         gtrk.reshape(B * M).clamp(0, n_keys - 1), reduction="none").reshape(B, M)
    L_key = (ce * gval).sum() / gsum

    # exist 目标:匹配到的槽→1(OR 语义,防同槽冲突),其余→0
    ex_t = torch.zeros_like(ex_p).scatter_reduce_(1, sel, gval, reduce="amax", include_self=True)
    L_ex = F.binary_cross_entropy(ex_p, ex_t)

    total = w_onset * L_on + w_dur * L_dur + w_key * L_key + w_exist * L_ex

    with torch.no_grad():
        off = 1.0 - ex_t
        metrics = {
            "OnsetMAEms": 1000.0 * (on_sel - gon).abs().mul(gval).sum() / gsum,
            "DurMAEms": 1000.0 * (dur_sel - gdur).abs().mul(gval).sum() / gsum,
            "KeyAcc": ((key_sel.argmax(-1) == gtrk).float() * gval).sum() / gsum,
            "ExistOn": (ex_p * ex_t).sum() / ex_t.sum().clamp(min=1.0),
            "ExistOff": (ex_p * off).sum() / off.sum().clamp(min=1.0),
        }
    return total, metrics
