"""序列对齐世界模型的损失函数(数学推导见 knowledge/mental_world.md)。

提供:
  latent_align_loss   — (4) 第一项:后果加权的潜对齐 MSE(预测 ẑ_{t*} → stop-grad 教师 z̄_{t*})。
  agreement_loss      — (4) 第二项:不同上下文截止对同一未来帧预测的互相一致。
  importance_from_effect — (5):由反事实效应 ‖e‖ 得后果权重 w(均值归一、上界 clamp,detach)。
  event_ce            — 𝒟 辅助离散通道:事件码交叉熵(目标来自 Δz_inv 的 VQ 索引)。
  noop_loss           — affordance/no-op 头回归实测 ‖e‖。
  path_invariance_loss— (6):到达同一未来态的不同序列其 ẑ_inv 须重合。
  recon_split         — 分通道重建指标(rev/inv)+ persistence-ratio 探针。
"""
import torch
import torch.nn.functional as F

EPS = 1e-4


def importance_from_effect(e_norm, w_max=5.0):
    """由反事实效应幅度 ‖e‖ 得后果权重 w(数学 (5))。

    e_norm: [B, M] 实测 ‖ẑ_inv(do a) − ẑ_inv(do null)‖。归一到均值 1、上界 clamp(I1/I3),
    **detach**(只重加权、不建梯度通路)。权重随未来效应、不随像素差 ⇒ 小像素高后果不被淹没。
    """
    with torch.no_grad():
        m = e_norm.float().mean().clamp(min=EPS)
        w = (e_norm.float() / m).clamp(max=w_max)
    return w


def latent_align_loss(z_hat, z_tgt, w=None):
    """后果加权潜对齐 MSE(数学 (4) 第一项)。

    z_hat: [B,M,d] 预测;z_tgt: [B,M,d] stop-grad 教师目标;w: [B,M] 后果权重(None=1)。
    返回 (loss 标量, 逐样本均方误差 detached 供监控)。
    """
    se = (z_hat.float() - z_tgt.float().detach()).pow(2).mean(dim=-1)   # [B,M]
    if w is not None:
        loss = (w * se).mean()
    else:
        loss = se.mean()
    return loss, se.detach()


def agreement_loss(z_hats):
    """不同上下文截止对同一 t* 预测的互相一致(数学 (4) 第二项)。

    z_hats: list of [B,M,d]。两两向对方的 stop-grad 靠拢(对称、无 teacher)。
    """
    if len(z_hats) < 2:
        return torch.zeros((), device=z_hats[0].device)
    acc = torch.zeros((), device=z_hats[0].device)
    n = 0
    for i in range(len(z_hats)):
        for j in range(i + 1, len(z_hats)):
            acc = acc + F.mse_loss(z_hats[i].float(), z_hats[j].float().detach())
            n += 1
    return acc / max(n, 1)


def event_ce(event_logits_pooled, event_idx):
    """𝒟 事件码交叉熵(辅助离散通道)。logits: [B,V];event_idx: [B](来自 Δz_inv 的 VQ)。"""
    return F.cross_entropy(event_logits_pooled, event_idx)


def noop_loss(e_norm_hat, e_norm_measured):
    """affordance/no-op 头回归实测效应幅度。两者均 [B,M];目标 stop-grad。"""
    return F.mse_loss(e_norm_hat.float(), e_norm_measured.float().detach())


def path_invariance_loss(z_inv_hat, reach_id):
    """(6):batch 内到达同一未来态(reach_id 相同)的样本,其 ẑ_inv 须重合。

    z_inv_hat: [B,M,d_inv];reach_id: [B](int,<0 表示无标注)。同组两两 MSE 平均;无同组返回 0。
    """
    device = z_inv_hat.device
    zi = z_inv_hat.float().mean(dim=1)                  # [B,d_inv] 帧级
    acc = torch.zeros((), device=device)
    n = 0
    B = zi.shape[0]
    for i in range(B):
        if reach_id[i] < 0:
            continue
        for j in range(i + 1, B):
            if reach_id[j] == reach_id[i]:
                acc = acc + (zi[i] - zi[j]).pow(2).mean()
                n += 1
    return acc / n if n > 0 else acc


def recon_split(z_hat_rev, z_hat_inv, z_tgt, d_rev):
    """分通道重建指标 + persistence-ratio 探针(是否跑赢"不变"baseline)。

    返回 dict(recon_rev, recon_inv, persistence_ratio) 均为 python float(只作监控)。
    """
    with torch.no_grad():
        tr, ti = z_tgt[..., :d_rev].float(), z_tgt[..., d_rev:].float()
        recon_rev = F.mse_loss(z_hat_rev.float(), tr).item()
        recon_inv = F.mse_loss(z_hat_inv.float(), ti).item()
        denom = z_tgt.float().square().mean().clamp(min=EPS)
        ratio = (F.mse_loss(torch.cat([z_hat_rev, z_hat_inv], -1).float(),
                            z_tgt.float()) / denom).item()
    return {"recon_rev": recon_rev, "recon_inv": recon_inv, "persistence_ratio": ratio}
