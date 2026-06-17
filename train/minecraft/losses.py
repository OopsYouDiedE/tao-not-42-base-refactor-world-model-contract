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
import math

import torch
import torch.nn.functional as F

EPS = 1e-4


def pearson_corr(a, b, eps=EPS):
    """展平后的全局 Pearson 相关(返回带梯度标量张量;b 通常已 detach)。fp32(I4),分母 clamp(I1)。"""
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b) / (a.norm() * b.norm()).clamp(min=eps)


def patch_pixel_diff(img, anchor, target, M):
    """anchor/target 两帧逐 patch 像素差幅度 → [B, M](方形网格 √M×√M,否则退化 1×M)。"""
    B = img.shape[0]
    hw = int(round(math.sqrt(M)))
    H, W = (hw, hw) if hw * hw == M else (1, M)
    diff = (img[:, target].float() - img[:, anchor].float()).abs().mean(dim=1, keepdim=True)  # [B,1,h,w]
    pooled = F.adaptive_avg_pool2d(diff, (H, W))                          # [B,1,H,W]
    return pooled.reshape(B, M)


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


def effect_guidance_loss(e_norm, fdiv, pdiff, lambda_pixel=1.0):
    """引导后果权重(数学 (8) 验收同口径)。

    把反事实效应 e_norm 的**全局**空间分布拉向「去掉像素可线性预测部分」的未来潜发散
    fdiv_resid,并显式惩罚 e_norm 与像素差 pdiff 的相关:
        loss = (1 − corr(e_norm, fdiv_resid)) + λ·corr(e_norm, pdiff)²
    训练目标 == eval 的 corr(w,future)↑ / corr(w,pixel)→0。e_norm 带梯度;fdiv/pdiff stop-grad。

    旧实现是**逐样本归一 MSE**(mean(dim=-1)),只约束样本内形状、丢掉跨样本幅度,且目标 fdiv
    被像素污染——降到 0 却让 eval 的全局 corr_w_future 变负、corr_w_pixel 反升。此处改为
    与 eval 完全同口径的全局相关,并把像素分量从目标里残差化掉。
    """
    en = e_norm.float().reshape(-1)
    fd = fdiv.float().reshape(-1).detach()
    pd = pdiff.float().reshape(-1).detach()
    # 从 fdiv 回归掉 pdiff 可线性预测的部分,只留与像素无关的未来发散(stop-grad 目标)
    pd_c = pd - pd.mean()
    fd_c = fd - fd.mean()
    beta = (pd_c @ fd_c) / pd_c.pow(2).sum().clamp(min=EPS)
    fd_resid = fd_c - beta * pd_c
    corr_future = pearson_corr(en, fd_resid)
    corr_pixel = pearson_corr(en, pd)
    return (1.0 - corr_future) + lambda_pixel * corr_pixel.pow(2)


def null_consequence_loss(z_inv_hat_null, z_inv_anchor):
    """no-op 锚定(替代真 VPT 无标签、恒为 0 的 path_invariance):不动作 ⇒ 无不可逆后果。

    强制 do(null) 的预测 ẑ_inv 回到锚点 z_inv₀,使 e_norm=‖ẑ_inv(a)−ẑ_inv(null)‖ 成为
    真正的反事实增量基线,并止住 z_inv/闭环漂移的无约束膨胀。锚点 stop-grad(只约束预测器)。
    """
    return F.mse_loss(z_inv_hat_null.float(), z_inv_anchor.float().detach())


def decorrelation_loss(z_rev, z_inv, eps=EPS):
    """z_rev 与 z_inv 跨特征去相关(Barlow 式 cross-cov 惩罚),从表征根上切断不可逆通道的外观泄漏。

    两通道各自按特征在 batch 维标准化(零均值单位方差)后求互相关矩阵 C[d_rev,d_inv],
    惩罚其平方均值——逼 z_inv 不再与承载可逆/外观的 z_rev 共享信息(corr_w_pixel 爬升的病根)。
    作用于 online z(带梯度,塑形编码器);fp32(I4),分母 clamp(I1);只进 loss、不进前向(I6)。
    """
    zr = z_rev.float().reshape(-1, z_rev.shape[-1])
    zi = z_inv.float().reshape(-1, z_inv.shape[-1])
    zr = (zr - zr.mean(0)) / zr.std(0).clamp(min=eps)
    zi = (zi - zi.mean(0)) / zi.std(0).clamp(min=eps)
    n = max(zr.shape[0], 1)
    c = (zr.transpose(0, 1) @ zi) / n                    # [d_rev, d_inv] 跨通道互相关
    return c.pow(2).mean()

