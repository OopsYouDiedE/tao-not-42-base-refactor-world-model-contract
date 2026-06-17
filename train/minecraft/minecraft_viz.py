"""train_minecraft 训练效果可视化:一张 PNG 面板,直观回答"世界模型学到了没"。

对外接口:
    visualize_minecraft(model, effect_tok, batch, cfg, device, amp_dev, use_amp,
                        out_path, epoch=0, n_rollout=5) — 在固定 holdout batch 上渲染诊断面板。

证伪原则(与 eval 的 persistence/反捷径标尺同源):每个子图都带一个不可作弊的对照,
低 loss 本身不构成证据。面板内容(固定一条 holdout 序列,跨 epoch 可前后对比):
  A. Δz 预测误差 vs persistence 基线(预测"什么都不变"=锚点)。模型条低于灰色 persistence
     条才说明在"预测变化"而非沉默;按上下文截止 k 分组。
  B. 后果权重 w(patch 网格):w=‖ẑ_inv(do a)−ẑ_inv(do null)‖ 归一,模型把"哪些 patch
     的变化由动作引起"压进 w。
  C. 像素差(patch 网格)+ 标题显示 corr(w,future)/corr(w,pixel):w 应与未来潜发散正相关、
     与像素差去相关 ⇒ B 与 C 看着**不像**才说明没在像素面积上偷懒(反捷径,数学 (8))。
  D. generator 系数 c(控制闸,gen×patch 热图):𝔤 可逆增量的有界系数,应随训练出现极化。
  E. 事件词表使用分布(𝒟 离散通道):argmax(event_logits) 在词表上的直方图,坍成单柱=词表塌缩。
  F. 开环 rollout 漂移:用预测作锚点再推 N 步,看 ẑ_inv 增量幅度是否有界(不发散)。

图内文字用英文:matplotlib 默认字体无 CJK,避免 Colab 上满图 tofu 方块。
"""
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.minecraft._seq import _to_float_img
from train.minecraft.eval import _pearson
from train.minecraft.losses import importance_from_effect, agreement_loss, patch_pixel_diff


def _grid(vec_m):
    """[M] float 向量 → [h,w] 方形网格(np);非完全平方退化 1×M。"""
    arr = vec_m.detach().float().cpu().numpy()
    m = arr.shape[0]
    hw = int(round(math.sqrt(m)))
    return arr.reshape(hw, hw) if hw * hw == m else arr.reshape(1, m)


@torch.no_grad()
def visualize_minecraft(model, effect_tok, batch, cfg, device, amp_dev, use_amp,
                        out_path, epoch=0, n_rollout=5):
    """在一条固定 holdout 序列上渲染诊断面板 PNG。

    Parameters
    ----------
    model, effect_tok : nn.Module
        活模型与效应分词器(eval 模式)。
    batch : dict
        一个 holdout batch:img[B,T,3,H,W] uint8/float、act_agg[B,T-1,act_dim]、dt[B,T-1]。
    cfg : ModelConfig
        结构配置(取 predictor.n_context_cutoffs 等)。
    device, amp_dev : str
        计算设备 / autocast 设备类型。
    use_amp : bool
        是否启用混合精度(骨干+编码段;预测器段统一 fp32 诊断)。
    out_path : str
        PNG 输出路径。
    epoch : int
        当前 epoch(标题用)。
    n_rollout : int
        开环漂移推演步数。
    """
    model.eval()
    effect_tok.eval()
    d_rev = model.d_rev

    img = _to_float_img(batch["img"].to(device))
    act_agg = batch["act_agg"].to(device)
    dt = batch["dt"].to(device)
    B, T = img.shape[0], img.shape[1]

    with torch.autocast(device_type=amp_dev, enabled=use_amp):
        feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
        z, _ = model.encode(feats)
    M = z.shape[-2]
    # 预测器段统一 fp32(与 eval 同口径:autocast 仅覆盖骨干+编码,见 eval.evaluate)
    z = z.view(B, T, M, model.d).float()
    z_tgt = model.encode_target(feats).view(B, T, M, model.d).float()

    tf = torch.cat([torch.zeros(B, 1, device=dt.device), dt.cumsum(dim=1)], dim=1)
    target = T - 1
    act, t_act, query_t = act_agg[:, :target], tf[:, :target], tf[:, target]
    cuts = sorted({max(1, target // 2), target - 1})

    z_hats, e_norm, out = [], None, None
    model_err_by_cut = []
    for k in cuts:
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            out = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=False)
            out0 = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=True)
        z_hats.append(out["z_hat"])
        e_norm = (out["z_hat_inv"].float() - out0["z_hat_inv"].float()).norm(dim=-1)   # [B,M]
        se = (out["z_hat"].float() - z_tgt[:, target]).pow(2).mean(dim=-1)             # [B,M]
        model_err_by_cut.append(se.mean().item())

    # persistence 基线:预测"不变"= 锚点 z[:,0],对同一未来帧的误差
    persistence_err = (z[:, 0] - z_tgt[:, target]).pow(2).mean(dim=-1).mean().item()
    ratio = persistence_err and model_err_by_cut[-1] / persistence_err

    # 反捷径:w vs 未来潜发散 / 像素差
    w = importance_from_effect(e_norm)                                                  # [B,M]
    fdiv = (z_tgt[:, target, :, d_rev:] - z_tgt[:, 0, :, d_rev:]).norm(dim=-1)          # [B,M]
    pdiff = patch_pixel_diff(img, 0, target, M)                                         # [B,M]
    corr_future = _pearson(w, fdiv)
    corr_pixel = _pearson(w, pdiff)
    agree = agreement_loss(z_hats).item()

    # 开环漂移:用预测作锚点逐步外推,看 z_inv 增量是否有界
    drifts, zr, t_cur = [], z_hats[-1], query_t.clone()
    for _ in range(n_rollout):
        ctx = torch.stack([z[:, 0], zr], dim=1)
        tctx = torch.stack([tf[:, 0], t_cur], dim=1)
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            out2 = model(ctx, tctx, act, t_act, t_cur + 1.0, null=False)
        drifts.append((out2["z_hat_inv"].float() - zr[..., d_rev:].float()).norm(dim=-1).mean().item())
        zr, t_cur = out2["z_hat"], t_cur + 1.0

    # ---- 绘制 2×2 面板 ----
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    # A. Δz 误差 vs persistence
    labels = ["persist"] + [f"k={k}" for k in cuts]
    heights = [persistence_err] + model_err_by_cut
    colors = ["gray"] + ["tab:blue"] * len(cuts)
    ax[0, 0].bar(labels, heights, color=colors)
    ax[0, 0].set_title("A. Δz error: model vs persistence (lower=better)")
    ax[0, 0].set_ylabel("mean squared error")

    # B. 开环漂移
    ax[0, 1].plot(range(1, n_rollout + 1), drifts, "o-", color="tab:red")
    ax[0, 1].set_title("B. open-loop rollout drift")
    ax[0, 1].set_xlabel("rollout step"); ax[0, 1].set_ylabel("‖Δz_inv‖")
    ax[0, 1].set_ylim(bottom=0)

    # C. 后果权重 w (sample 0)
    im = ax[1, 0].imshow(_grid(w[0]), cmap="viridis")
    ax[1, 0].set_title("C. consequence weight w (patch grid)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    # D. 未来潜发散 fdiv (sample 0)
    im = ax[1, 1].imshow(_grid(fdiv[0]), cmap="magma")
    ax[1, 1].set_title(f"D. future lat div | corr(w,fut)={corr_future:+.2f} corr(w,pix)={corr_pixel:+.2f}")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)

    fig.suptitle(
        f"epoch {epoch} | align={model_err_by_cut[-1]:.3f} persistence={persistence_err:.3f} "
        f"ratio={ratio:.2f} | agree={agree:.4f} | corr(w,fut)={corr_future:+.2f} "
        f"corr(w,pix)={corr_pixel:+.2f}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
