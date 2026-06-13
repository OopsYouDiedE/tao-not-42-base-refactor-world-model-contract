# -*- coding: utf-8 -*-
"""train_minecraft 训练效果可视化:一张 PNG 面板,直观回答"世界模型学到了没"。

证伪原则(与 train_probe 的 persistence/oracle 标尺同源):每个子图都带一个
不可作弊的对照,低 loss 本身不构成证据。

面板内容(固定一条验证序列,跨 epoch 可前后对比):
  A. Δz 预测误差 |μ − Δz| vs 零运动基线 |Δz|(persistence = 预测"什么都不变")。
     模型曲线低于基线才说明在"预测变化"而非沉默;灰色区为开环蒙眼段
     (感知输入换成自身预测 ẑ+μ 累积,只剩记忆+动作推演)。
  B. 逆动力学键盘读出:GT 按键热图(上)vs 预测概率热图(下),按键×时间。
     两图条纹对齐 = Δz 里真的写进了动作效果。
  C. 鼠标 dx/dy:GT 连续曲线 vs 分箱预测解码值(mu-law bin 中心,故呈阶梯状)。
  D. 可控闸 c 热图(slot×时间,逐 slot 标量):应随训练出现亮暗分化(极化)。
  E. SlotBinder 注意力叠加:最局部化的几个 slot 各自盯着画面哪里(对象绑定)。
  F. 未来动作规划(BC):槽 0 的下一转移键盘预测热图(与 B 的 GT 左移一格对照)
     + 开环起点的 K 槽 onset/duration/exist vs 真实未来(时间锚:0 = "现在",帧)。

图内文字用英文:matplotlib 默认字体无 CJK,避免 Colab 上满图 tofu 方块。
"""
import os

import numpy as np
import torch

from train.minecraft.vpt_dataset import VPT_KEYS
from train.minecraft.vpt_action import bin_to_camera

N_MOUSE = 2


@torch.no_grad()
def collect_rollout(model, img, act_seq, act_agg, dt, t_vec, device, open_loop_from=None,
                    task_emb=None):
    """跑一遍序列收集可视化轨迹。img: [B,T,3,H,W] float∈[0,1]。取样本 0。

    闭环段:感知输入 = encode_obs(img_t);开环段(t >= open_loop_from):
    感知输入 = 上一步预测 ẑ = z_ref + μ 的累积(训练是 teacher forcing,
    开环漂移速度是"学没学到可推演动力学"的检验,不是训练目标)。
    序列应来自 holdout clip(train_minecraft 已按 split 切分)。
    """
    was_training = model.training
    model.eval()
    try:
        B, T = img.shape[:2]
        N, d = model.N, model.d
        if open_loop_from is None:
            open_loop_from = max(2, (T - 1) // 2)

        # 批量预计算:冻结骨干特征一次提取,在线编码与 EMA 目标编码共用
        feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
        feats = feats.view(B, T, *feats.shape[-2:])
        z_obs = model.encode_obs(
            feats=feats[:, :T - 1].reshape(B * (T - 1), *feats.shape[-2:])
        ).view(B, T - 1, N, d).float()
        z_tg = model.encode_target(
            feats=feats.reshape(B * T, *feats.shape[-2:])).view(B, T, N, d).float()

        # slot 注意力:单独对 attn_t 帧再编码一次(store_attn 走慢路径,只跑一帧)
        attn_t = max(1, open_loop_from - 1)
        model.binder.attn.store_attn = True
        model.encode_obs(feats=feats[:, attn_t])
        attn_map = (model.binder.attn.last_attn[0].float().cpu().numpy()
                    if model.binder.attn.last_attn is not None else None)
        model.binder.attn.store_attn = False
        attn_frame = img[0, attn_t].float().cpu().numpy().transpose(1, 2, 0)

        h = torch.zeros(B, 1, d, device=device)
        a_hist = torch.zeros(B, model.J, act_agg.shape[-1], device=device)
        t_hist = torch.zeros(B, model.J, device=device)   # 历史条目距"现在"的帧数
        hv = torch.zeros(B, model.J, device=device)       # 历史有效位(空槽=0)
        Z_state = z_obs[:, 0]                      # 开环推演的滚动状态(锚坐标系)

        traj = {k: [] for k in ["pred_err", "pers_err", "mu_norm", "kb_pred", "kb_true",
                                "mouse_pred", "mouse_true", "c", "plan_kb"]}
        plan_snap = None

        for t in range(T - 1):
            z_ref = z_obs[:, t] if t < open_loop_from else Z_state
            out = model(z_ref, h, a_hist, act_seq[:, t], dt[:, t], t_vec[:, t],
                        t_hist=t_hist, hist_valid=hv, task_emb=task_emb)
            a_hist = torch.cat([a_hist[:, 1:], act_agg[:, t].unsqueeze(1)], dim=1)
            t_hist = torch.cat([t_hist[:, 1:] + dt[:, t].unsqueeze(1),
                                torch.zeros(B, 1, device=device)], dim=1)
            hv = torch.cat([hv[:, 1:], torch.ones(B, 1, device=device)], dim=1)
            mu, c = out["mu"].float(), out["c"].float()

            dz = z_tg[:, t + 1] - z_tg[:, t]
            traj["pred_err"].append((mu - dz).pow(2).mean().sqrt().item())
            traj["pers_err"].append(dz.pow(2).mean().sqrt().item())
            traj["mu_norm"].append(mu.pow(2).mean().sqrt().item())

            pdz = (feats[:, t + 1].mean(1) - feats[:, t].mean(1)).float()
            mouse_logits, kb_prob = model.inv_dyn(
                (z_tg[:, t + 1] - z_obs[:, t]) * c, patch_dz=pdz)
            traj["kb_pred"].append(kb_prob[0].float().cpu().numpy())
            traj["kb_true"].append(act_agg[0, t, N_MOUSE:].float().cpu().numpy())
            traj["mouse_pred"].append(
                bin_to_camera(mouse_logits[0].argmax(-1)).float().cpu().numpy())
            traj["mouse_true"].append(act_agg[0, t, :N_MOUSE].float().cpu().numpy())
            traj["c"].append(c[0].squeeze(-1).cpu().numpy())          # [N] 逐 slot 标量

            # 未来动作规划:槽 0 = 下一个转移的键盘预测(逐步收集 → 热图与 GT 对照);
            # 在开环起点拍一张完整快照(K 槽的 onset/duration/exist vs 真实未来)
            plan = out["action_plan"]
            traj["plan_kb"].append(plan["keyboard"][0, 0].float().cpu().numpy())
            if t == open_loop_from:
                n = min(model.K, (T - 1) - (t + 1))
                if n > 0:
                    fdt = dt[0, t + 1:t + 1 + n].float()
                    onset_tgt = dt[0, t].float() + torch.cat(
                        [torch.zeros(1, device=fdt.device), fdt.cumsum(0)[:-1]])
                    plan_snap = {
                        "onset": plan["onset"][0].float().cpu().numpy(),
                        "dur": plan["duration"][0].float().cpu().numpy(),
                        "exist": plan["exist"][0].float().cpu().numpy(),
                        "onset_tgt": onset_tgt.cpu().numpy(),
                        "dur_tgt": fdt.cpu().numpy(),
                    }

            Z_state = z_ref + mu                   # ẑ(t+dt) = 当前估计 + 预测增量
            h = out["h_next"]

        result = {k: np.stack(v) for k, v in traj.items()}
        result["open_loop_from"] = open_loop_from
        result["attn_map"] = attn_map            # [N, M] or None
        result["attn_frame"] = attn_frame        # [H, W, 3] or None
        result["plan_snap"] = plan_snap          # 开环起点的规划快照 or None
        return result
    finally:
        model.binder.attn.store_attn = False
        if was_training:
            model.train()


def _attn_overlays(attn_map, frame, n_show=3):
    """挑最局部化(峰值-均值最大)且**空间互异**的 n_show 个 slot。

    只按 peakiness 排序时,多个冗余 slot 盯同一显著区域(如快捷栏 UI)会霸占
    全部面板,看不出 slot 间有没有分化——贪心跳过峰值位置与已选 slot 过近者,
    凑不够再用次峰值的补足。返回 [(slot_idx, HxW 热图)]。
    """
    import cv2
    N, M = attn_map.shape
    g = int(round(M ** 0.5))
    if g * g != M:
        return []
    H, W = frame.shape[:2]
    peaky = attn_map.max(axis=1) - attn_map.mean(axis=1)
    order = np.argsort(-peaky)
    peak_yx = np.stack(np.unravel_index(attn_map.argmax(axis=1), (g, g)), axis=1)
    chosen, min_dist = [], g / 4.0
    for si in order:                      # 第一轮:要求峰值位置互异
        if len(chosen) >= n_show:
            break
        if all(np.linalg.norm(peak_yx[si] - peak_yx[sj]) >= min_dist for sj in chosen):
            chosen.append(int(si))
    for si in order:                      # 补足:全都挤在一处时退化为纯 peakiness 排序
        if len(chosen) >= n_show:
            break
        if int(si) not in chosen:
            chosen.append(int(si))
    outs = []
    for si in chosen:
        m = attn_map[si].reshape(g, g)
        m = (m - m.min()) / max(m.max() - m.min(), 1e-8)
        outs.append((si, cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)))
    return outs


def render_panel(traj, out_path, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[viz skipped: {e}]")
        return None

    Tm1 = len(traj["pred_err"])
    olf = traj["open_loop_from"]
    steps = np.arange(Tm1)
    key_names = [k.replace("key_", "") for k in VPT_KEYS]

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.3)

    # A. Δz 预测误差 vs 零运动基线,开环区灰色。
    # 对数轴:两条线可能差量级,线性轴会把小的一条压成贴零直线。
    ax = fig.add_subplot(gs[0, 0:2])
    pred_e = np.maximum(traj["pred_err"], 1e-6)
    pers_e = np.maximum(traj["pers_err"], 1e-6)
    mu_n = np.maximum(traj["mu_norm"], 1e-6)
    ax.plot(steps, pred_e, "r-", lw=2, label="model |mu - dz|")
    ax.plot(steps, pers_e, "g--", lw=1.5, label="zero-motion |dz| (persistence)")
    # |mu|:区分"摆烂(μ≡0,本线趴底)"与"在学但解释方差还小"。注意指标算术:
    # pred(平方比)=0.9 ⇒ 红线只比绿线低 5%(RMS+对数轴),视觉必然"贴合";
    # 此时最优 |mu| ≈ √(1−0.9)·|dz| ≈ 0.32|dz|——看蓝线是否朝绿线抬升。
    ax.plot(steps, mu_n, "b-", lw=1.2, alpha=0.8,
            label="model |mu| (0 = no-motion policy)")
    ax.axvspan(olf, Tm1 - 1, color="gray", alpha=0.18, label="OPEN-LOOP (blind)")
    ax.set_yscale("log")
    ratio = pred_e.mean() / max(pers_e.mean(), 1e-6)
    ax.text(0.02, 0.04,
            f"closed {pred_e[:olf].mean():.3g} | open {pred_e[olf:].mean():.3g} | "
            f"|dz| {pers_e.mean():.3g} | |mu| {mu_n.mean():.3g} | "
            f"model/pers = {ratio:.2f} (<1 = beats copy)",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(fc="white", alpha=0.7, ec="none"))
    ax.set_xlabel("step"); ax.set_ylabel("latent RMS error (log)")
    ax.set_title("Delta-latent prediction vs zero-motion baseline (<1 = real model)")
    ax.legend(fontsize=8)

    # C. 鼠标(GT 连续 vs 分箱解码——预测呈阶梯状是分箱所致,正常)
    ax = fig.add_subplot(gs[0, 2:4])
    ax.plot(steps, traj["mouse_true"][:, 0], "k-", lw=1.5, label="dx true")
    ax.plot(steps, traj["mouse_pred"][:, 0], "r--", lw=1.2, label="dx pred (bin)")
    ax.plot(steps, traj["mouse_true"][:, 1], "b-", lw=1.5, label="dy true")
    ax.plot(steps, traj["mouse_pred"][:, 1], "c--", lw=1.2, label="dy pred (bin)")
    ax.axvspan(olf, Tm1 - 1, color="gray", alpha=0.18)
    ax.set_xlabel("step"); ax.set_ylabel("camera (normalized)")
    ax.set_title("Inverse-dynamics mouse readout (mu-law binned)")
    ax.legend(fontsize=7, ncol=2)

    # B. 键盘 GT / 预测热图(共享 0..1 色标)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(traj["kb_true"].T, aspect="auto", cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_yticks(range(len(key_names))); ax.set_yticklabels(key_names, fontsize=5)
    ax.axvline(olf, color="orange", lw=1)
    ax.set_title("Keyboard GT (keys x time)")

    ax = fig.add_subplot(gs[2, 0:2])
    ax.imshow(traj["kb_pred"].T, aspect="auto", cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_yticks(range(len(key_names))); ax.set_yticklabels(key_names, fontsize=5)
    ax.axvline(olf, color="orange", lw=1)
    ax.set_xlabel("step")
    ax.set_title("Keyboard inv-dyn prediction (stripes should match GT above)")

    # D. 可控闸 c(逐 slot 标量)
    ax = fig.add_subplot(gs[1, 2:4])
    im = ax.imshow(traj["c"].T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.axvline(olf, color="orange", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("slot")
    ax.set_title("Controllability gate c (should polarize during training)")
    fig.colorbar(im, ax=ax, fraction=0.025)

    # E. slot 注意力叠加
    if traj["attn_map"] is not None and traj["attn_frame"] is not None:
        overlays = _attn_overlays(traj["attn_map"], traj["attn_frame"])
        sub = gs[2, 2:4].subgridspec(1, max(len(overlays) + 1, 2), wspace=0.05)
        ax = fig.add_subplot(sub[0, 0])
        ax.imshow(np.clip(traj["attn_frame"], 0, 1)); ax.axis("off")
        ax.set_title("frame", fontsize=8)
        for i, (si, heat) in enumerate(overlays):
            ax = fig.add_subplot(sub[0, i + 1])
            ax.imshow(np.clip(traj["attn_frame"], 0, 1))
            ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.axis("off"); ax.set_title(f"slot {si} binds", fontsize=8)

    # F1. 规划槽 0 = "下一个转移"的键盘预测(条纹应与 B 的 GT 左移一格对齐)
    ax = fig.add_subplot(gs[3, 0:2])
    ax.imshow(traj["plan_kb"].T, aspect="auto", cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_yticks(range(len(key_names))); ax.set_yticklabels(key_names, fontsize=5)
    ax.axvline(olf, color="orange", lw=1)
    ax.set_xlabel("step")
    ax.set_title("Plan slot-0: NEXT-transition keyboard (compare GT shifted 1 left)")

    # F2. 开环起点的完整规划快照:K 槽 onset/duration vs 真实未来(单位:帧)
    ax = fig.add_subplot(gs[3, 2:4])
    snap = traj.get("plan_snap")
    if snap is not None:
        K = len(snap["onset"]); nt = len(snap["onset_tgt"])
        ks = np.arange(K)
        ax.plot(ks, snap["onset"], "r.-", lw=1.5, label="onset pred")
        ax.plot(np.arange(nt), snap["onset_tgt"], "g.--", lw=1.5, label="onset target")
        ax.bar(ks - 0.15, snap["dur"], width=0.3, alpha=0.35, color="r", label="dur pred")
        ax.bar(np.arange(nt) + 0.15, snap["dur_tgt"], width=0.3, alpha=0.35, color="g",
               label="dur target")
        for k in ks:                                # exist 概率标在槽上方
            ax.text(k, float(snap["onset"][k]) + 0.5, f"{snap['exist'][k]:.2f}",
                    fontsize=6, ha="center")
        ax.set_xlabel("plan slot k (slot k = (k+1)-th future transition)")
        ax.set_ylabel("frames from 'now'")
        ax.set_title("Action plan @open-loop start (exist prob above slots)")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def visualize_minecraft(model, batch, device, out_path, title="", task_emb=None):
    """入口:对一个 batch(取样本 0)出一张面板 PNG。返回路径或 None(matplotlib 缺失)。"""
    img = batch["img"].to(device)
    img = img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()
    traj = collect_rollout(model, img,
                           batch["act_seq"].to(device), batch["act_agg"].to(device),
                           batch["dt"].to(device), batch["t_vec"].to(device), device,
                           task_emb=task_emb.to(device) if task_emb is not None else None)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    return render_panel(traj, out_path, title=title)
