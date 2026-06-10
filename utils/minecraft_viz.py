# -*- coding: utf-8 -*-
"""train_minecraft 训练效果可视化:一张 PNG 面板,直观回答"世界模型学到了没"。

证伪原则(与 train_probe 的 persistence/oracle 标尺同源):每个子图都带一个
不可作弊的对照,低 loss 本身不构成证据。

面板内容(固定一条验证序列,跨 epoch 可前后对比):
  A. 潜空间预测误差 vs persistence 基线(把"当前帧感知"原样当预测)。
     模型曲线低于基线才说明在"预测"而非"复读";灰色区为开环蒙眼段
     (patch 全零,只剩记忆+动作推演),该段误差温和增长 = 真前向模拟。
  B. 逆动力学键盘读出:GT 按键热图(上)vs 预测概率热图(下),按键×时间。
     两图条纹对齐 = Z 的变化里真的写进了动作效果。
  C. 鼠标 dx/dy 预测 vs GT 曲线。
  D. 可控闸 c 热图(slot×时间,逐维均值):应随训练出现亮暗分化(极化)。
  E. SlotBinder 注意力叠加:最局部化的几个 slot 各自盯着画面哪里(对象绑定)。

图内文字用英文:matplotlib 默认字体无 CJK,避免 Colab 上满图 tofu 方块。
"""
import os

import numpy as np
import torch

from utils.vpt_dataset import VPT_KEYS

N_MOUSE = 2


@torch.no_grad()
def collect_rollout(model, img, action, t_vec, device, open_loop_from=None):
    """跑一遍序列收集可视化轨迹。img: [B,T,3,H,W] float∈[0,1]。取样本 0。"""
    was_training = model.training
    model.eval()
    model.binder.attn.store_attn = True
    try:
        B, T = img.shape[:2]
        N, d = model.N, model.d
        act_dim = action.shape[-1]
        if open_loop_from is None:
            open_loop_from = max(2, (T - 1) // 2)

        # 批量预计算:目标编码 z_tg[t] = encode(frame t+1);当前帧编码 z_now[t] = encode(frame t)
        z_tg = model.encode_target(
            img[:, 1:].reshape(B * (T - 1), *img.shape[2:])).view(B, T - 1, N, d)
        z_now = torch.cat([model.encode_target(img[:, 0]).unsqueeze(1),
                           z_tg[:, :-1]], dim=1)                      # [B,T-1,N,d]
        patch_all = model.vision_encoder(
            img[:, :T - 1].reshape(B * (T - 1), *img.shape[2:])).view(B, T - 1, -1, d)
        zero_patch = torch.zeros_like(patch_all[:, 0])

        Z = torch.zeros(B, N, d, device=device)
        h = torch.zeros(B, 1, d, device=device)
        a_raw = torch.zeros(B, model.J, act_dim, device=device)

        traj = {k: [] for k in ["pred_err", "pers_err", "kb_pred", "kb_true",
                                "mouse_pred", "mouse_true", "c"]}
        attn_t = max(1, open_loop_from - 1)      # 取闭环末帧的 slot 注意力
        attn_map, attn_frame = None, None

        for t in range(T - 1):
            patch = patch_all[:, t] if t < open_loop_from else zero_patch
            a_raw = torch.cat([a_raw[:, 1:], action[:, t].unsqueeze(1)], dim=1)
            out = model(patch, Z, h, a_raw, t_vec[:, t])

            if t == attn_t and model.binder.attn.last_attn is not None:
                attn_map = model.binder.attn.last_attn[0].float().cpu().numpy()  # [N, M]
                attn_frame = img[0, t].float().cpu().numpy().transpose(1, 2, 0)

            tgt = z_tg[:, t]
            traj["pred_err"].append(
                (out["mu"].float() - tgt.float()).pow(2).mean().sqrt().item())
            traj["pers_err"].append(
                (z_now[:, t].float() - tgt.float()).pow(2).mean().sqrt().item())
            inv = model.inv_dyn((tgt - out["Z_enc"]) * out["c"])
            traj["kb_pred"].append(inv[0, N_MOUSE:].float().cpu().numpy())
            traj["kb_true"].append(action[0, t, N_MOUSE:].float().cpu().numpy())
            traj["mouse_pred"].append(inv[0, :N_MOUSE].float().cpu().numpy())
            traj["mouse_true"].append(action[0, t, :N_MOUSE].float().cpu().numpy())
            traj["c"].append(out["c"][0].float().mean(-1).cpu().numpy())  # [N]

            Z, h = out["mu"], out["h_next"]

        result = {k: np.stack(v) for k, v in traj.items()}
        result["open_loop_from"] = open_loop_from
        result["attn_map"] = attn_map            # [N, M] or None
        result["attn_frame"] = attn_frame        # [H, W, 3] or None
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

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3)

    # A. 预测误差 vs persistence,开环区灰色。
    # 对数轴:模型误差与 persistence 基线可能差几个量级,线性轴会把基线压成
    # 一条贴零的直线,完全读不出两者的相对关系(正是要看的东西)。
    ax = fig.add_subplot(gs[0, 0:2])
    pred_e = np.maximum(traj["pred_err"], 1e-6)
    pers_e = np.maximum(traj["pers_err"], 1e-6)
    ax.plot(steps, pred_e, "r-", lw=2, label="model |mu - z_next|")
    ax.plot(steps, pers_e, "g--", lw=1.5, label="persistence |z_now - z_next|")
    ax.axvspan(olf, Tm1 - 1, color="gray", alpha=0.18, label="OPEN-LOOP (blind)")
    ax.set_yscale("log")
    ratio = pred_e.mean() / max(pers_e.mean(), 1e-6)
    ax.text(0.02, 0.04,
            f"closed {pred_e[:olf].mean():.3g} | open {pred_e[olf:].mean():.3g} | "
            f"pers {pers_e.mean():.3g} | model/pers = {ratio:.2f} (<1 = beats copy)",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(fc="white", alpha=0.7, ec="none"))
    ax.set_xlabel("step"); ax.set_ylabel("latent RMS error (log)")
    ax.set_title("Latent prediction vs copy-baseline (lower & stable in blind zone = real model)")
    ax.legend(fontsize=8)

    # C. 鼠标
    ax = fig.add_subplot(gs[0, 2:4])
    ax.plot(steps, traj["mouse_true"][:, 0], "k-", lw=1.5, label="dx true")
    ax.plot(steps, traj["mouse_pred"][:, 0], "r--", lw=1.2, label="dx pred")
    ax.plot(steps, traj["mouse_true"][:, 1], "b-", lw=1.5, label="dy true")
    ax.plot(steps, traj["mouse_pred"][:, 1], "c--", lw=1.2, label="dy pred")
    ax.axvspan(olf, Tm1 - 1, color="gray", alpha=0.18)
    ax.set_xlabel("step"); ax.set_ylabel("camera (normalized)")
    ax.set_title("Inverse-dynamics mouse readout"); ax.legend(fontsize=7, ncol=2)

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

    # D. 可控闸 c
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

    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def visualize_minecraft(model, batch, device, out_path, title=""):
    """入口:对一个 batch(取样本 0)出一张面板 PNG。返回路径或 None(matplotlib 缺失)。"""
    img = batch["img"].to(device)
    img = img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()
    action = batch["action"].to(device)
    t_vec = batch["t_vec"].to(device)
    traj = collect_rollout(model, img, action, t_vec, device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    return render_panel(traj, out_path, title=title)
