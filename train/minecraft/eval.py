"""离线诊断评估:逆动力学读出 + 多步开环 rollout 保真度。

  evaluate      — 逆动力学评估(键盘平衡准确率/跳变召回、鼠标分箱、holdout Δz 预测比值、
                  ξ 的 best-of-K 与后验天花板、按转移类型分桶 pred)。loader 应来自 holdout split。
  rollout_probe — 从真首帧盲滚 H 步,按深度桶量 roll_err 与逆动力学读出衰减(规划可用性的真考题)。

两者都 @torch.no_grad、只读模型,不更新参数;指标口径与训练曲线对齐(详见函数 docstring)。
"""
import itertools

import torch

from domains.minecraft.vpt_action import camera_to_bin, CAMERA_BINS, N_MOUSE, ACTION_DIM as ACT_DIM
from train.minecraft._seq import roll_hist, _to_float_img


@torch.no_grad()
def evaluate(model, loader, device, steps, amp_dev, use_amp, open_k=4, remap=None):
    """逆动力学评估:能否从潜变化里反推出"做了什么操作"。

    键盘:平衡准确率 + **跳变**(onset/release recall)——VPT 数据里 w/attack 等键
    常整段按住,逐帧 balanced acc 会被"输出基率常数"的平凡解灌高;只有按下/松开
    瞬间的检出率才证明模型从 ΔZ 里读出了动作。
    鼠标:分箱准确率,分全帧 acc 与 **move_acc**(仅 GT 非中心 bin、即真动了
    鼠标的帧)——基率解(恒中心 bin)在 move_acc 上得 0,同 onset 逻辑。
    loader 应来自 **holdout split**(不进训练的 clip),量泛化而非记忆。

    pred_bestk(ξ 的诚实量尺,deep-yogurt-28 复盘新增):pred_move 用先验**均值**
    ——均值不携带任何新内容(就是期望),所以确定性框架下它永远碰不到比 persistence
    更好的开环。ξ 的真实价值在**采样多样性**:想象的若干种未来里有一种接近真实。
    故 best-of-K = 每步从先验采 open_k 个 ξ、取最小比值——衡量"先验分布是否覆盖
    真值"。clean 口径:输入用 z_obs(闭环,无复合误差),只隔离"ξ 采样有没有用":
    若 pred_bestk 明显 < pred_move,说明 ξ 在工作。
    """
    model.eval()
    tp = fp = fn = tn = 0
    on_tp = on_n = off_tp = off_n = 0
    m_hit = m_n = mv_hit = mv_n = 0
    # 子集拆分(in-context 重绑定的诚实读出,见 mental_world §6):被 remap 的子集键(remap.key_idx,
    # 已是 0..19 键空间列索引)语义须靠 context 推断 = 真信号;其余键恒等(白送)。混算会把真信号
    # 虚高稀释 ⇒ 分两套计数器。再把子集键 bal-acc 按窗口内步位置(早/晚半窗)分桶——晚步 h 累积更多
    # context,late>early 即「在用 context 适应」的近乎免费 dose-response 代理。
    sub_cols = remap.key_idx.tolist() if remap is not None else []
    rest_cols = [c for c in range(ACT_DIM - N_MOUSE) if c not in set(sub_cols)]
    has_sub = len(sub_cols) > 0
    grp = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
           for k in ("sub", "rest", "sub_early", "sub_late")}   # bal-acc 组件
    grp_on = {k: {"tp": 0, "n": 0} for k in ("sub", "rest")}    # onset recall
    # Stage 0 onset 诊断:onset 帧上 kb_prob 的分布(槽路 vs patch 旁路),阈值无关。
    # 直方图累加(O(1) 内存)→ recall@≥θ = prob 高于 θ 的占比。判 onset 是被 0.5
    # 硬阈误杀(降阈即抬头 = C-阈值)还是信号真没了;槽路 vs patch 落差 = 编码器丢没丢 onset。
    N_PB = 100
    on_pb_slot = torch.zeros(N_PB + 1, device=device)
    on_pb_patch = torch.zeros(N_PB + 1, device=device)
    # 全帧 slot kb_prob 按真值分桶 → 任意 θ 的 recall(held)/spec(off):查 onset@.2 那个工作点的
    # 误报率,确认低阈召回不是靠把 off 帧概率一起抬上去换的(精度侧)。
    held_pb_slot = torch.zeros(N_PB + 1, device=device)
    off_pb_slot = torch.zeros(N_PB + 1, device=device)

    def _acc(d, pred, true):
        d["tp"] += int((pred & true).sum());  d["fp"] += int((pred & ~true).sum())
        d["fn"] += int((~pred & true).sum()); d["tn"] += int((~pred & ~true).sum())

    pred_sum, pred_mv_sum, pred_bk_sum, pred_n = 0.0, 0.0, 0.0, 0
    pred_post_sum = 0.0          # 后验条件 pred(偷看真值 ξ)= ξ 通道信息天花板
    # 按转移类型分桶 pred 比值(动作维度,与 Δz 幅度正交)——诊断 0.58 是
    # "转头揭示新内容的不可预测下限"还是"可预测结构(平移/静止)被漏失"
    bkt_sum = {"turn": 0.0, "walk": 0.0, "still": 0.0}
    bkt_n = {"turn": 0, "walk": 0, "still": 0}
    center = (CAMERA_BINS - 1) // 2
    for batch in itertools.islice(loader, steps):
        img = _to_float_img(batch["img"].to(device))
        act_seq = batch["act_seq"].to(device)
        act_agg = batch["act_agg"].to(device)
        dt = batch["dt"].to(device)
        t_vec = batch["t_vec"].to(device)
        task_emb = batch.get("task_emb")
        task_emb = task_emb.to(device) if task_emb is not None else None
        B, T = img.shape[0], img.shape[1]
        # 控制重映射:eval 用 **disjoint 的 holdout 置换**(训练从没见过的控制方案),
        # 固定种子 ⇒ 每次 eval 同一组方案(低方差固定 eval 集)。量的是「靠观察掌握新控制」。
        if remap is not None:
            spec = remap.sample(B, device, holdout=True,
                                generator=torch.Generator().manual_seed(12345))
            act_seq, act_agg = remap.apply(act_seq, spec), remap.apply(act_agg, spec)
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
            featsBT = feats.view(B, T, *feats.shape[-2:])
            z_obs = model.encode_obs(
                feats=featsBT[:, :T - 1].reshape(B * (T - 1), *feats.shape[-2:])
            ).view(B, T - 1, model.N, model.d)
            z_tg = model.encode_target(feats=feats).view(B, T, model.N, model.d)
        z_obs, z_tg = z_obs.float(), z_tg.float()
        h = torch.zeros(B, 1, model.d, device=device)
        a_hist = torch.zeros(B, model.J, ACT_DIM, device=device)
        t_hist = torch.zeros(B, model.J, device=device)
        hv = torch.zeros(B, model.J, device=device)
        prev_true = None
        for t in range(T - 1):
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                out = model(z_obs[:, t], h, a_hist, act_seq[:, t], dt[:, t], t_vec[:, t],
                            t_hist=t_hist, hist_valid=hv, task_emb=task_emb)
            # 先存当前步的历史(roll_hist 返回新张量、不就地改 ⇒ 旧引用仍有效),
            # best-of-K 重前向须用与 out 同一份 pre-roll 历史
            a0, th0, hv0 = a_hist, t_hist, hv
            a_hist, t_hist, hv = roll_hist(a_hist, t_hist, hv, act_agg[:, t], dt[:, t])
            # holdout 上的 Δz 预测比值(与训练 pred 同定义:1.0=复读基线)——
            # 与训练曲线对照即泛化差距,这是面板之外唯一能连续监控它的地方
            dz = z_tg[:, t + 1] - z_tg[:, t]
            per = (out["mu"].float() - dz).square().mean(dim=(1, 2))
            den = dz.square().mean(dim=(1, 2))
            ratio = per / den.clamp(min=1e-3)
            moved_s = den > den.median()
            pred_sum += ratio.mean().item()
            pred_mv_sum += (ratio[moved_s].mean() if bool(moved_s.any())
                            else ratio.mean()).item()
            # best-of-K:每步从先验采 open_k 个 ξ,逐样本取最小比值(运动样本口径)
            if open_k > 1:
                den_c = den.clamp(min=1e-3)
                best = ratio.clone()
                mu_p, lv_p = model.xi_prior(z_obs[:, t], h, dt[:, t])
                for _ in range(open_k):
                    with torch.autocast(device_type=amp_dev, enabled=use_amp):
                        o_k = model(z_obs[:, t], h, a0, act_seq[:, t], dt[:, t],
                                    t_vec[:, t], t_hist=th0, hist_valid=hv0,
                                    task_emb=task_emb,
                                    xi=model.xi_sample(mu_p, lv_p))
                    rk = (o_k["mu"].float() - dz).square().mean(dim=(1, 2)) / den_c
                    best = torch.minimum(best, rk)
                pred_bk_sum += (best[moved_s].mean() if bool(moved_s.any())
                                else best.mean()).item()
            # 后验条件 pred:喂偷看真值 Δz 的后验 ξ(取均值,稳定读数)。
            # ≈ pred_move ⇒ ξ 通道关死/内容不可降;≪ pred_move ⇒ 通道有容量、瓶颈在先验猜不中。
            mu_q, _ = model.xi_posterior(z_obs[:, t], h, dt[:, t], dz)
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                o_post = model(z_obs[:, t], h, a0, act_seq[:, t], dt[:, t], t_vec[:, t],
                               t_hist=th0, hist_valid=hv0, task_emb=task_emb, xi=mu_q)
            r_post = (o_post["mu"].float() - dz).square().mean(dim=(1, 2)) / den.clamp(min=1e-3)
            pred_post_sum += (r_post[moved_s].mean() if bool(moved_s.any())
                              else r_post.mean()).item()
            # 转移类型分桶(soft-floor 比值,同 loss 口径——堵住近静止样本 den→0 除爆):
            #   still = 低 Δz 锚(den ≤ 中位数,模型平凡复读,≈floor 行为);
            #   在真运动样本(den > 中位数)中再按相机偏移分:
            #   turn = 相机偏移 ≥2 bin(转头揭示新内容,疑不可降)/ walk = <2(平移/破坏方块,应可预测)。
            floor_b = (0.1 * den.mean()).clamp(min=1e-3)
            ratio_sf = per / (den + floor_b)
            moved_b = den > den.median()
            cam_dev = (camera_to_bin(act_agg[:, t, :N_MOUSE]) - center).abs().amax(-1)
            sels = {"turn": moved_b & (cam_dev >= 2), "walk": moved_b & (cam_dev < 2),
                    "still": ~moved_b}
            for name, sel in sels.items():
                if bool(sel.any()):
                    bkt_sum[name] += ratio_sf[sel].sum().item()
                    bkt_n[name] += int(sel.sum().item())
            pred_n += 1
            pdz = (featsBT[:, t + 1].mean(1) - featsBT[:, t].mean(1)).float()
            ctx_h = h.squeeze(1).float() if model.inv_dyn.use_ctx else None  # pre-step h(无泄漏)
            mouse_logits, kb_prob, parts = model.inv_dyn(    # 槽路诚实读出 + 接住 patch 旁路(诊断)
                (z_tg[:, t + 1] - z_obs[:, t]) * out["c"].float(), patch_dz=pdz, ctx=ctx_h)
            kb_prob_patch = torch.sigmoid(parts[1]) if parts is not None else None
            kb_pred = (kb_prob > 0.5)
            kb_true = (act_agg[:, t, N_MOUSE:] > 0.5)
            tp += (kb_pred & kb_true).sum().item();  fp += (kb_pred & ~kb_true).sum().item()
            fn += (~kb_pred & kb_true).sum().item();  tn += (~kb_pred & ~kb_true).sum().item()
            ph = kb_prob[kb_true].clamp(0, 1)        # held / off 帧 prob 直方图(任意 θ 的 recall/spec)
            if ph.numel():
                held_pb_slot.index_add_(0, (ph * N_PB).long(), torch.ones_like(ph))
            po = kb_prob[~kb_true].clamp(0, 1)
            if po.numel():
                off_pb_slot.index_add_(0, (po * N_PB).long(), torch.ones_like(po))
            if prev_true is not None:
                onset = kb_true & ~prev_true; release = ~kb_true & prev_true
                on_tp += (kb_pred & onset).sum().item();    on_n += onset.sum().item()
                off_tp += (~kb_pred & release).sum().item(); off_n += release.sum().item()
                if bool(onset.any()):          # Stage 0:onset 帧 prob 直方图(槽路 + patch 路)
                    ps = kb_prob[onset].clamp(0, 1)
                    on_pb_slot.index_add_(0, (ps * N_PB).long(), torch.ones_like(ps))
                    if kb_prob_patch is not None:
                        pp = kb_prob_patch[onset].clamp(0, 1)
                        on_pb_patch.index_add_(0, (pp * N_PB).long(), torch.ones_like(pp))
            # 子集/其余拆分 + 窗口内位置桶(用 prev_true 的旧值算子集 onset,故在更新前)
            if has_sub:
                sp, st = kb_pred[:, sub_cols], kb_true[:, sub_cols]
                _acc(grp["sub"], sp, st)
                _acc(grp["sub_late" if t >= (T - 1) // 2 else "sub_early"], sp, st)
                if rest_cols:
                    _acc(grp["rest"], kb_pred[:, rest_cols], kb_true[:, rest_cols])
                if prev_true is not None:
                    s_on = st & ~prev_true[:, sub_cols]
                    grp_on["sub"]["tp"] += int((sp & s_on).sum())
                    grp_on["sub"]["n"] += int(s_on.sum())
                    if rest_cols:
                        rp, rt = kb_pred[:, rest_cols], kb_true[:, rest_cols]
                        r_on = rt & ~prev_true[:, rest_cols]
                        grp_on["rest"]["tp"] += int((rp & r_on).sum())
                        grp_on["rest"]["n"] += int(r_on.sum())
            prev_true = kb_true
            mb_pred = mouse_logits.argmax(-1)
            mb_true = camera_to_bin(act_agg[:, t, :N_MOUSE])
            hit = (mb_pred == mb_true)
            m_hit += hit.sum().item(); m_n += hit.numel()
            moved = (mb_true != center)
            mv_hit += (hit & moved).sum().item(); mv_n += moved.sum().item()
            h = out["h_next"]
    nan = float("nan")

    def _bal(d):
        rc = d["tp"] / max(d["tp"] + d["fn"], 1); sp = d["tn"] / max(d["tn"] + d["fp"], 1)
        return 0.5 * (rc + sp)

    def _on_rec(hist, th):       # onset recall@(prob≥th),从直方图读
        tot = hist.sum()
        return (hist[int(round(th * N_PB)):].sum() / tot).item() if tot >= 1 else nan

    def _on_med(hist):           # onset 帧 prob 中位数(ramp 落点)
        tot = hist.sum()
        if tot < 1:
            return nan
        cdf = torch.cumsum(hist, 0) / tot
        idx = torch.searchsorted(cdf, torch.tensor(0.5, device=hist.device)).clamp(max=N_PB)
        return idx.item() / N_PB

    rec20 = _on_rec(held_pb_slot, 0.2)           # θ=0.2 工作点(= onset@.2 那个点)的整体 recall/spec
    spec20 = 1.0 - _on_rec(off_pb_slot, 0.2)     # spec@.2 仍高 ⇒ onset@.2 召回非靠误报灌(精度检查)
    recall = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"pred": pred_sum / max(pred_n, 1),
            "pred_move": pred_mv_sum / max(pred_n, 1),
            "pred_bestk": pred_bk_sum / max(pred_n, 1),   # ξ 价值:明显 < pred_move = 有用
            "kb_recall": recall, "kb_spec": spec, "kb_bal_acc": 0.5 * (recall + spec),
            "kb_onset_recall": on_tp / max(on_n, 1),
            "kb_release_recall": off_tp / max(off_n, 1),
            "kb_edges": on_n + off_n,
            # Stage 0 onset 阈值诊断(槽路 vs patch 旁路;recall@≥θ + onset 帧 prob 中位数)
            "onset_slot_r50": _on_rec(on_pb_slot, 0.5), "onset_slot_r35": _on_rec(on_pb_slot, 0.35),
            "onset_slot_r20": _on_rec(on_pb_slot, 0.2), "onset_slot_r10": _on_rec(on_pb_slot, 0.1),
            "onset_slot_med": _on_med(on_pb_slot),
            "onset_patch_r50": _on_rec(on_pb_patch, 0.5), "onset_patch_r35": _on_rec(on_pb_patch, 0.35),
            "onset_patch_r20": _on_rec(on_pb_patch, 0.2), "onset_patch_r10": _on_rec(on_pb_patch, 0.1),
            "onset_patch_med": _on_med(on_pb_patch),
            # θ=0.2 工作点精度检查(spec@.2 仍高 ⇒ onset@.2 召回不是靠误报灌的)
            "kb_recall20": rec20, "kb_spec20": spec20, "kb_bal20": 0.5 * (rec20 + spec20),
            # 子集拆分(remap 开时才有意义,否则 nan):重绑定的诚实读数
            "kb_onset_sub": grp_on["sub"]["tp"] / max(grp_on["sub"]["n"], 1) if has_sub else nan,
            "kb_onset_rest": (grp_on["rest"]["tp"] / max(grp_on["rest"]["n"], 1)
                              if has_sub and rest_cols else nan),
            "kb_bal_sub": _bal(grp["sub"]) if has_sub else nan,
            "kb_bal_rest": _bal(grp["rest"]) if has_sub and rest_cols else nan,
            "kb_sub_early": _bal(grp["sub_early"]) if has_sub else nan,   # context 少
            "kb_sub_late": _bal(grp["sub_late"]) if has_sub else nan,     # context 多;>early=在适应
            "pred_post": pred_post_sum / max(pred_n, 1),       # ξ 通道天花板:≈pred_move=通道关死
            "pred_turn": bkt_sum["turn"] / max(bkt_n["turn"], 1),    # 转头(疑不可降)
            "pred_walk": bkt_sum["walk"] / max(bkt_n["walk"], 1),    # 平移(应可预测,高=预测器漏失)
            "pred_still": bkt_sum["still"] / max(bkt_n["still"], 1), # 静止(≈floor,锚)
            "frac_turn": bkt_n["turn"] / max(sum(bkt_n.values()), 1),
            "frac_walk": bkt_n["walk"] / max(sum(bkt_n.values()), 1),
            "mouse_bin_acc": m_hit / max(m_n, 1),
            "mouse_move_acc": mv_hit / max(mv_n, 1),
            "mouse_moves": mv_n}


@torch.no_grad()
def rollout_probe(model, loader, device, steps, amp_dev, use_amp, horizon=16):
    """多步开环 rollout 保真度(脑内 rollout 能否用于规划的真考题)。从真首帧 z_obs[0]
    起,用真动作序列在潜空间盲滚 H 步(ξ=先验均值,h 携带),按 rollout 深度 d 量:
      roll_err = ‖ẑ_d − z_tg[d]‖² / ‖z_tg[d] − z_tg[0]‖²
                 (frozen@start 基线 = 1.0;<1 = 滚动胜过"原地不动",>1 = 滚飞了)
      OL_move/OL_onset = 从滚出的 μ_d 经逆动力学读动作的 move_acc/onset_recall
                 (rollout 无真未来 patch ⇒ patch_dz=None,仅槽路)
      CL_move/CL_onset = 从真 Δz 读(逆动力学天花板参考,近深度无关)
    判读:OL 读出随深度衰减多快 = 滚出的潜还能保住多少控制相关结构。若到深度
    5~10 仍 ≈ CL ⇒ 世界模型可用于多步规划;若深度 2 就塌到 chance ⇒ 不可用,
    且这才是该修的真瓶颈(开环训练 / 可用的 ξ)。"""
    model.eval()
    center = (CAMERA_BINS - 1) // 2
    # 深度分桶池化(per-depth 在小集上太噪);比值用"批方差之比"(分子分母分别累加再除,
    # 杀浅层除爆);CL 走独立闭环前向(真 z_obs + 闭环 c)= 干净的逆动力学天花板。
    buckets = [(0, 2, "1-2"), (2, 5, "3-5"), (5, 10, "6-10"), (10, 16, "11-16")]
    agg = {nm: {"rn": 0.0, "rd": 0.0, "olh": 0, "oln": 0, "clh": 0, "cln": 0}
           for _, _, nm in buckets}

    def _bk(d):
        for lo, hi, nm in buckets:
            if lo <= d < hi:
                return agg[nm]
        return None

    for batch in itertools.islice(loader, steps):
        img = _to_float_img(batch["img"].to(device))
        act_seq = batch["act_seq"].to(device)
        act_agg = batch["act_agg"].to(device)
        dt = batch["dt"].to(device); t_vec = batch["t_vec"].to(device)
        task_emb = batch.get("task_emb")
        task_emb = task_emb.to(device) if task_emb is not None else None
        B, T = img.shape[0], img.shape[1]
        Hb = min(horizon, T - 1)
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
            featsBT = feats.view(B, T, *feats.shape[-2:])
            z_obs = model.encode_obs(
                feats=featsBT[:, :T - 1].reshape(B * (T - 1), *feats.shape[-2:])
            ).view(B, T - 1, model.N, model.d).float()
            z_tg = model.encode_target(feats=feats).view(B, T, model.N, model.d).float()
        z0 = z_tg[:, 0]
        zhat = z_obs[:, 0]                                          # 开环状态
        h_ol = torch.zeros(B, 1, model.d, device=device)
        h_cl = torch.zeros(B, 1, model.d, device=device)
        zero_j = torch.zeros(B, model.J, ACT_DIM, device=device)
        zj = torch.zeros(B, model.J, device=device)
        a_ol, t_ol, hv_ol = zero_j, zj, zj
        a_cl, t_cl, hv_cl = zero_j.clone(), zj.clone(), zj.clone()
        for d in range(Hb):
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                out_cl = model(z_obs[:, d], h_cl, a_cl, act_seq[:, d], dt[:, d], t_vec[:, d],
                               t_hist=t_cl, hist_valid=hv_cl, task_emb=task_emb)
                out = model(zhat, h_ol, a_ol, act_seq[:, d], dt[:, d], t_vec[:, d],
                            t_hist=t_ol, hist_valid=hv_ol, task_emb=task_emb)
            a_cl, t_cl, hv_cl = roll_hist(a_cl, t_cl, hv_cl, act_agg[:, d], dt[:, d])
            a_ol, t_ol, hv_ol = roll_hist(a_ol, t_ol, hv_ol, act_agg[:, d], dt[:, d])
            mu = out["mu"].float()
            zhat_next = zhat + mu
            pdz = (featsBT[:, d + 1].mean(1) - featsBT[:, d].mean(1)).float()
            uc = model.inv_dyn.use_ctx
            cl_logits, _, _ = model.inv_dyn(   # 真 Δz + 槽路:逆动力学天花板(同槽路,与 OL 可比)
                (z_tg[:, d + 1] - z_obs[:, d]) * out_cl["c"].float(), patch_dz=pdz,
                ctx=h_cl.squeeze(1).float() if uc else None)
            ol_logits, _, _ = model.inv_dyn(mu * out["c"].float(), patch_dz=None,    # 从滚出 μ 读
                                            ctx=h_ol.squeeze(1).float() if uc else None)
            A = _bk(d)
            if A is not None:
                A["rn"] += ((zhat_next - z_tg[:, d + 1]) ** 2).sum().item()
                A["rd"] += ((z0 - z_tg[:, d + 1]) ** 2).sum().item()                 # frozen@start
                mb_true = camera_to_bin(act_agg[:, d, :N_MOUSE]); moved = (mb_true != center)
                A["olh"] += ((ol_logits.argmax(-1) == mb_true) & moved).sum().item()
                A["clh"] += ((cl_logits.argmax(-1) == mb_true) & moved).sum().item()
                A["oln"] += moved.sum().item(); A["cln"] += moved.sum().item()
            zhat = zhat_next; h_ol = out["h_next"]; h_cl = out_cl["h_next"]
    print("\n--- 多步开环 rollout 保真度(真首帧盲滚,真动作,ξ先验均值)---")
    print(f"  {'深度桶':>6} {'vs_freeze':>10} {'OL_move':>8} {'CL_move':>8}   "
          f"(vs_freeze<1=胜原地不动;OL_move→CL=滚出潜仍可读出动作)")
    for _, _, nm in buckets:
        a = agg[nm]
        if a["oln"] == 0:
            continue
        print(f"  {nm:>6} {a['rn']/max(a['rd'],1e-9):>10.3f} "
              f"{a['olh']/max(a['oln'],1):>8.3f} {a['clh']/max(a['cln'],1):>8.3f}")
