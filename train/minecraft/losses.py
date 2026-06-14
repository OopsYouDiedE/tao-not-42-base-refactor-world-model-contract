"""MinecraftWorldModel 自监督训练的损失函数。

  dz_pred_loss          — 逐样本归一化 Δz 预测(soft-floor + Huber);返回可读比值。
  slot_diversity_loss   — 竞争注意力图成对重叠软惩罚(修 slot 冗余)。
  minecraft_inv_dyn_loss— 逆动力学:从 (z_tg−z_obs)⊙c 反推混合动作(槽路/patch 分开监督)。
  plan_bc_loss          — 未来动作规划的行为克隆(查询槽 ↔ 未来转移,时间序对齐)。
  kl_diag_gauss         — 对角高斯 KL(q‖p),随机隐变量 ξ 通道用。

各项的物理动机与防坍缩/防平凡解的设计理由见函数 docstring;总损失装配见 train_minecraft。
"""
import torch
import torch.nn.functional as F

from domains.minecraft.vpt_action import camera_to_bin, CAMERA_BINS, N_MOUSE

EPS = 1e-4          # I1


def dz_pred_loss(mu, dz_tg, eps=1e-3):
    """逐样本归一化 Δz 预测损失,分母带**软地板**。返回 (loss, pred, pred_move)。

    归一化的作用:(1) 损失值直接可读——就是 (model/persistence)² 比值,1.0 = 复读
    基线;(2) 对编码尺度不变,堵死"经 EMA 把变化抹平"的间接捷径;(3) **可变 Δt 下
    必须逐样本归一**:同一 batch 混合不同跨度,大 Δt 的 |Δz| 大,全局分母会让长跨度
    样本主导损失、小间隔的精细动力学被相对忽略。

    软地板(loss 用 denom + 0.1·batch均值,而非 clamp(min=eps)):Minecraft 大量
    转移近似静止——其 Δz 几乎纯是编码噪声,分子分母同源 ⇒ 比值恒 ≈1,**无论怎么
    学都下不去**,还与真运动样本平权稀释梯度(实测 pred 被这批样本托底在 ~0.9)。
    加软地板后静止样本损失 ≈ denom/floor << 1 自动降权,真运动样本(denom >> floor)
    仍是比值归一;地板取 batch 均值的 10%,随表征尺度漂移自适应,不引新超参。

    指标(均旧定义比值,可与历史曲线对齐):pred = 全样本均值(被静止样本托底,
    读数偏保守);pred_move = 分母高于 batch 中位数的"真运动"样本均值——**诚实
    读数**,模型有没有学动力学看它。注意诚实下限:转身揭示的新内容在潜空间本质
    不可预测,好模型的 pred_move 预期 ~0.5-0.7,不应拿 0 当目标。
    """
    r = mu - dz_tg
    denom = dz_tg.square().mean(dim=(1, 2)).detach()
    floor = (0.1 * denom.mean()).clamp(min=eps)
    # 分子 Huber 化(δ = 3×batch RMS):|残差| ≤ δ 时与平方完全一致,超出部分
    # 线性——极端不可预测事件(GUI 突开/整屏剧变/爆炸)的梯度有界,不再偶发
    # 拽乱整个 batch。δ 随表征尺度自适应;指标仍用真平方比值(不可被 Huber 美化)。
    d = 3.0 * denom.mean().sqrt()
    q = r.abs()
    qc = torch.minimum(q, d)
    per_h = (qc * (2.0 * q - qc)).mean(dim=(1, 2))     # = r² (|r|≤δ) / δ(2|r|−δ) (超出)
    loss = (per_h / (denom + floor)).mean()
    per = r.detach().square().mean(dim=(1, 2))
    ratio = per / denom.clamp(min=eps)
    moved = denom > denom.median()
    pred_move = ratio[moved].mean() if bool(moved.any()) else ratio.mean()
    return loss, ratio.mean(), pred_move


def slot_diversity_loss(attn):
    """槽间多样性损失:软惩罚竞争注意力图的成对空间重叠。

    attn [*, N, M]:每行 = 一个 slot 在 M 个 patch 上的注意力分布(非负、沿 M 和≈1,
    SlotCompetitiveAttn 头平均后的 attn_map)。Gram G=attn·attnᵀ 的非对角元 ⟨w_i,w_j⟩
    度量 slot i,j 的空间重叠(分布非负 ⇒ ⟨·⟩=0 ⟺ 支撑不相交)。最小化非对角均值,
    逼不同 slot 落在不同 patch——修"多个 slot 盯同一最显著物体"的冗余。

    软版替代硬施密特正交化:前向 attn 保持合法分布,约束只进损失;且作用在
    **注意力空间(谁看哪里)**而非表征空间(后者与"实体语义相关性应可由内积表达"冲突,
    见 SlotCompetitiveAttn docstring)。
    """
    if attn is None or attn.shape[-2] < 2:
        return attn.new_zeros(()) if attn is not None else None
    N = attn.shape[-2]
    a = attn.float().flatten(0, -3) if attn.dim() > 3 else attn.float()        # [*,N,M]→[B,N,M]
    g = a @ a.transpose(-1, -2)                                                # [B,N,N] 成对重叠
    off = g.sum(dim=(-1, -2)) - torch.diagonal(g, dim1=-1, dim2=-2).sum(-1)    # 减对角(自重叠)
    return (off / (N * (N - 1))).mean()


def minecraft_inv_dyn_loss(delta_z, c, true_action, inv_dyn_head, move_w=4.0,
                           prev_action=None, kb_edge_w=1.0, patch_dz=None, ctx=None):
    """逆动力学:从 (z_tg(t+1) − z_obs(t)) ⊙ c 反推混合动作。

    ⚠ 槽路 / patch 旁路**分开监督**(2026-06-14,见 head.forward):两路 logits 不再相加。
    加法融合下损失只作用在「和」上,patch 旁路(直读冻结 patch-mean Δz、信号更干净)先把
    目标解释掉、残差→0,把槽路+c 的梯度掐断(gradient starvation,rollout 时 patch_dz=None
    只剩从没单独训过的槽路 ⇒ "啥也读不出")。改为各算各的 CE/BCE 再相加:槽路拿全量梯度
    (也是 c 的唯一梯度来源),patch 旁路参数互斥、纯天花板诊断,不回流稀释槽路。
    **读数/loss 主项取槽路**(= rollout 实际可用的诚实读出);patch 损失并入总 inv 但只训
    patch 头自己。注意:slot-only 损失值会高于旧的"槽+patch"合并值,inv 对编码器/c 的压力
    随之变大——可能要把 alpha_inv 适当下调再看 l_pred 是否被挤。

    鼠标 = mu-law 分箱加权 CE(非中心 ×move_w 堵基率不动点);键盘 = 20 键 BCE
    (跳变元素 ×kb_edge_w:整段按住的键早学会,按下/松开瞬间样本太稀)。
    delta_z 的 z_obs 端带梯度——本损失是编码器"让 Δ 编码动作"的唯一直接压力。
    """
    mouse_logits, kb_prob, parts = inv_dyn_head(delta_z * c, patch_dz=patch_dz, ctx=ctx)
    mouse_bin = camera_to_bin(true_action[:, :N_MOUSE])               # [B,2] long
    kb_t = true_action[:, N_MOUSE:]
    center = (CAMERA_BINS - 1) // 2
    prev_kb = prev_action[:, N_MOUSE:] if prev_action is not None else None

    def _mouse_ce(logits):
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             mouse_bin.reshape(-1), reduction="none")
        w = torch.where(mouse_bin.reshape(-1) == center,
                        torch.ones_like(ce), torch.full_like(ce, move_w))
        return (ce * w).sum() / w.sum()

    def _kb_bce(prob):
        bce = F.binary_cross_entropy(prob.clamp(EPS, 1 - EPS), kb_t, reduction="none")
        if prev_kb is not None and kb_edge_w > 1.0:
            w_kb = 1.0 + (kb_edge_w - 1.0) * (kb_t != prev_kb).to(bce.dtype)
            return (bce * w_kb).sum() / w_kb.sum()
        return bce.mean()

    l_mouse = _mouse_ce(mouse_logits)                  # 槽路(诚实主项)
    l_kb = _kb_bce(kb_prob)
    # patch 旁路:独立损失,参数与槽路互斥 + 不共享 logit 和 ⇒ 梯度不稀释槽路/c。
    # patch_dz 来自冻结骨干 ⇒ 这项只训 patch 头,不回流编码器。
    if parts is not None:
        l_patch = _mouse_ce(parts[0]) + _kb_bce(torch.sigmoid(parts[1]))
    else:
        l_patch = mouse_logits.new_zeros(())
    hit = (mouse_logits.argmax(-1) == mouse_bin)       # 读数取槽路
    moved = (mouse_bin != center)
    return (l_kb + l_mouse + l_patch, l_mouse.detach(), l_kb.detach(),
            hit.float().mean().detach(),
            (hit & moved).float().sum().detach(), moved.float().sum().detach())


def plan_bc_loss(plan, act_agg, dt, t, K, move_w=4.0):
    """未来动作规划的行为克隆:查询槽 k ↔ 未来第 k+1 个转移(时间序对齐;
    onset 经 cumsum-softplus 单调 ⇒ 不需要匈牙利匹配)。

    时间锚契约:0 时刻 = 本次 forward 的 t_vec("现在"),onset/duration 单位帧。
    槽 k 的目标(示范者实际做了什么):
      keyboard = act_agg[t+1+k] 键区(BCE);mouse = 同帧聚合鼠标的 mu-law 分箱
      (加权 CE,非中心 ×move_w——与 inv-dyn 同套路,堵基率不动点);
      onset = 该转移起点距"现在"的帧数(= dt[t] + 之前未来转移的累计跨度);
      duration = dt[t+1+k];exist = 槽是否落在序列界内(末端不足 K 个未来
      转移时,多余槽学习输出"无计划")。
    返回 (loss, onset_MAE[帧]);t 已无未来转移时返回 None。
    """
    B, T1 = dt.shape
    n = min(K, T1 - (t + 1))
    if n <= 0:
        return None
    agg = act_agg[:, t + 1:t + 1 + n].float()                    # [B,n,A]
    fdt = dt[:, t + 1:t + 1 + n].float()                         # [B,n]
    onset_tgt = dt[:, t:t + 1].float() + torch.cat(
        [torch.zeros_like(fdt[:, :1]), fdt.cumsum(dim=1)[:, :-1]], dim=1)

    l_kb = F.binary_cross_entropy(
        plan["keyboard"][:, :n].float().clamp(EPS, 1 - EPS), agg[:, :, N_MOUSE:])

    m_logits = plan["mouse_logits"][:, :n].float()               # [B,n,2,bins]
    m_bin = camera_to_bin(agg[:, :, :N_MOUSE])                   # [B,n,2]
    ce = F.cross_entropy(m_logits.reshape(-1, m_logits.shape[-1]),
                         m_bin.reshape(-1), reduction="none")
    center = (CAMERA_BINS - 1) // 2
    w = torch.where(m_bin.reshape(-1) == center,
                    torch.ones_like(ce), torch.full_like(ce, move_w))
    l_mouse = (ce * w).sum() / w.sum()

    onset = plan["onset"][:, :n].float()
    l_time = (F.smooth_l1_loss(onset, onset_tgt)
              + F.smooth_l1_loss(plan["duration"][:, :n].float(), fdt))

    exist_tgt = torch.zeros_like(plan["exist"], dtype=torch.float32)
    exist_tgt[:, :n] = 1.0
    l_exist = F.binary_cross_entropy(
        plan["exist"].float().clamp(EPS, 1 - EPS), exist_tgt)

    # 0.1:时间项以帧计(量级 ~几十),压到与 BCE/CE 同量级,防 timing 主导
    loss = l_kb + l_mouse + 0.1 * l_time + l_exist
    return loss, (onset - onset_tgt).abs().mean().detach()


def kl_diag_gauss(mu_q, lv_q, mu_p, lv_p):
    """对角高斯 KL(q‖p),逐样本求和过维度 → [B](nats)。fp32 调用方负责。"""
    return 0.5 * (lv_p - lv_q + ((lv_q.exp() + (mu_q - mu_p).square())
                                 / lv_p.exp()) - 1.0).sum(dim=-1)
