"""MinecraftWorldModel 自监督训练(Δz-JEPA 版):在离线 VPT 视频(画面+动作)上学世界动力学。

目标:把"看着 Minecraft 录像 + 录像里的真实动作"变成自监督信号,让世界模型学到
"动作在画面里有什么效果",为后续 prompt-tuning 迁移打底座。

  L_total = L_pred + ρ·L_open + α·L_inv + γ·L_plan + β·L_sigreg

  L_pred (Δz 预测,逐样本归一化 MSE): 从 (z_obs(t), 区间动作序列, dt, h, t) 预测
    潜表征**增量** Δz = sg[enc_ema(img_{t+dt}) − enc_ema(img_t)],逐样本除以
    E[Δz²](detach)。损失值直接可读:1.0 = persistence(预测 0)基线,<1 = 胜过复读。
    旧版预测绝对 z_{t+1} 时静态内容占目标能量 ~99.8%,动力学信号被淹没。
  L_inv  (逆动力学): 从 (z_tg(t+dt) − z_obs(t)) ⊙ c 反推区间聚合动作。
    鼠标 = mu-law 分箱分类(CE,堵死"恒 0"平凡解),键盘 = 20 键 BCE。
    z_obs 端带梯度 ⇒ 这是视觉编码器"必须让 Δ 编码动作"的唯一直接压力。
  L_open (开环 rollout): 同一转移再做一次前向,但感知输入换成**上一步自己的预测**
    ẑ = z + μ(记忆 h/历史/目标全同闭环)。400ep 基线的头号短板是"闭环胜过复读、
    开环劣于复读"——teacher forcing 从不训练"把自己的预测当立足点"。梯度穿过
    上一步的 μ,直接优化预测的**可推演性**;这是"脑内记忆指导行为"的数学前提。
  L_plan (未来动作 BC): 规划头查询槽 k ↔ 未来第 k+1 个转移的聚合动作+时长
    (onset 单调参数化 ⇒ 时间序对齐,免匈牙利匹配;0 时刻 = t_vec,单位帧)。
    学"示范者接下来会做什么"——脑内世界的前向动作输出通道。
  L_sigreg (防坍缩): sliced 高斯正则,施加在**在线感知编码 z_obs** 上(坍缩
    发生的位置;旧版施加在 Z_out 上,Transformer 可放大时间 PE/动作 token 的
    方差来满足检验,视觉内容照样坍缩)。

目标编码器 = 在线权重的 EMA 副本(每个优化步后 model.ema_update()),平稳靶。
可变 Δt(--frame_skip = 跨度上限,逐转移 Δt~U{1..skip}):消除固定步长的"默认
漂移先验",逼模型积分区间内动作序列(jumpy prediction,直攻开环复合误差);
模型同时接收区间内完整原始动作序列与聚合历史(见 VPTStreamDataset/模型 docstring)。
可视化与最终评估用 **holdout clip**(按文件名扣末 1 个,不进训练),展示泛化而非记忆。

数据:utils.vpt_dataset.VPTStreamDataset —— 流式加载(不全量预载),每个 worker 维护
<=cache_size 个已解码序列的滚动缓存,多 worker 并行喂数据,每步随机取 batch 个序列。
吞吐优化:截断 BPTT、AMP 混合精度、损失张量累积(避免逐步 .item() 同步)。

先准备数据(Colab 见 colab_demo.ipynb 的转换;本地合成见 download_sample_data.py),然后:
    python train/train_minecraft.py --data_dir runs/vpt_sample --epochs 50 --device cuda
"""
import argparse
import itertools
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from net.minecraft_world_model import MinecraftWorldModel
from utils.vpt_dataset import VPTStreamDataset
from utils.vpt_action import CAMERA_SCALE, CAMERA_BINS, camera_to_bin
from utils.minecraft_viz import visualize_minecraft
from utils.task_text import TaskTextEncoder
from blocks.primitives import SIGReg

EPS = 1e-4          # I1
ACT_DIM = 22        # 2 鼠标 + 20 键盘(见 VPT_KEYS)
N_MOUSE = 2


# =====================================================================
# 损失函数
# =====================================================================

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


def minecraft_inv_dyn_loss(delta_z, c, true_action, inv_dyn_head, move_w=4.0,
                           prev_action=None, kb_edge_w=1.0):
    """逆动力学:从 (z_tg(t+1) − z_obs(t)) ⊙ c 反推混合动作。

    鼠标 = mu-law 分箱**加权** CE:中心 bin(没动鼠标)在区间聚合后占 ~2/3,
    无权重时基率解(恒猜中心)的 CE 已接近最优——实测 100 epoch 模型先学会抓
    极端转身、又退化回恒中心(基率解损失更低,把它推了回去)。非中心目标
    ×move_w 后,基率解不再是不动点,运动帧才有有效梯度。键盘 = 20 键 BCE。
    delta_z 的 z_obs 端带梯度:本损失是编码器"让 Δ 编码动作"的唯一直接压力,
    也是可控闸 c 的唯一梯度来源(动作可解释 → c↑;噪声槽稀释池化特征 → c↓)。
    返回的 move_hit/move_n 用于训练侧 mouse_move_acc(全帧 acc 会被中心基率
    灌到 0.66 一动不动,运动帧 acc 才是诚实读数)。
    """
    mouse_logits, kb_prob = inv_dyn_head(delta_z * c)
    mouse_bin = camera_to_bin(true_action[:, :N_MOUSE])               # [B,2] long
    ce = F.cross_entropy(mouse_logits.reshape(-1, mouse_logits.shape[-1]),
                         mouse_bin.reshape(-1), reduction="none")
    center = (CAMERA_BINS - 1) // 2
    w = torch.where(mouse_bin.reshape(-1) == center,
                    torch.ones_like(ce), torch.full_like(ce, move_w))
    l_mouse = (ce * w).sum() / w.sum()
    # 键盘跳变加权:400ep 基线 kb_onset_recall 钉死 ~0.17——整段按住的键(占帧
    # 绝大多数)早学会了,按下/松开瞬间的样本太稀(对平均 BCE 无感)。与上一区间
    # 聚合键比对,发生跳变的 (样本,键) 元素 ×kb_edge_w,与 mouse_move_w 同思路。
    kb_t = true_action[:, N_MOUSE:]
    bce = F.binary_cross_entropy(kb_prob.clamp(EPS, 1 - EPS), kb_t, reduction="none")
    if prev_action is not None and kb_edge_w > 1.0:
        edges = (kb_t != prev_action[:, N_MOUSE:]).to(bce.dtype)
        w_kb = 1.0 + (kb_edge_w - 1.0) * edges
        l_kb = (bce * w_kb).sum() / w_kb.sum()
    else:
        l_kb = bce.mean()
    hit = (mouse_logits.argmax(-1) == mouse_bin)
    moved = (mouse_bin != center)
    mouse_acc = hit.float().mean()
    return (l_kb + l_mouse, l_mouse.detach(), l_kb.detach(), mouse_acc.detach(),
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


def _to_float_img(img):
    """uint8 [.,3,H,W] → float∈[0,1](归一化推迟到 GPU 上做,PCIe 传 uint8 省 4×)。"""
    return img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()


# =====================================================================
# GPU 利用率采样(优先 NVML,兜底 nvidia-smi)
# =====================================================================

_GPU_UTIL_FN = None


def _resolve_gpu_util():
    try:
        torch.cuda.utilization()                       # 需 nvidia-ml-py
        return torch.cuda.utilization
    except Exception:
        pass
    import shutil, subprocess
    smi = shutil.which("nvidia-smi")
    if smi:
        def _via_smi():
            try:
                out = subprocess.run(
                    [smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                return float(out.stdout.strip().splitlines()[0])
            except Exception:
                return None
        return _via_smi
    return lambda: None


def _gpu_util():
    """瞬时 GPU 利用率(%);CPU 或无法采样时返回 None。"""
    global _GPU_UTIL_FN
    if _GPU_UTIL_FN is None:
        _GPU_UTIL_FN = _resolve_gpu_util()
    return _GPU_UTIL_FN()


# =====================================================================
# 训练 / 评估
# =====================================================================

def roll_hist(a_hist, t_hist, hv, action, dt_cur):
    """时间前进 dt_cur 帧并滚入一个刚结束的聚合动作。

    a_hist [B,J,A] / t_hist [B,J](各条目结束时刻距"现在"的帧数)/ hv [B,J] 有效位。
    旧条目统一变老 dt_cur 帧;新条目刚结束 ⇒ age=0、有效位=1。开头的空槽
    (a_hist 初始全零)有效位为 0——全零动作是合法的"没按",不能用值判空。
    """
    B = action.shape[0]
    z1 = torch.zeros(B, 1, device=action.device, dtype=t_hist.dtype)
    return (torch.cat([a_hist[:, 1:], action.unsqueeze(1)], dim=1),
            torch.cat([t_hist[:, 1:] + dt_cur.unsqueeze(1), z1], dim=1),
            torch.cat([hv[:, 1:], torch.ones_like(z1)], dim=1))


def _run_sequence(model, sigreg, batch_dev, device, k_bptt,
                  alpha_inv, beta_sigreg, move_w, gamma_plan, rho_open, open_every,
                  kb_edge_w, amp_dev, use_amp, scaler, acc):
    """对一个 batch 的完整序列做截断 BPTT 前向+反向;损失以张量形式累加进 acc。

    时序 = teacher forcing:每步感知输入 z_obs(t) 都来自真实画面(批量编码),
    跨步记忆只走 h token(窗口内带梯度,窗口边界 detach)。开环推演(ẑ+μ 累积)
    只在可视化/推理里做;可变 Δt 的"积分区间动作"目标已覆盖多步效应。
    历史 a_hist 在 forward **之后**滚入当前聚合动作:历史=严格过去,
    当前区间的动作走 a_cur 完整序列通道,不重复注入。

    吞吐:目标编码整条序列一次批量(EMA + no_grad);在线编码按 BPTT 窗口批量;
    只有 Transformer 递归本身逐步(固有串行)。
    """
    img, t_vec = batch_dev["img"], batch_dev["t_vec"]
    act_seq, act_agg, dt = batch_dev["act_seq"], batch_dev["act_agg"], batch_dev["dt"]
    task_emb = batch_dev.get("task_emb")
    B, T = img.shape[0], img.shape[1]
    h = torch.zeros(B, 1, model.d, device=device)
    a_hist = torch.zeros(B, model.J, ACT_DIM, device=device)
    t_hist = torch.zeros(B, model.J, device=device)    # 各历史条目距"现在"的帧数
    hv = torch.zeros(B, model.J, device=device)        # 历史有效位(空槽=0)
    n_win = -(-(T - 1) // k_bptt)        # ceil:窗口数,用于 SIGReg 权重尺度归一
    last_c = None

    # 冻结骨干特征:整条序列一次批量提取(no_grad),在线/目标两路共用
    with torch.autocast(device_type=amp_dev, enabled=use_amp):
        feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))   # [B*T, M, Ed]
        # JEPA 目标:EMA 投影+binder(平稳靶,no_grad)
        z_tg = model.encode_target(feats=feats).view(B, T, model.N, model.d)
    feats = feats.view(B, T, *feats.shape[-2:])
    z_tg = z_tg.float()

    for w0 in range(0, T - 1, k_bptt):
        w1 = min(w0 + k_bptt, T - 1)
        k = w1 - w0
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            # 在线感知:窗口 B·k 帧(梯度只进 proj+binder,骨干特征已缓存)
            z_obs = model.encode_obs(
                feats=feats[:, w0:w1].reshape(B * k, *feats.shape[-2:])
            ).view(B, k, model.N, model.d)

        accum = torch.zeros((), device=device)
        zhat = None     # 上一步预测的状态 ẑ = z + μ(窗口内携带梯度;窗口边界重置)
        for i in range(k):
            t = w0 + i
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                out = model(z_obs[:, i], h, a_hist, act_seq[:, t], dt[:, t], t_vec[:, t],
                            t_hist=t_hist, hist_valid=hv, task_emb=task_emb)
                out_open = None
                if (rho_open > 0 and zhat is not None
                        and i % open_every == open_every - 1):
                    # 开环支路:感知输入换成上一步预测 ẑ,记忆/历史/动作/目标全同
                    # 闭环——梯度穿过上一步 μ,训练"自己的预测可作下一步立足点"
                    out_open = model(zhat, h, a_hist, act_seq[:, t], dt[:, t],
                                     t_vec[:, t], t_hist=t_hist, hist_valid=hv,
                                     task_emb=task_emb)
            prev_z = z_obs[:, i]
            prev_act = act_agg[:, t - 1] if t > 0 else act_agg[:, t]   # t=0 无跳变
            a_hist, t_hist, hv = roll_hist(a_hist, t_hist, hv, act_agg[:, t], dt[:, t])

            # 损失在 fp32 计算(autocast 外:BCE 不可 autocast,CE/归一化 MSE 在 fp32 更稳)
            mu, c = out["mu"].float(), out["c"].float()
            dz = z_tg[:, t + 1] - z_tg[:, t]                    # Δz 目标(no_grad)
            l_pred, r_pred, r_pred_mv = dz_pred_loss(mu, dz)
            l_inv, l_mouse, l_kb, m_acc, mv_hit, mv_n = minecraft_inv_dyn_loss(
                z_tg[:, t + 1] - z_obs[:, i].float(), c, act_agg[:, t],
                model.inv_dyn, move_w, prev_action=prev_act, kb_edge_w=kb_edge_w)
            step_loss = l_pred + alpha_inv * l_inv
            if out_open is not None:
                l_open, _, r_open_mv = dz_pred_loss(out_open["mu"].float(), dz)
                step_loss = step_loss + rho_open * l_open
                acc["pred_open"] += r_open_mv; acc["open_n"] += 1
            pl = plan_bc_loss(out["action_plan"], act_agg, dt, t, model.K, move_w) \
                if gamma_plan > 0 else None
            if pl is not None:
                step_loss = step_loss + gamma_plan * pl[0]
                acc["plan"] += pl[0].detach(); acc["onset_mae"] += pl[1]
                acc["plan_n"] += 1

            accum = accum + step_loss / (T - 1)
            last_c = c
            zhat = prev_z + out["mu"]    # ẑ(t+1) 估计,供下一步开环支路(带梯度)
            h = out["h_next"]            # 窗口内保留梯度(截断 BPTT,跨步记忆只走 h)
            # 张量累加(不 .item(),epoch 末一次性取出 → 去掉逐步 GPU↔CPU 同步)
            acc["pred"] += r_pred; acc["pred_move"] += r_pred_mv
            acc["inv"] += l_inv.detach()
            acc["mouse"] += l_mouse; acc["kb"] += l_kb; acc["mouse_acc"] += m_acc
            acc["mv_hit"] += mv_hit; acc["mv_n"] += mv_n
            acc["pred_rms"] += (mu - dz).square().mean().sqrt().detach()
            acc["dz_rms"] += dz.square().mean().sqrt()
            acc["inner"] += 1

        # SIGReg:施加在**在线感知编码 z_obs** 上——防坍缩要钉在坍缩发生的张量上。
        # slot 为分组、(窗口时间 × batch) 为样本维(近似平稳假设,把样本数提到 k·B,
        # 小 batch 下 ECF 检验才有功效)。除以 n_win:总权重 = β·mean_w(sig),
        # 不随 rollout 长度/k_bptt 改变与 pred 项的相对比例。
        l_sig = sigreg(z_obs.float().permute(2, 1, 0, 3).reshape(model.N, k * B, model.d))
        win_loss = accum + (beta_sigreg / n_win) * l_sig
        scaler.scale(win_loss).backward()
        acc["loss"] += win_loss.detach()
        acc["sigreg"] += l_sig.detach(); acc["win"] += 1
        h = h.detach()                                           # 窗口边界截断
    return last_c


def train_epoch(model, sigreg, data_iter, opt, scaler, device, steps, k_bptt,
                alpha_inv, beta_sigreg, move_w, gamma_plan, rho_open, open_every,
                kb_edge_w, text_enc, amp_dev, use_amp):
    """data_iter:**全程唯一**的 DataLoader 迭代器(islice 消费,不重建)。
    每 epoch 重建迭代器会丢弃预取队列里的在途 batch 并重置 worker 状态——
    无限流数据集下表现为 GPU 功率按 epoch 周期(~25s)锯齿振荡。"""
    model.train()
    acc = {k: torch.zeros((), device=device) for k in
           ["loss", "pred", "pred_move", "pred_open", "pred_rms", "dz_rms", "inv",
            "mouse", "mouse_acc", "kb", "sigreg", "mv_hit", "mv_n", "plan", "onset_mae"]}
    acc["inner"] = acc["win"] = acc["plan_n"] = acc["open_n"] = 0
    gpu_samples, c_last, n_batches = [], None, 0

    for batch in itertools.islice(data_iter, steps):
        batch_dev = {
            "img": _to_float_img(batch["img"].to(device, non_blocking=True)),
            "act_seq": batch["act_seq"].to(device, non_blocking=True),
            "act_agg": batch["act_agg"].to(device, non_blocking=True),
            "dt": batch["dt"].to(device, non_blocking=True),
            "t_vec": batch["t_vec"].to(device, non_blocking=True),
            # 任务文本 → 冻结句向量(编码器内有唯一串缓存,实为查表)
            "task_emb": (text_enc.encode(batch["task_text"]).to(device)
                         if text_enc is not None else None),
        }

        opt.zero_grad(set_to_none=True)
        c_last = _run_sequence(model, sigreg, batch_dev, device, k_bptt,
                               alpha_inv, beta_sigreg, move_w, gamma_plan,
                               rho_open, open_every, kb_edge_w,
                               amp_dev, use_amp, scaler, acc)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        model.ema_update()                  # 目标编码器跟踪在线权重(每优化步一次)
        n_batches += 1
        u = _gpu_util()
        if u is not None:
            gpu_samples.append(u)

    ic = max(acc["inner"], 1); wc = max(acc["win"], 1); pc = max(acc["plan_n"], 1)
    cv = c_last.squeeze(-1).flatten()
    return {
        "plan": (acc["plan"] / pc).item(),
        "plan_onset_mae": (acc["onset_mae"] / pc).item(),   # 帧
        "loss": (acc["loss"] / wc).item(),
        "pred": (acc["pred"] / ic).item(),          # 1.0 = persistence 基线(含静止样本托底)
        "pred_move": (acc["pred_move"] / ic).item(),  # 仅真运动样本——诚实读数
        "pred_open": (acc["pred_open"] / max(acc["open_n"], 1)).item(),  # 开环支路(运动样本)
        "pred_rms": (acc["pred_rms"] / ic).item(),  # 真实预测误差(不可被 σ/归一化美化)
        "dz_rms": (acc["dz_rms"] / ic).item(),      # 目标本身的变化幅度(基线尺度)
        "inv": (acc["inv"] / ic).item(),
        "mouse": (acc["mouse"] / ic).item(), "mouse_acc": (acc["mouse_acc"] / ic).item(),
        "mouse_move_acc": (acc["mv_hit"] / acc["mv_n"].clamp(min=1)).item(),
        "kb": (acc["kb"] / ic).item(),
        "sigreg": (acc["sigreg"] / wc).item(),
        "c_mean": cv.mean().item(), "c_std": cv.std().item(),
        "batches": n_batches,
        "gpu_util": (sum(gpu_samples) / len(gpu_samples)) if gpu_samples else None,
    }


@torch.no_grad()
def evaluate(model, loader, device, steps, amp_dev, use_amp):
    """逆动力学评估:能否从潜变化里反推出"做了什么操作"。

    键盘:平衡准确率 + **跳变**(onset/release recall)——VPT 数据里 w/attack 等键
    常整段按住,逐帧 balanced acc 会被"输出基率常数"的平凡解灌高;只有按下/松开
    瞬间的检出率才证明模型从 ΔZ 里读出了动作。
    鼠标:分箱准确率,分全帧 acc 与 **move_acc**(仅 GT 非中心 bin、即真动了
    鼠标的帧)——基率解(恒中心 bin)在 move_acc 上得 0,同 onset 逻辑。
    loader 应来自 **holdout split**(不进训练的 clip),量泛化而非记忆。
    """
    model.eval()
    tp = fp = fn = tn = 0
    on_tp = on_n = off_tp = off_n = 0
    m_hit = m_n = mv_hit = mv_n = 0
    pred_sum, pred_mv_sum, pred_n = 0.0, 0.0, 0
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
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
            z_obs = model.encode_obs(
                feats=feats.view(B, T, *feats.shape[-2:])[:, :T - 1]
                .reshape(B * (T - 1), *feats.shape[-2:])
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
            pred_n += 1
            mouse_logits, kb_prob = model.inv_dyn(
                (z_tg[:, t + 1] - z_obs[:, t]) * out["c"].float())
            kb_pred = (kb_prob > 0.5)
            kb_true = (act_agg[:, t, N_MOUSE:] > 0.5)
            tp += (kb_pred & kb_true).sum().item();  fp += (kb_pred & ~kb_true).sum().item()
            fn += (~kb_pred & kb_true).sum().item();  tn += (~kb_pred & ~kb_true).sum().item()
            if prev_true is not None:
                onset = kb_true & ~prev_true; release = ~kb_true & prev_true
                on_tp += (kb_pred & onset).sum().item();    on_n += onset.sum().item()
                off_tp += (~kb_pred & release).sum().item(); off_n += release.sum().item()
            prev_true = kb_true
            mb_pred = mouse_logits.argmax(-1)
            mb_true = camera_to_bin(act_agg[:, t, :N_MOUSE])
            hit = (mb_pred == mb_true)
            m_hit += hit.sum().item(); m_n += hit.numel()
            moved = (mb_true != center)
            mv_hit += (hit & moved).sum().item(); mv_n += moved.sum().item()
            h = out["h_next"]
    recall = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"pred": pred_sum / max(pred_n, 1),
            "pred_move": pred_mv_sum / max(pred_n, 1),
            "kb_recall": recall, "kb_spec": spec, "kb_bal_acc": 0.5 * (recall + spec),
            "kb_onset_recall": on_tp / max(on_n, 1),
            "kb_release_recall": off_tp / max(off_n, 1),
            "kb_edges": on_n + off_n,
            "mouse_bin_acc": m_hit / max(m_n, 1),
            "mouse_move_acc": mv_hit / max(mv_n, 1),
            "mouse_moves": mv_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample", help="训练数据目录(.mp4+.jsonl)")
    ap.add_argument("--holdout_dir", default=None,
                    help="独立的固定 holdout 目录(eval/viz 专用)。设置后进入**滚动目录"
                         "模式**:data_dir 内容可随时增删(后台滚动下载器),worker 每次"
                         "换 clip 重扫目录随机抽取;不设则按文件名从 data_dir 扣末"
                         " holdout_n 个(静态目录模式)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=50, help="每 epoch 迭代多少个 batch(流式)")
    ap.add_argument("--batch", type=int, default=16,
                    help="每步随机取几个序列(batch=8 时实测 L4 显存仅用 ~0.9GB/24GB、"
                         "利用率 ~16%%——显存余量极大,默认提到 16;SIGReg 检验功效同步受益)")
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame_skip", type=int, default=8,
                    help="可变预测跨度上限:每个转移 Δt~U{1..skip}(帧),区间内完整动作"
                         "序列喂给模型(jumpy prediction,逼模型积分动作、直攻开环复合误差)")
    ap.add_argument("--img_size", type=int, default=128,
                    help="训练分辨率(数据集端 INTER_AREA 下采样;0=保持原始分辨率)")
    ap.add_argument("--camera_scale", type=float, default=None,
                    help="鼠标 dx/dy 归一化尺度,固定超参(None=用 vpt_action.CAMERA_SCALE 默认 10;"
                         "真 BASALT 转身 ±190 用 ~20)")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader 并行加载进程数;默认 None=自动取 CPU 核数-1(上限 8)。"
                         "窗口解码是 CPU 大头,旧默认 2 是 GPU 利用率 ~16%% 的主因")
    ap.add_argument("--cache_size", type=int, default=32, help="每个 worker 缓存的动作表文件数")
    ap.add_argument("--refresh_every", type=int, default=64, help="已废弃(窗口化解码无整段缓存),仅保留兼容")
    ap.add_argument("--clip_cache", type=int, default=4,
                    help="每个 worker 常驻内存的整段 clip 数(顺序解码到 img_size 分辨率"
                         "uint8,128px 下 ≈0.3GB/段;窗口=纯内存切片,吞吐与解码解耦)")
    ap.add_argument("--clip_refresh", type=int, default=256,
                    help="每产出多少个窗口滚动换入一段新 clip(一段解码 ~20s,256 时"
                         "摊到每窗口 <0.1s;调小=数据更新更快但解码停顿更频)")
    ap.add_argument("--buffer_size", type=int, default=0,
                    help="窗口级滚动缓存(兼容遗留;clip 缓存落地后窗口切片已免费,"
                         "正常保持 0 关闭)")
    ap.add_argument("--buffer_reuse", type=int, default=1,
                    help="窗口级缓存的复用倍数(兼容遗留,正常保持 1)")
    ap.add_argument("--holdout_n", type=int, default=1,
                    help="按文件名扣末几个 clip 做 holdout(eval/viz 专用,不进训练)")
    ap.add_argument("--log_every", type=int, default=5,
                    help="每多少 epoch 打印一行训练指标(batch 大时每步覆盖样本多,应调小)")
    ap.add_argument("--viz_every", type=int, default=10,
                    help="每多少 epoch 输出一次可视化面板(0=关闭;首次在第 viz_every 个"
                         "epoch 之后——训练没开始不出图)")
    ap.add_argument("--eval_every", type=int, default=5,
                    help="每多少 epoch 在 holdout 上跑一次轻量评估并入 wandb 曲线"
                         "(0=只在训练结束跑;首次在第 eval_every 个 epoch 之后——"
                         "训练没开始不评估)。泛化差距 = eval/pred 与训练 pred 之差")
    ap.add_argument("--viz_dir", default="runs/mc_viz")
    ap.add_argument("--encoder", choices=["dinov3", "dinov2", "mock"], default="dinov2",
                    help="视觉骨干(冻结):dinov3=ViT-S/16(默认;稠密特征最强,patch=16 "
                         "整除 128;hubconf 依赖 torchmetrics,Colab 端 pip 安装即可;"
                         "权重若 gated 经 --encoder_weights 传入);dinov2=ViT-S/14(权重"
                         "完全开放,降级备选,img_size 用 126);mock=随机冻结卷积(离线冒烟)")
    ap.add_argument("--encoder_weights", default=None,
                    help="dinov3 权重 URL 或本地 .pth 路径(权重 gated 时在 Meta/HF 接受许可证后获得)")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--N", type=int, default=16, help="实体槽数")
    ap.add_argument("--K", type=int, default=5, help="动作查询数")
    ap.add_argument("--J", type=int, default=8, help="历史动作长度")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha_inv", type=float, default=1.0, help="逆动力学损失权重")
    ap.add_argument("--mouse_move_w", type=float, default=4.0,
                    help="鼠标 CE 中非中心 bin 目标的权重(中心 bin 占 ~2/3,不加权时"
                         "基率解近似最优,模型会退化成恒猜中心;1.0=不加权)")
    ap.add_argument("--gamma_plan", type=float, default=0.5,
                    help="未来动作规划 BC 损失权重(查询槽 k ↔ 未来第 k+1 个转移的"
                         "聚合动作+时长;0=关闭,规划头退回无监督)")
    ap.add_argument("--rho_open", type=float, default=0.5,
                    help="开环 rollout 损失权重:同一转移再前向一次,感知输入换成上一步"
                         "自己的预测 ẑ=z+μ(0=关闭)。直攻'闭环胜过复读、开环劣于复读'"
                         "的差距——脑内推演可用性的训练信号")
    ap.add_argument("--open_every", type=int, default=1,
                    help="每几个内步做一次开环支路(成本旋钮:1=每步,实测总墙钟 ~+20%%;"
                         "2=减半开环前向,~+10%%;k_bptt=4 时取 3 则每窗口仅 1 次)")
    ap.add_argument("--kb_edge_w", type=float, default=4.0,
                    help="键盘 BCE 中发生跳变(与上一区间不同)的 (样本,键) 元素权重"
                         "(onset/release 样本稀疏,平均 BCE 对其无感,kb_onset_recall"
                         "曾 400ep 钉死 0.17;1.0=不加权)")
    ap.add_argument("--no_cosine", action="store_true",
                    help="关闭余弦 lr 衰减(默认开启:lr 余弦退火到 0.1×,按 epoch 步进;"
                         "恒定 lr 的噪声地板是 pred_move 尾段平台的嫌疑人之一)")
    ap.add_argument("--text_encoder", choices=["minilm", "mock", "none"], default="minilm",
                    help="任务文本条件:minilm=冻结句向量(需 transformers,语义空间"
                         "可外推);mock=哈希伪嵌入(可区分任务无语义,离线冒烟);"
                         "none=不条件化(text token 用可学常数)。单任务数据下文本是"
                         "常数,条件只在多任务混采时携带信息")
    ap.add_argument("--beta_sigreg", type=float, default=0.1, help="SIGReg 防坍缩权重(施加在 z_obs 上)")
    ap.add_argument("--ema_decay", type=float, default=0.99,
                    help="目标编码器 EMA 衰减(时间常数 ≈ 1/(1−τ) 个优化步;短跑程用 0.99,"
                         "长跑程可升 0.996+)")
    ap.add_argument("--k_bptt", type=int, default=4, help="截断 BPTT 窗口")
    ap.add_argument("--no_amp", action="store_true", help="关闭 AMP 混合精度(默认 cuda 上开启)")
    ap.add_argument("--wandb", action="store_true", help="开启 wandb 远程记录(key 从环境变量 WANDB_API_KEY 读)")
    ap.add_argument("--wandb_project", default="minecraft-world-model")
    ap.add_argument("--wandb_run", default=None, help="wandb run 名(默认自动生成)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_cuda = str(dev).startswith("cuda")
    amp_dev = "cuda" if is_cuda else "cpu"
    use_amp = is_cuda and not args.no_amp
    print(f"=== MINECRAFT WORLD MODEL (Δz-JEPA + InvDyn + SIGReg) | device={dev} | amp={use_amp} ===")

    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        except Exception as ex:
            print(f"[wandb] 初始化失败,关闭远程记录: {ex}")
            use_wandb = False

    if args.holdout_dir:
        # 滚动目录模式:train 目录允许暂空(后台下载器在填,worker 会等),
        # 但 holdout 必须就绪——固定 eval 集/可视化都依赖它。
        if not os.path.isdir(args.holdout_dir) or not any(
                f.endswith(".mp4") for f in os.listdir(args.holdout_dir)):
            print(f"[!] holdout 目录 '{args.holdout_dir}' 里没有 .mp4。先跑 colab §2(固定 holdout 下载)。")
            sys.exit(1)
    elif not os.path.isdir(args.data_dir) or not any(
            f.endswith(".mp4") for f in os.listdir(args.data_dir)):
        print(f"[!] 数据目录 '{args.data_dir}' 里没有 .mp4。先准备数据(download_sample_data.py / colab 转换)。")
        sys.exit(1)

    # 任务文本编码器(冻结,CPU 上跑——任务文本基数极小,编码即查表)
    text_enc = None if args.text_encoder == "none" else \
        TaskTextEncoder(args.text_encoder, device="cpu")

    img_size = args.img_size if args.img_size > 0 else None
    cam_scale = args.camera_scale if args.camera_scale is not None else CAMERA_SCALE
    n_workers = args.workers if args.workers is not None \
        else max(2, min(8, (os.cpu_count() or 2) - 1))
    # 双目录(滚动)模式:train 池整目录用且随时重扫,holdout 是独立固定目录;
    # 单目录(静态)模式:按文件名扣末 holdout_n 个,与旧行为一致。
    train_split = None if args.holdout_dir else "train"
    hold_dir = args.holdout_dir or args.data_dir
    hold_split = None if args.holdout_dir else "holdout"
    os.makedirs(args.data_dir, exist_ok=True)      # 滚动模式下可先于下载器启动
    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                          cache_size=args.cache_size, refresh_every=args.refresh_every,
                          seed=args.seed, img_size=img_size, camera_scale=cam_scale,
                          frame_skip=args.frame_skip, split=train_split,
                          holdout_n=args.holdout_n, buffer_size=args.buffer_size,
                          buffer_reuse=args.buffer_reuse,
                          clip_cache=args.clip_cache, clip_refresh=args.clip_refresh)
    # prefetch_factor=2:解码抖动已由滚动缓存+reuse 吸收;大 batch 下每个预取
    # batch 是 ~0.4GB 的大块(256×30 帧 uint8),更深的队列只是白占主存、
    # 拉长积压(worker 数 × 深度 × batch 个窗口的解码欠账)。
    loader = DataLoader(ds, batch_size=args.batch, num_workers=n_workers,
                        pin_memory=is_cuda,
                        persistent_workers=(n_workers > 0),
                        prefetch_factor=(2 if n_workers > 0 else None))
    # 全程唯一迭代器:每 epoch 重建会丢在途预取 batch、重置 worker 迭代状态,
    # GPU 功率按 epoch 周期锯齿(无限流数据集没有"epoch 末尾",不需要重建)
    data_iter = iter(loader)
    # 可视化与评估都用 holdout clip(独立固定目录或按文件名扣末 holdout_n 个,
    # 不进训练):在训练数据上展示/评估,量到的是记忆不是泛化。
    eval_ds = VPTStreamDataset(hold_dir, seq_len=args.seq_len, fps=args.fps,
                               cache_size=4, seed=args.seed + 555, img_size=img_size,
                               camera_scale=cam_scale, frame_skip=args.frame_skip,
                               split=hold_split, holdout_n=args.holdout_n)

    # 固定 eval 集:首次评估时从 holdout 一次性采集 4×eval_bs 条序列,之后每次评估
    # 复用同一份数据——指标跨 epoch 严格可比,且评估期零解码、不与训练抢 CPU。
    # 惰性采集(而非启动时):训练先跑起来,主流程不被评估数据准备耽误。
    eval_bs = min(args.batch, 64)
    eval_batches = []

    def _get_eval_batches():
        if not eval_batches:
            import time
            t0 = time.time()
            it = iter(DataLoader(eval_ds, batch_size=eval_bs,
                                 num_workers=min(4, n_workers)))
            for _ in range(4):
                b = next(it)
                if text_enc is not None:    # 固定集随采随编码任务文本(CPU 驻留)
                    b["task_emb"] = text_enc.encode(b["task_text"])
                eval_batches.append(b)
            del it                       # 立刻放掉 eval worker,不留后台解码
            src = args.holdout_dir if args.holdout_dir else f"扣末 {args.holdout_n} clip"
            print(f"  [eval] 固定评估集已采集:4×{eval_bs} 序列 "
                  f"(holdout={src},{time.time() - t0:.0f}s,此后复用)")
        return eval_batches

    viz_batch = None
    if args.viz_every > 0:
        # 固定一条可视化序列(独立 seed,跨 epoch 同一窗口 → 面板可前后对比)
        viz_ds = VPTStreamDataset(hold_dir, seq_len=args.seq_len, fps=args.fps,
                                  cache_size=4, seed=args.seed + 999, img_size=img_size,
                                  camera_scale=cam_scale, frame_skip=args.frame_skip,
                                  split=hold_split, holdout_n=args.holdout_n)
        viz_batch = next(iter(DataLoader(viz_ds, batch_size=1)))
        os.makedirs(args.viz_dir, exist_ok=True)

    model = MinecraftWorldModel(d=args.d, N=args.N, K=args.K, J=args.J, act_dim=ACT_DIM,
                                n_cam_bins=CAMERA_BINS, ema_decay=args.ema_decay,
                                max_skip=args.frame_skip, encoder=args.encoder,
                                encoder_weights=args.encoder_weights).to(dev)
    sigreg = SIGReg(knots=17, num_proj=512).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = None if args.no_cosine else torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"trainable params: {n_train / 1e6:.1f}M | frozen {args.encoder} backbone: "
          f"{n_frozen / 1e6:.1f}M | batch={args.batch} workers={n_workers} "
          f"steps/epoch={args.steps_per_epoch} frame_skip={args.frame_skip}")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    util_hist = []

    for ep in range(args.epochs):
        r = train_epoch(model, sigreg, data_iter, opt, scaler, dev, args.steps_per_epoch,
                        args.k_bptt, args.alpha_inv, args.beta_sigreg,
                        args.mouse_move_w, args.gamma_plan, args.rho_open,
                        args.open_every, args.kb_edge_w, text_enc, amp_dev, use_amp)
        if sched is not None:
            sched.step()
        if r.get("gpu_util") is not None:
            util_hist.append(r["gpu_util"])
        if use_wandb:
            _wb = {k: r[k] for k in ("loss", "pred", "pred_move", "pred_open",
                                     "pred_rms", "dz_rms",
                                     "inv", "mouse", "mouse_acc", "mouse_move_acc", "kb",
                                     "plan", "plan_onset_mae",
                                     "sigreg", "c_mean", "c_std")}
            if r.get("gpu_util") is not None:
                _wb["gpu_util"] = r["gpu_util"]
            wandb.log(_wb, step=ep)
        if ep % args.log_every == 0 or ep == args.epochs - 1:
            gpu = f" | gpu {r['gpu_util']:.0f}%" if r.get("gpu_util") is not None else ""
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | pred {r['pred']:.3f} "
                  f"mv {r['pred_move']:.3f} open {r['pred_open']:.3f}×copy "
                  f"(rms {r['pred_rms']:.4f}/dz {r['dz_rms']:.4f}) | "
                  f"inv {r['inv']:.3f} (kb {r['kb']:.3f}/mouse {r['mouse']:.2f} "
                  f"acc {r['mouse_acc']:.2f} mv {r['mouse_move_acc']:.2f}) | "
                  f"plan {r['plan']:.2f} (onset±{r['plan_onset_mae']:.1f}f) | "
                  f"sig {r['sigreg']:.2f} | "
                  f"c mean={r['c_mean']:.3f} std={r['c_std']:.3f}{gpu}")
        # 节奏用 (ep+1):第 viz_every/eval_every 个 epoch 训练完才首次出图/评估,
        # 不在训练刚起步时就花算力展示一个随机初始化的模型
        if viz_batch is not None and ((ep + 1) % args.viz_every == 0 or ep == args.epochs - 1):
            viz_emb = (text_enc.encode(viz_batch["task_text"]) if text_enc is not None
                       else None)
            p = visualize_minecraft(model, viz_batch, dev,
                                    os.path.join(args.viz_dir, f"ep{ep:04d}.png"),
                                    task_emb=viz_emb)
            if p:
                print(f"  [viz] {p}")
                if use_wandb:
                    wandb.log({"viz/panel": wandb.Image(p)}, step=ep)
        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
            # 轻量 holdout 评估(固定 4 个 batch,首次采集后复用):
            # eval/pred 对照训练 pred = 泛化差距曲线
            ev = evaluate(model, _get_eval_batches(), dev, 4, amp_dev, use_amp)
            print(f"  [eval] pred {ev['pred']:.3f} mv {ev['pred_move']:.3f}×copy | "
                  f"kb bal {ev['kb_bal_acc']:.3f} "
                  f"onset {ev['kb_onset_recall']:.3f} | mouse move {ev['mouse_move_acc']:.3f}")
            if use_wandb:
                wandb.log({f"eval/{k}": v for k, v in ev.items()}, step=ep)

    if util_hist:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[GPU] 训练平均利用率 {sum(util_hist) / len(util_hist):.0f}% "
              f"({len(util_hist)} epoch) | 峰值显存 {peak:.2f} GB")

    print("\n--- 最终评估:逆动力学(从画面变化反推操作;holdout clip,未进训练)---")
    e = evaluate(model, _get_eval_batches(), dev, 4, amp_dev, use_amp)
    print(f"Δz 预测 {e['pred']:.3f}×copy(holdout;<1 = 泛化意义上胜过复读)")
    print(f"键盘 recall {e['kb_recall']:.3f} | spec {e['kb_spec']:.3f} | "
          f"平衡准确率 {e['kb_bal_acc']:.3f}(随机基线≈0.5)")
    print(f"跳变检出 onset {e['kb_onset_recall']:.3f} | release {e['kb_release_recall']:.3f} "
          f"({e['kb_edges']} 次跳变;常按键灌不高这两项,是更硬的证据)")
    print(f"鼠标 bin acc {e['mouse_bin_acc']:.3f} | move acc {e['mouse_move_acc']:.3f} "
          f"({e['mouse_moves']} 个运动帧;恒中心 bin 的基率解 move acc = 0)")
    ok = e["kb_bal_acc"] > 0.6 or e["mouse_move_acc"] > 0.2
    print(f"=> {'✅ 世界模型从画面里读出了动作信息' if ok else '⚠ 动作信息尚不显著(欠训练/调 α/数据太少)'}")

    if use_wandb:
        wandb.log({f"eval/{k}": v for k, v in e.items()})
        wandb.summary["kb_bal_acc"] = e["kb_bal_acc"]
        wandb.summary["mouse_move_acc"] = e["mouse_move_acc"]
        wandb.summary["eval_pred"] = e["pred"]
        wandb.finish()


if __name__ == "__main__":
    main()
