"""VPT 动作轨迹蒸馏训练(train/vpt)。

目标:把 VPT 数据里**记录的动作轨迹**(示范者 / VPT-IDM 的逐帧操作)蒸馏进世界模型的
规划头 [decode_action_plan](net/world_model.py)——DETR 式未来 K 步动作通道,
即"VPT-mini"行为策略。仓库内**没有**可加载的真 OpenAI VPT 神经网(无 minerl/权重),
所以"蒸馏 VPT 模型"= 用 VPT 轨迹做序列级行为蒸馏(soft 目标改 hard 轨迹的退化版);
若日后接入真 VPT 网,把目标换成它的逐帧动作分布(KL)即可,本脚本骨架不变。

设计要点(2026-06-14 设计对话定案):
  1. VPT **只作蒸馏目标,绝不进模型输入** ⇒ 无答案泄漏:模型在输入里看不到自己要
     预测的未来动作(规划头 query 槽是可学常数,不喂未来 act)。这是堵"平凡解逃生
     通道"的第一道闸——历史上多次栽在输入/目标同源。
  2. **VPT-影响 sidecar**(VPTBiasSidecar):一组低容量全局可学偏置,加在规划头 logits 上。
     蒸馏作用在"内容路(规划头读 slot+h)+ sidecar 偏置"**之和**上;部署/消融时**只用内容路**。
     - 容量受限(全局常数偏置、非输入函数,仅 2·bins + n_keys 个标量)⇒ 它只能平移
       **边际/基率分布**(VPT 的系统性风格偏置),所有**状态相关结构**被逼进内容路。
       这正是"加法旁路饿死主路"(见 inv-dyn 槽路解耦复盘)的自动解药:旁路太弱抢不动主路。
     - 消融旋钮(--ablate_sidecar / eval 自动算关 sidecar 的 loss):量"内容路在没有 VPT
       基率校准下学到多少"——这是 in-context 北极星要的"不靠 VPT 先验也能预测动作"判决。
     - 零初始化 ⇒ 冷启动恒等(纯内容路),VPT 偏置有利可图才被打开(与全仓零初始化约定一致)。
  3. 仍保留 **SIGReg 防坍缩**:只有蒸馏损失时,若 slot 表征坍成常数,规划头就只能输出
     边际分布——而那恰好是 sidecar 干的活 ⇒ 内容路被架空。SIGReg 钉住 z_obs 不坍,
     逼内容路携带状态相关信息。

这是"多任务统一模型"里的**第一个任务**(先单独跑通蒸馏);后续与前向预测/逆动力学/
SIGReg 等任务在同一 MinecraftWorldModel 上联合训练时,本文件的 loss 作为 L_distill 并入。

运行:
    python train/vpt/distill_vpt.py --data_dir runs/vpt_sample --epochs 50 --device cuda
"""
import argparse
import itertools
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from net.world_model import MinecraftWorldModel
from train.minecraft.vpt_dataset import VPTStreamDataset
from train.minecraft.vpt_action import CAMERA_SCALE, CAMERA_BINS, camera_to_bin
from train.minecraft.task_text import TaskTextEncoder
from blocks.primitives import SIGReg

EPS = 1e-4
ACT_DIM = 22        # 2 鼠标 + 20 键盘
N_MOUSE = 2
N_KEYS = ACT_DIM - N_MOUSE


# =====================================================================
# VPT-影响 sidecar
# =====================================================================

class VPTBiasSidecar(nn.Module):
    """VPT-影响汇:低容量全局可学偏置,吸收 VPT 的系统性基率/风格偏置,
    使内容路(规划头)保持 VPT-无关、可消融。

    容量 = 2·n_cam_bins(鼠标分箱)+ n_keys(键盘) 个标量,**全局常数、非输入函数**:
    它能移动边际动作分布,但表达不了"某状态下该按什么"——状态相关结构只能由内容路承担。
    这把"加法旁路饿死主路"反过来用:旁路弱到只够装基率,主路被逼学真东西。

    零初始化 ⇒ 起步是恒等(σ(logit)=原概率、logits+0),VPT 偏置有利可图才被打开;
    bias 的范数即"VPT 系统性影响有多大"的用量计(panel 上看 sidecar_norm)。
    """
    def __init__(self, n_keys=N_KEYS, n_cam_bins=CAMERA_BINS):
        super().__init__()
        self.mouse_bias = nn.Parameter(torch.zeros(2, n_cam_bins))
        self.kb_bias = nn.Parameter(torch.zeros(n_keys))

    def forward(self):
        return self.mouse_bias, self.kb_bias

    def norm(self):
        return (self.mouse_bias.detach().norm() + self.kb_bias.detach().norm()).item()


# =====================================================================
# 蒸馏损失(规划头内容路 + sidecar 之和 → VPT 记录轨迹)
# =====================================================================

def vpt_distill_loss(plan, mouse_bias, kb_bias, act_agg, dt, t, K, move_w=4.0):
    """VPT 动作轨迹蒸馏:查询槽 k ↔ 未来第 k+1 个转移(onset cumsum-softplus 单调 ⇒
    时间序自对齐,免匈牙利匹配,与 train_minecraft.plan_bc_loss 同口径)。

    与 plan_bc_loss 的唯一区别:键盘/鼠标在 **logits 空间**加 sidecar 偏置再算损失
    (规划头对外吐的是 sigmoid 概率,这里 torch.logit 还原成 logits;mouse_logits 本就是 logits)。
    mouse_bias / kb_bias 传 None ⇒ 纯内容路(消融基线)。

    时间锚契约:0 时刻 = 本次 forward 的 t_vec("现在"),onset/duration 单位帧。
    返回 (loss, onset_MAE, l_kb, l_mouse);t 已无未来转移时返回 None。
    """
    B, T1 = dt.shape
    n = min(K, T1 - (t + 1))
    if n <= 0:
        return None
    agg = act_agg[:, t + 1:t + 1 + n].float()                    # [B,n,A]
    fdt = dt[:, t + 1:t + 1 + n].float()                         # [B,n]
    onset_tgt = dt[:, t:t + 1].float() + torch.cat(
        [torch.zeros_like(fdt[:, :1]), fdt.cumsum(dim=1)[:, :-1]], dim=1)

    # 键盘:内容路概率 → logit + sidecar 偏置 → BCEWithLogits(数值更稳,免二次 sigmoid)
    kb_logit = torch.logit(plan["keyboard"][:, :n].float().clamp(EPS, 1 - EPS))
    if kb_bias is not None:
        kb_logit = kb_logit + kb_bias.view(1, 1, -1)
    l_kb = F.binary_cross_entropy_with_logits(kb_logit, agg[:, :, N_MOUSE:])

    # 鼠标:内容路 logits + sidecar 偏置 → 加权 CE(非中心 bin ×move_w,堵基率不动点)
    m_logits = plan["mouse_logits"][:, :n].float()               # [B,n,2,bins]
    if mouse_bias is not None:
        m_logits = m_logits + mouse_bias.view(1, 1, 2, -1)
    m_bin = camera_to_bin(agg[:, :, :N_MOUSE])                   # [B,n,2]
    ce = F.cross_entropy(m_logits.reshape(-1, m_logits.shape[-1]),
                         m_bin.reshape(-1), reduction="none")
    center = (CAMERA_BINS - 1) // 2
    w = torch.where(m_bin.reshape(-1) == center,
                    torch.ones_like(ce), torch.full_like(ce, move_w))
    l_mouse = (ce * w).sum() / w.sum()

    # 时间(onset/duration):sidecar 不碰——VPT 影响只定义在"按什么"而非"何时按"上
    onset = plan["onset"][:, :n].float()
    l_time = (F.smooth_l1_loss(onset, onset_tgt)
              + F.smooth_l1_loss(plan["duration"][:, :n].float(), fdt))
    exist_tgt = torch.zeros_like(plan["exist"], dtype=torch.float32)
    exist_tgt[:, :n] = 1.0
    l_exist = F.binary_cross_entropy(
        plan["exist"].float().clamp(EPS, 1 - EPS), exist_tgt)

    loss = l_kb + l_mouse + 0.1 * l_time + l_exist           # 0.1:时间项以帧计,压到与 CE/BCE 同量级
    return loss, (onset - onset_tgt).abs().mean().detach(), l_kb.detach(), l_mouse.detach()


# =====================================================================
# 小工具(本地实现,避免拉入 train_minecraft 的 viz/matplotlib 依赖链)
# =====================================================================

def _to_float_img(img):
    """uint8 [.,3,H,W] → float∈[0,1](归一化推迟到 GPU,PCIe 传 uint8 省 4×)。"""
    return img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()


def roll_hist(a_hist, t_hist, hv, action, dt_cur):
    """时间前进 dt_cur 帧并滚入一个刚结束的聚合动作(与 train_minecraft 同义)。

    旧条目统一变老 dt_cur 帧、新条目 age=0 有效位=1;开头空槽有效位 0
    (全零动作是合法的"没按",不能用值判空)。
    """
    B = action.shape[0]
    z1 = torch.zeros(B, 1, device=action.device, dtype=t_hist.dtype)
    return (torch.cat([a_hist[:, 1:], action.unsqueeze(1)], dim=1),
            torch.cat([t_hist[:, 1:] + dt_cur.unsqueeze(1), z1], dim=1),
            torch.cat([hv[:, 1:], torch.ones_like(z1)], dim=1))


def _gpu_util():
    try:
        return torch.cuda.utilization()
    except Exception:
        return None


# =====================================================================
# 训练 / 评估
# =====================================================================

def _run_distill_sequence(model, sidecar, sigreg, batch_dev, device, k_bptt,
                          beta_sigreg, move_w, ablate, amp_dev, use_amp, scaler, acc):
    """对一个 batch 做截断 BPTT 前向+反向,只算 **蒸馏 + SIGReg** 损失。

    teacher forcing:每步感知输入 z_obs(t) 来自真画面;跨步记忆走 h(窗口内带梯度,
    窗口边界 detach)。规划头从 (z_ref, h, 历史动作, 当前区间动作, dt, t) 预测未来 K 步
    动作轨迹——注意它**不**接收未来动作,只读"现在及过去",无泄漏。
    """
    img, t_vec = batch_dev["img"], batch_dev["t_vec"]
    act_seq, act_agg, dt = batch_dev["act_seq"], batch_dev["act_agg"], batch_dev["dt"]
    task_emb = batch_dev.get("task_emb")
    B, T = img.shape[0], img.shape[1]
    h = torch.zeros(B, 1, model.d, device=device)
    a_hist = torch.zeros(B, model.J, ACT_DIM, device=device)
    t_hist = torch.zeros(B, model.J, device=device)
    hv = torch.zeros(B, model.J, device=device)
    n_win = -(-(T - 1) // k_bptt)        # ceil:窗口数,SIGReg 权重尺度归一

    with torch.autocast(device_type=amp_dev, enabled=use_amp):
        feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))   # 冻结骨干,整条一次
    feats = feats.view(B, T, *feats.shape[-2:])
    mb, kb = sidecar()

    for w0 in range(0, T - 1, k_bptt):
        w1 = min(w0 + k_bptt, T - 1)
        k = w1 - w0
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            z_obs = model.encode_obs(
                feats=feats[:, w0:w1].reshape(B * k, *feats.shape[-2:])
            ).view(B, k, model.N, model.d)

        accum = torch.zeros((), device=device)
        for i in range(k):
            t = w0 + i
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                out = model(z_obs[:, i], h, a_hist, act_seq[:, t], dt[:, t], t_vec[:, t],
                            t_hist=t_hist, hist_valid=hv, task_emb=task_emb)
            # 历史在 forward **之后**滚入当前聚合动作(历史=严格过去)
            a_hist, t_hist, hv = roll_hist(a_hist, t_hist, hv, act_agg[:, t], dt[:, t])
            res = vpt_distill_loss(out["action_plan"],
                                   None if ablate else mb, None if ablate else kb,
                                   act_agg, dt, t, model.K, move_w)
            if res is not None:
                accum = accum + res[0] / (T - 1)
                acc["distill"] += res[0].detach(); acc["onset_mae"] += res[1]
                acc["kb"] += res[2]; acc["mouse"] += res[3]; acc["n"] += 1
            h = out["h_next"]                # 窗口内保留梯度(截断 BPTT)

        # SIGReg:钉在在线感知编码 z_obs 上(坍缩发生处);除以 n_win 使总权重不随长度漂移
        l_sig = sigreg(z_obs.float().permute(2, 1, 0, 3).reshape(model.N, k * B, model.d))
        win_loss = accum + (beta_sigreg / n_win) * l_sig
        scaler.scale(win_loss).backward()
        acc["loss"] += win_loss.detach(); acc["sigreg"] += l_sig.detach(); acc["win"] += 1
        h = h.detach()                       # 窗口边界截断


def train_epoch(model, sidecar, sigreg, data_iter, opt, scaler, device, steps, k_bptt,
                beta_sigreg, move_w, ablate, text_enc, amp_dev, use_amp):
    model.train()
    acc = {k: torch.zeros((), device=device) for k in
           ["loss", "distill", "onset_mae", "kb", "mouse", "sigreg"]}
    acc["n"] = acc["win"] = 0
    gpu_samples, n_batches = [], 0

    for batch in itertools.islice(data_iter, steps):
        batch_dev = {
            "img": _to_float_img(batch["img"].to(device, non_blocking=True)),
            "act_seq": batch["act_seq"].to(device, non_blocking=True),
            "act_agg": batch["act_agg"].to(device, non_blocking=True),
            "dt": batch["dt"].to(device, non_blocking=True),
            "t_vec": batch["t_vec"].to(device, non_blocking=True),
            "task_emb": (text_enc.encode(batch["task_text"]).to(device)
                         if text_enc is not None else None),
        }
        opt.zero_grad(set_to_none=True)
        _run_distill_sequence(model, sidecar, sigreg, batch_dev, device, k_bptt,
                              beta_sigreg, move_w, ablate, amp_dev, use_amp, scaler, acc)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(sidecar.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        model.ema_update()
        n_batches += 1
        u = _gpu_util()
        if u is not None:
            gpu_samples.append(u)

    nc = max(acc["n"], 1); wc = max(acc["win"], 1)
    return {
        "loss": (acc["loss"] / wc).item(),
        "distill": (acc["distill"] / nc).item(),
        "onset_mae": (acc["onset_mae"] / nc).item(),
        "kb": (acc["kb"] / nc).item(),
        "mouse": (acc["mouse"] / nc).item(),
        "sigreg": (acc["sigreg"] / wc).item(),
        "sidecar_norm": sidecar.norm(),
        "batches": n_batches,
        "gpu_util": (sum(gpu_samples) / len(gpu_samples)) if gpu_samples else None,
    }


@torch.no_grad()
def evaluate(model, sidecar, loader, device, steps, k_bptt, move_w, amp_dev, use_amp):
    """holdout 蒸馏评估:量内容路+sidecar 的轨迹拟合,并算 **消融差**(关 sidecar 的 loss)。

    sidecar_gain = distill_ablate − distill ≥ 0:VPT 系统性基率校准带来的增益,
    = "VPT 影响"的量;它越小,说明内容路已自给自足(越接近 VPT-无关,北极星越近)。
    """
    model.eval()
    mb, kb = sidecar()
    agg = {k: 0.0 for k in ["distill", "distill_abl", "onset_mae", "kb", "mouse"]}
    nstep = 0
    for batch in itertools.islice(loader, steps):
        img = _to_float_img(batch["img"].to(device))
        act_seq = batch["act_seq"].to(device); act_agg = batch["act_agg"].to(device)
        dt = batch["dt"].to(device); t_vec = batch["t_vec"].to(device)
        task_emb = batch.get("task_emb")
        task_emb = task_emb.to(device) if task_emb is not None else None
        B, T = img.shape[0], img.shape[1]
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
        feats = feats.view(B, T, *feats.shape[-2:])
        h = torch.zeros(B, 1, model.d, device=device)
        a_hist = torch.zeros(B, model.J, ACT_DIM, device=device)
        t_hist = torch.zeros(B, model.J, device=device)
        hv = torch.zeros(B, model.J, device=device)
        for t in range(T - 1):
            z_ref = model.encode_obs(feats=feats[:, t]).view(B, model.N, model.d)
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                out = model(z_ref, h, a_hist, act_seq[:, t], dt[:, t], t_vec[:, t],
                            t_hist=t_hist, hist_valid=hv, task_emb=task_emb)
            a_hist, t_hist, hv = roll_hist(a_hist, t_hist, hv, act_agg[:, t], dt[:, t])
            res = vpt_distill_loss(out["action_plan"], mb, kb, act_agg, dt, t, model.K, move_w)
            res_abl = vpt_distill_loss(out["action_plan"], None, None, act_agg, dt, t, model.K, move_w)
            if res is not None:
                agg["distill"] += res[0].item(); agg["onset_mae"] += res[1].item()
                agg["kb"] += res[2].item(); agg["mouse"] += res[3].item()
                agg["distill_abl"] += res_abl[0].item()
                nstep += 1
            h = out["h_next"]
    nstep = max(nstep, 1)
    out = {k: v / nstep for k, v in agg.items()}
    out["sidecar_gain"] = out["distill_abl"] - out["distill"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample", help="训练数据目录(.mp4+.jsonl)")
    ap.add_argument("--holdout_dir", default=None, help="独立固定 holdout 目录(滚动目录模式);"
                    "不设则按文件名从 data_dir 扣末 holdout_n 个")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame_skip", type=int, default=8, help="可变预测跨度上限 Δt~U{1..skip}")
    ap.add_argument("--img_size", type=int, default=128, help="训练分辨率(0=原始)")
    ap.add_argument("--camera_scale", type=float, default=None, help="鼠标归一化尺度(None=默认 10)")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader 进程数(None=自动)")
    ap.add_argument("--clip_cache", type=int, default=4)
    ap.add_argument("--clip_refresh", type=int, default=256)
    ap.add_argument("--holdout_n", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--ckpt_dir", default="runs/vpt_distill_ckpt")
    ap.add_argument("--encoder", choices=["dinov3", "dinov2", "mock"], default="dinov3")
    ap.add_argument("--encoder_weights", default=None, help="覆盖骨干 HF repo id")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--N", type=int, default=16, help="实体槽数")
    ap.add_argument("--K", type=int, default=5, help="动作查询数(= 蒸馏的未来转移步数)")
    ap.add_argument("--J", type=int, default=8, help="历史动作长度")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mouse_move_w", type=float, default=4.0, help="鼠标非中心 bin 的 CE 权重")
    ap.add_argument("--beta_sigreg", type=float, default=0.1, help="SIGReg 防坍缩权重(施加在 z_obs)")
    ap.add_argument("--ema_decay", type=float, default=0.99)
    ap.add_argument("--k_bptt", type=int, default=4, help="截断 BPTT 窗口")
    ap.add_argument("--no_cosine", action="store_true", help="关闭余弦 lr 衰减")
    ap.add_argument("--ablate_sidecar", action="store_true",
                    help="训练时关 sidecar(纯内容路)——对照实验:不给 VPT 基率校准时内容路学得如何")
    ap.add_argument("--text_encoder", choices=["minilm", "mock", "none"], default="minilm")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="minecraft-vpt-distill")
    ap.add_argument("--wandb_run", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_cuda = str(dev).startswith("cuda")
    amp_dev = "cuda" if is_cuda else "cpu"
    use_amp = is_cuda and not args.no_amp
    print(f"=== VPT 动作轨迹蒸馏(plan-head + VPT-影响 sidecar)| device={dev} | amp={use_amp} "
          f"| sidecar={'OFF(消融)' if args.ablate_sidecar else 'ON'} ===")

    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        except Exception as ex:
            print(f"[wandb] 初始化失败,关闭远程记录: {ex}")
            use_wandb = False

    if args.holdout_dir:
        if not os.path.isdir(args.holdout_dir) or not any(
                f.endswith(".mp4") for f in os.listdir(args.holdout_dir)):
            print(f"[!] holdout 目录 '{args.holdout_dir}' 里没有 .mp4。"); sys.exit(1)
    elif not os.path.isdir(args.data_dir) or not any(
            f.endswith(".mp4") for f in os.listdir(args.data_dir)):
        print(f"[!] 数据目录 '{args.data_dir}' 里没有 .mp4。先准备数据。"); sys.exit(1)

    text_enc = None if args.text_encoder == "none" else \
        TaskTextEncoder(args.text_encoder, device="cpu")
    img_size = args.img_size if args.img_size > 0 else None
    cam_scale = args.camera_scale if args.camera_scale is not None else CAMERA_SCALE
    n_workers = args.workers if args.workers is not None \
        else max(2, min(8, (os.cpu_count() or 2) - 1))
    train_split = None if args.holdout_dir else "train"
    hold_dir = args.holdout_dir or args.data_dir
    hold_split = None if args.holdout_dir else "holdout"
    os.makedirs(args.data_dir, exist_ok=True)

    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps, seed=args.seed,
                          img_size=img_size, camera_scale=cam_scale,
                          frame_skip=args.frame_skip, split=train_split,
                          holdout_n=args.holdout_n, clip_cache=args.clip_cache,
                          clip_refresh=args.clip_refresh)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=n_workers, pin_memory=is_cuda,
                        persistent_workers=(n_workers > 0),
                        prefetch_factor=(2 if n_workers > 0 else None))
    data_iter = iter(loader)

    eval_ds = VPTStreamDataset(hold_dir, seq_len=args.seq_len, fps=args.fps,
                               seed=args.seed + 555, img_size=img_size, camera_scale=cam_scale,
                               frame_skip=args.frame_skip, split=hold_split,
                               holdout_n=args.holdout_n, clip_cache=4)
    eval_bs = min(args.batch, 64)
    eval_batches = []

    def _get_eval_batches():
        if not eval_batches:
            it = iter(DataLoader(eval_ds, batch_size=eval_bs, num_workers=min(4, n_workers)))
            for _ in range(4):
                b = next(it)
                if text_enc is not None:
                    b["task_emb"] = text_enc.encode(b["task_text"])
                eval_batches.append(b)
            del it
            print(f"  [eval] 固定评估集已采集:4×{eval_bs} 序列(此后复用)")
        return eval_batches

    model = MinecraftWorldModel(d=args.d, N=args.N, K=args.K, J=args.J, act_dim=ACT_DIM,
                                n_cam_bins=CAMERA_BINS, ema_decay=args.ema_decay,
                                max_skip=args.frame_skip, encoder=args.encoder,
                                encoder_weights=args.encoder_weights).to(dev)
    sidecar = VPTBiasSidecar(n_keys=N_KEYS, n_cam_bins=CAMERA_BINS).to(dev)

    sigreg = SIGReg(knots=17, num_proj=512).to(dev)
    params = [p for p in model.parameters() if p.requires_grad] + list(sidecar.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = None if args.no_cosine else torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train / 1e6:.1f}M(含 sidecar {sum(p.numel() for p in sidecar.parameters())} 个)"
          f" | batch={args.batch} workers={n_workers} K={args.K} frame_skip={args.frame_skip}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_score, best_ep = None, -1
    best_path = os.path.join(args.ckpt_dir, f"best_{args.wandb_run or 'run'}.pt")

    def _save_best(ep, score):
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
        torch.save({"model": sd, "sidecar": sidecar.state_dict(), "epoch": ep,
                    "metric": "eval_distill", "score": score, "encoder": args.encoder,
                    "camera_scale": cam_scale, "args": vars(args)}, best_path)

    for ep in range(args.epochs):
        r = train_epoch(model, sidecar, sigreg, data_iter, opt, scaler, dev,
                        args.steps_per_epoch, args.k_bptt, args.beta_sigreg,
                        args.mouse_move_w, args.ablate_sidecar, text_enc, amp_dev, use_amp)
        if sched is not None:
            sched.step()
        if use_wandb:
            _wb = {k: r[k] for k in ("loss", "distill", "kb", "mouse", "onset_mae",
                                     "sigreg", "sidecar_norm")}
            if r.get("gpu_util") is not None:
                _wb["gpu_util"] = r["gpu_util"]
            wandb.log(_wb, step=ep)
        if ep % args.log_every == 0 or ep == args.epochs - 1:
            gpu = f" | gpu {r['gpu_util']:.0f}%" if r.get("gpu_util") is not None else ""
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | distill {r['distill']:.3f} "
                  f"(kb {r['kb']:.3f} mouse {r['mouse']:.3f} onset±{r['onset_mae']:.1f}f) "
                  f"| sig {r['sigreg']:.2f} | sidecar‖·‖ {r['sidecar_norm']:.3f}{gpu}")
        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
            ev = evaluate(model, sidecar, _get_eval_batches(), dev, 4, args.k_bptt,
                          args.mouse_move_w, amp_dev, use_amp)
            print(f"  [eval] distill {ev['distill']:.3f} | 消融(关 sidecar) {ev['distill_abl']:.3f} "
                  f"| sidecar_gain {ev['sidecar_gain']:.3f}(↓=内容路越自给/越 VPT-无关) "
                  f"| kb {ev['kb']:.3f} mouse {ev['mouse']:.3f}")
            if use_wandb:
                wandb.log({f"eval/{k}": v for k, v in ev.items()}, step=ep)
            if best_score is None or ev["distill"] < best_score:
                best_score, best_ep = ev["distill"], ep
                _save_best(ep, ev["distill"])
                print(f"  [best] eval_distill={ev['distill']:.4f} @ep{ep} → {best_path}")

    print(f"\n=== 完成。best eval_distill={best_score:.4f} @ep{best_ep} → {best_path} ===")
    if use_wandb:
        wandb.summary["best_eval_distill"] = best_score
        wandb.summary["best_epoch"] = best_ep
        wandb.finish()


if __name__ == "__main__":
    main()
